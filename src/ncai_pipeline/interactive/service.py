from __future__ import annotations

import io
import json
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from ncai_pipeline.config import EnvironmentSettings, ProviderSettings
from ncai_pipeline.parsing.normalizer import normalize_provider_payload
from ncai_pipeline.parsing.validators import ensure_page_reviews, load_corrected_document
from ncai_pipeline.pipeline.stages import consolidate_text
from ncai_pipeline.providers.base import ExtractionProvider
from ncai_pipeline.providers.gemini_provider import GeminiProvider
from ncai_pipeline.review.html_report import render_review_report
from ncai_pipeline.review.dry_run_policy import corrected_document_is_dry_run
from ncai_pipeline.review.handwriting import block_handwriting_status
from ncai_pipeline.schemas.common import BoundingBox, PipelineStatus, ReviewStatus
from ncai_pipeline.schemas.document import DocumentManifest
from ncai_pipeline.schemas.extraction import NormalizedExtraction, RawExtractionRecord
from ncai_pipeline.schemas.review import (
    CorrectedDocument,
    PageReviewState,
    PageVerificationStatus,
    ReviewRecord,
)
from ncai_pipeline.storage.locks import document_lock
from ncai_pipeline.storage.paths import DataPaths
from ncai_pipeline.storage.repository import FileRepository
from ncai_pipeline.utils.time import utc_now, utc_now_iso


@dataclass
class QueueDocument:
    document_id: str
    source_path: str
    pipeline_status: str
    review_status: str
    page_count: int
    has_review_bundle: bool
    is_dry_run: bool
    manifest_path: str
    review_record_path: str | None = None
    correction_path: str | None = None
    report_path: str | None = None
    error_details: str | None = None


@dataclass
class PageFragment:
    fragment_id: str
    source_kind: str
    label: str
    text: str
    bbox: BoundingBox | None = None
    confidence_notes: str | None = None
    handwritten_auto: bool = False
    handwritten_manual: bool = False
    handwritten_effective: bool = False
    handwritten_source: str = "none"
    field_name: str = ""
    anchor_id: str = ""

    @property
    def has_bbox(self) -> bool:
        if self.bbox is None:
            return False
        return None not in {self.bbox.left, self.bbox.top, self.bbox.right, self.bbox.bottom}


@dataclass
class PageWorkspace:
    mode: str
    fragments: list[PageFragment]
    warning_count: int
    crop_count: int
    handwritten_count: int


@dataclass
class PageSummary:
    page_number: int
    review_status: str
    warning_count: int
    fragment_count: int
    fragment_mode: str
    handwritten_count: int


@dataclass
class DocumentContext:
    queue_entry: QueueDocument
    corrected: CorrectedDocument
    review_record: ReviewRecord
    manifest: dict[str, Any]
    raw_records: list[RawExtractionRecord]
    selected_page_number: int
    selected_page_index: int
    page_image_path: str
    page_image_width: int
    page_image_height: int
    page_review: PageReviewState
    selected_page: NormalizedExtraction
    selected_raw_page: NormalizedExtraction
    selected_raw_record: RawExtractionRecord | None
    selected_page_workspace: PageWorkspace
    selected_raw_workspace: PageWorkspace
    page_summaries: list[PageSummary]
    auto_heal_note: str | None
    latest_run_summary: dict[str, Any] | None
    latest_export_summary: dict[str, Any] | None
    command_hints: dict[str, str]


class InteractiveReviewService:
    def __init__(
        self,
        data_paths: DataPaths,
        repository: FileRepository,
        parser_provider: ExtractionProvider | None = None,
        report_title: str = "NCAI OCR Review",
        storage_mode: str = "local",
        locks_enabled: bool = False,
        lock_stale_after_minutes: int = 240,
    ) -> None:
        self.data_paths = data_paths
        self.repository = repository
        self.parser_provider = parser_provider or GeminiProvider(ProviderSettings(), EnvironmentSettings())
        self.report_title = report_title
        self.storage_mode = storage_mode
        self.locks_enabled = locks_enabled
        self.lock_stale_after_minutes = lock_stale_after_minutes

    @contextmanager
    def _document_write_lock(self, document_id: str) -> Iterator[None]:
        with document_lock(
            self.data_paths.locks_dir,
            document_id,
            self.repository,
            stale_after_minutes=self.lock_stale_after_minutes,
            enabled=self.storage_mode == "nas" and self.locks_enabled,
        ):
            yield

    def load_latest_run_summary(self) -> dict[str, Any] | None:
        path = self.data_paths.status_dir / "latest_run.json"
        if path.exists():
            return self.repository.read_json(path)
        run_manifests = sorted(
            self.data_paths.run_manifests_dir.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if run_manifests:
            return self.repository.read_json(run_manifests[0])
        legacy_manifests = sorted(
            self.data_paths.status_dir.glob("*/run_manifest.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if legacy_manifests:
            return self.repository.read_json(legacy_manifests[0])
        return None

    def load_latest_export_summary(self) -> dict[str, Any] | None:
        path = self.data_paths.exports_dir / "export_manifest.json"
        if not path.exists():
            path = self.data_paths.export_target_dir / "export_manifest.json"
        if not path.exists():
            return None
        return self.repository.read_json(path)

    def get_dashboard_state(
        self,
        pipeline_status: str = "all",
        review_status: str = "all",
        include_dry_run: bool = True,
    ) -> dict[str, Any]:
        queue = self.list_documents(
            pipeline_status=pipeline_status,
            review_status=review_status,
            include_dry_run=include_dry_run,
        )
        return {
            "pipeline_counts": self.pipeline_status_counts(),
            "review_counts": self.review_status_counts(),
            "latest_run": self.load_latest_run_summary(),
            "latest_export": self.load_latest_export_summary(),
            "documents": queue,
            "command_hints": self.command_hints(),
        }

    def pipeline_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in sorted(self.data_paths.manifests_dir.glob("*/manifest.json")):
            manifest = self.repository.read_json(path)
            counts[manifest["status"]] = counts.get(manifest["status"], 0) + 1
        return counts

    def review_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in sorted(self.data_paths.status_dir.glob("*/review_record.json")):
            record = ReviewRecord.model_validate(self.repository.read_json(path))
            counts[record.review_status.value] = counts.get(record.review_status.value, 0) + 1
        return counts

    def list_documents(
        self,
        pipeline_status: str = "all",
        review_status: str = "all",
        include_dry_run: bool = True,
    ) -> list[QueueDocument]:
        review_by_document = self._load_review_record_map()
        queue: list[QueueDocument] = []
        for path in sorted(self.data_paths.manifests_dir.glob("*/manifest.json")):
            manifest = self.repository.read_json(path)
            document_id = manifest["document_id"]
            review_record = review_by_document.get(document_id)
            is_dry_run = bool(review_record and review_record.is_dry_run)
            correction_path: str | None = None
            report_path: str | None = None
            review_record_path: str | None = None
            effective_review_status = "pending"
            if review_record is not None:
                review_record_path = str(self.data_paths.status_dir / document_id / "review_record.json")
                correction_path = review_record.corrected_output_path
                report_path = review_record.report_path
                effective_review_status = review_record.review_status.value
                corrected = load_corrected_document(Path(review_record.corrected_output_path), self.repository)
                is_dry_run = corrected_document_is_dry_run(corrected)
                effective_review_status = corrected.approval_metadata.review_status.value

            if pipeline_status != "all" and manifest["status"] != pipeline_status:
                continue
            if review_status != "all" and effective_review_status != review_status:
                continue
            if not include_dry_run and is_dry_run:
                continue

            queue.append(
                QueueDocument(
                    document_id=document_id,
                    source_path=manifest["source_path"],
                    pipeline_status=manifest["status"],
                    review_status=effective_review_status,
                    page_count=manifest.get("page_count") or 0,
                    has_review_bundle=review_record is not None,
                    is_dry_run=is_dry_run,
                    manifest_path=str(path),
                    review_record_path=review_record_path,
                    correction_path=correction_path,
                    report_path=report_path,
                    error_details=manifest.get("error_details"),
                )
            )
        return queue

    def get_document_context(self, document_id: str, page_number: int | None = None) -> DocumentContext:
        manifest_path = self.data_paths.manifests_dir / document_id / "manifest.json"
        review_record_path = self.data_paths.status_dir / document_id / "review_record.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for document: {document_id}")
        if not review_record_path.exists():
            raise FileNotFoundError(f"Review bundle not found for document: {document_id}")

        manifest = self.repository.read_json(manifest_path)
        review_record = ReviewRecord.model_validate(self.repository.read_json(review_record_path))
        corrected = load_corrected_document(Path(review_record.corrected_output_path), self.repository)
        raw_records = self._load_raw_records(review_record)
        healed_pages = self._auto_heal_placeholder_pages(document_id, corrected, review_record, raw_records)
        manifest_model = self._sync_manifest_review_state(
            document_id,
            corrected,
            manifest_payload=manifest,
        )
        manifest = manifest_model.model_dump(mode="json")

        queue_entry = QueueDocument(
            document_id=document_id,
            source_path=manifest["source_path"],
            pipeline_status=manifest["status"],
            review_status=corrected.approval_metadata.review_status.value,
            page_count=manifest.get("page_count") or len(corrected.extracted_pages),
            has_review_bundle=True,
            is_dry_run=corrected_document_is_dry_run(corrected),
            manifest_path=str(manifest_path),
            review_record_path=str(review_record_path),
            correction_path=review_record.corrected_output_path,
            report_path=review_record.report_path,
            error_details=manifest.get("error_details"),
        )
        selected_page_index = self._resolve_page_index(corrected, page_number)
        selected_page = corrected.extracted_pages[selected_page_index]
        selected_page_number = selected_page.page_number or (selected_page_index + 1)
        selected_raw_record = self._raw_record_for_page(raw_records, selected_page_number)
        selected_raw_page = self._load_raw_page(document_id, selected_page_number, raw_records, fallback_page=selected_page)
        page_image_path = self.page_image_path(document_id, selected_page_number)
        image_width, image_height = self.page_image_size(document_id, selected_page_number)
        page_review = self._page_review_by_number(corrected)[selected_page_number]

        page_summaries: list[PageSummary] = []
        reviews_by_page = self._page_review_by_number(corrected)
        for index, page in enumerate(corrected.extracted_pages, start=1):
            effective_page_number = page.page_number or index
            page_workspace = self.build_page_workspace(page)
            page_summaries.append(
                PageSummary(
                    page_number=effective_page_number,
                    review_status=reviews_by_page[effective_page_number].status.value,
                    warning_count=len(self.filter_display_warnings(page.warnings)),
                    fragment_count=len(page_workspace.fragments),
                    fragment_mode=page_workspace.mode,
                    handwritten_count=page_workspace.handwritten_count,
                )
            )

        auto_heal_note = None
        if selected_page_number in healed_pages:
            auto_heal_note = "Recovered parsed content from the stored raw provider response for this page."

        return DocumentContext(
            queue_entry=queue_entry,
            corrected=corrected,
            review_record=review_record,
            manifest=manifest,
            raw_records=raw_records,
            selected_page_number=selected_page_number,
            selected_page_index=selected_page_index,
            page_image_path=page_image_path,
            page_image_width=image_width,
            page_image_height=image_height,
            page_review=page_review,
            selected_page=selected_page,
            selected_raw_page=selected_raw_page,
            selected_raw_record=selected_raw_record,
            selected_page_workspace=self.build_page_workspace(selected_page),
            selected_raw_workspace=self.build_page_workspace(selected_raw_page),
            page_summaries=page_summaries,
            auto_heal_note=auto_heal_note,
            latest_run_summary=self.load_latest_run_summary(),
            latest_export_summary=self.load_latest_export_summary(),
            command_hints=self.command_hints(document_id),
        )

    def save_page_edits(
        self,
        document_id: str,
        page_number: int,
        *,
        full_text: str,
        structured_content_text: str,
        reviewer: str = "",
        bbox_payload_text: str = "",
    ) -> None:
        corrected, review_record = self._load_corrected_bundle(document_id)
        page_index = self._resolve_page_index(corrected, page_number)
        extraction = corrected.extracted_pages[page_index]
        previous_consolidated = consolidate_text(corrected.extracted_pages)
        bbox_payload = self._parse_bbox_payload(bbox_payload_text)
        bbox_changed = self._apply_bbox_updates(
            document_id,
            page_number,
            extraction,
            workspace=self.build_page_workspace(extraction),
            bbox_payload=bbox_payload,
        )
        extraction.full_text = full_text
        extraction.structured_content = self._parse_json_object(structured_content_text, field_name="structured_content")
        extraction.language = extraction.structured_content.get("language") or extraction.language
        self._sync_target_output(corrected)
        if not corrected.consolidated_text or corrected.consolidated_text == previous_consolidated:
            corrected.consolidated_text = consolidate_text(corrected.extracted_pages)
            corrected.target_output["consolidated_text"] = corrected.consolidated_text
        self._mark_corrected(corrected, reviewer)
        self._sync_review_metadata(review_record, reviewer=reviewer)
        if bbox_changed and corrected.approval_metadata.review_status == ReviewStatus.APPROVED:
            corrected.approval_metadata.review_status = ReviewStatus.IN_REVIEW
            review_record.review_status = ReviewStatus.IN_REVIEW
        else:
            self._set_in_review(corrected, review_record)
        self._save_corrected_bundle(review_record, corrected)
        self._sync_manifest_review_state(document_id, corrected)

    def save_page_bboxes(
        self,
        document_id: str,
        page_number: int,
        *,
        bbox_payload_text: str,
        reviewer: str = "",
    ) -> None:
        corrected, review_record = self._load_corrected_bundle(document_id)
        page_index = self._resolve_page_index(corrected, page_number)
        extraction = corrected.extracted_pages[page_index]
        bbox_payload = self._parse_bbox_payload(bbox_payload_text)
        bbox_changed = self._apply_bbox_updates(
            document_id,
            page_number,
            extraction,
            workspace=self.build_page_workspace(extraction),
            bbox_payload=bbox_payload,
        )
        self._sync_target_output(corrected)
        self._mark_corrected(corrected, reviewer)
        self._sync_review_metadata(review_record, reviewer=reviewer)
        if bbox_changed:
            if corrected.approval_metadata.review_status == ReviewStatus.APPROVED:
                corrected.approval_metadata.review_status = ReviewStatus.IN_REVIEW
                review_record.review_status = ReviewStatus.IN_REVIEW
            else:
                self._set_in_review(corrected, review_record)
        self._save_corrected_bundle(review_record, corrected)
        self._sync_manifest_review_state(document_id, corrected)

    def commit_page_review(
        self,
        document_id: str,
        page_number: int,
        *,
        fragment_texts: list[str],
        fragment_handwritten_flags: list[bool] | None = None,
        structured_content_text: str,
        reviewer: str = "",
        status: PageVerificationStatus = PageVerificationStatus.PENDING,
        comments: str = "",
        bbox_payload_text: str = "",
    ) -> None:
        corrected, review_record = self._load_corrected_bundle(document_id)
        page_index = self._resolve_page_index(corrected, page_number)
        extraction = corrected.extracted_pages[page_index]
        previous_consolidated = consolidate_text(corrected.extracted_pages)
        workspace = self.build_page_workspace(extraction)
        bbox_payload = self._parse_bbox_payload(bbox_payload_text)
        bbox_changed = self._apply_bbox_updates(
            document_id,
            page_number,
            extraction,
            workspace=workspace,
            bbox_payload=bbox_payload,
        )
        fragment_values = [
            fragment_texts[index] if index < len(fragment_texts) else fragment.text
            for index, fragment in enumerate(workspace.fragments)
        ]

        if workspace.mode == "blocks":
            for index, block in enumerate(extraction.blocks):
                block.text = fragment_values[index]
                if fragment_handwritten_flags is not None and index < len(fragment_handwritten_flags):
                    block.handwritten_manual = bool(fragment_handwritten_flags[index])
            extraction.full_text = "\n\n".join(value for value in fragment_values if value.strip())
        elif workspace.mode == "lines":
            for index, line in enumerate(extraction.lines):
                line.text = fragment_values[index]
            extraction.full_text = "\n".join(value for value in fragment_values if value.strip())
        else:
            extraction.full_text = fragment_values[0] if fragment_values else extraction.full_text

        extraction.structured_content = self._parse_json_object(structured_content_text, field_name="structured_content")
        extraction.language = extraction.structured_content.get("language") or extraction.language

        self._sync_target_output(corrected)
        if not corrected.consolidated_text or corrected.consolidated_text == previous_consolidated:
            corrected.consolidated_text = consolidate_text(corrected.extracted_pages)
            corrected.target_output["consolidated_text"] = corrected.consolidated_text

        review_state = self._page_review_by_number(corrected)[page_number]
        review_state.status = status
        review_state.reviewer = reviewer or review_state.reviewer
        review_state.reviewed_at = utc_now()
        review_state.comments = comments

        self._mark_corrected(corrected, reviewer)
        self._sync_review_metadata(review_record, reviewer=reviewer, comments=comments)
        if bbox_changed and corrected.approval_metadata.review_status == ReviewStatus.APPROVED:
            corrected.approval_metadata.review_status = ReviewStatus.IN_REVIEW
            review_record.review_status = ReviewStatus.IN_REVIEW
        elif status != PageVerificationStatus.VERIFIED and corrected.approval_metadata.review_status == ReviewStatus.APPROVED:
            corrected.approval_metadata.review_status = ReviewStatus.IN_REVIEW
            review_record.review_status = ReviewStatus.IN_REVIEW
        elif corrected.approval_metadata.review_status in {ReviewStatus.PENDING, ReviewStatus.NEEDS_RETRY}:
            self._set_in_review(corrected, review_record)
        self._save_corrected_bundle(review_record, corrected)
        self._sync_manifest_review_state(document_id, corrected)

    def set_page_verification(
        self,
        document_id: str,
        page_number: int,
        *,
        status: PageVerificationStatus,
        reviewer: str = "",
        comments: str = "",
    ) -> None:
        corrected, review_record = self._load_corrected_bundle(document_id)
        review_state = self._page_review_by_number(corrected)[page_number]
        review_state.status = status
        review_state.reviewer = reviewer or review_state.reviewer
        review_state.reviewed_at = utc_now()
        review_state.comments = comments or review_state.comments
        self._mark_corrected(corrected, reviewer)
        self._sync_review_metadata(review_record, reviewer=reviewer, comments=comments)
        if status != PageVerificationStatus.VERIFIED and corrected.approval_metadata.review_status == ReviewStatus.APPROVED:
            corrected.approval_metadata.review_status = ReviewStatus.IN_REVIEW
            review_record.review_status = ReviewStatus.IN_REVIEW
        elif corrected.approval_metadata.review_status in {ReviewStatus.PENDING, ReviewStatus.NEEDS_RETRY}:
            self._set_in_review(corrected, review_record)
        self._save_corrected_bundle(review_record, corrected)
        self._sync_manifest_review_state(document_id, corrected)

    def save_document_edits(
        self,
        document_id: str,
        *,
        consolidated_text: str,
        target_output_text: str,
        reviewer: str = "",
        comments: str = "",
    ) -> None:
        corrected, review_record = self._load_corrected_bundle(document_id)
        corrected.consolidated_text = consolidated_text
        corrected.target_output = self._parse_json_object(target_output_text, field_name="target_output")
        if comments:
            corrected.approval_metadata.comments = comments
        self._mark_corrected(corrected, reviewer)
        self._sync_review_metadata(review_record, reviewer=reviewer, comments=comments)
        self._set_in_review(corrected, review_record)
        self._save_corrected_bundle(review_record, corrected)
        self._sync_manifest_review_state(document_id, corrected)

    def set_document_review_status(
        self,
        document_id: str,
        *,
        status: ReviewStatus,
        reviewer: str = "",
        comments: str = "",
        require_verified_pages: bool = False,
        auto_verify_pages_on_approve: bool = False,
    ) -> None:
        corrected, review_record = self._load_corrected_bundle(document_id)
        if status == ReviewStatus.APPROVED:
            if require_verified_pages and not self.all_pages_verified(corrected):
                raise ValueError("All pages must be verified before approving the document.")
            if auto_verify_pages_on_approve:
                for page_review in corrected.page_reviews:
                    page_review.status = PageVerificationStatus.VERIFIED
                    page_review.reviewer = reviewer or page_review.reviewer
                    page_review.reviewed_at = utc_now()
                    if not page_review.comments:
                        page_review.comments = "Verified during CLI approval."
            if not (reviewer or corrected.approval_metadata.reviewer):
                raise ValueError("Approved documents must include a reviewer.")
        corrected.approval_metadata.review_status = status
        corrected.approval_metadata.reviewer = reviewer or corrected.approval_metadata.reviewer
        corrected.approval_metadata.reviewed_at = utc_now()
        corrected.approval_metadata.comments = comments or corrected.approval_metadata.comments
        self._sync_review_metadata(review_record, reviewer=reviewer, comments=comments, status=status)
        self._mark_corrected(corrected, reviewer)
        self._save_corrected_bundle(review_record, corrected)
        self._sync_manifest_review_state(document_id, corrected)

    def all_pages_verified(self, corrected: CorrectedDocument) -> bool:
        corrected = ensure_page_reviews(corrected)
        if not corrected.page_reviews:
            return False
        return all(item.status == PageVerificationStatus.VERIFIED for item in corrected.page_reviews)

    def build_page_workspace(self, page: NormalizedExtraction) -> PageWorkspace:
        fragments: list[PageFragment] = []
        if page.blocks:
            for index, block in enumerate(page.blocks, start=1):
                handwriting = block_handwriting_status(block)
                fragments.append(
                    PageFragment(
                        fragment_id=f"block-{index}",
                        source_kind="block",
                        label=f"Block {index}",
                        text=block.text,
                        bbox=block.bbox,
                        confidence_notes=block.confidence_notes,
                        handwritten_auto=bool(handwriting["handwritten_auto"]),
                        handwritten_manual=bool(handwriting["handwritten_manual"]),
                        handwritten_effective=bool(handwriting["handwritten_effective"]),
                        handwritten_source=str(handwriting["handwritten_source"]),
                        field_name=f"fragment_{index - 1}",
                        anchor_id=f"fragment-{index - 1}",
                    )
                )
            mode = "blocks"
        elif page.lines:
            for index, line in enumerate(page.lines, start=1):
                fragments.append(
                    PageFragment(
                        fragment_id=f"line-{index}",
                        source_kind="line",
                        label=f"Line {index}",
                        text=line.text,
                        bbox=line.bbox,
                        confidence_notes=line.confidence_notes,
                        field_name=f"fragment_{index - 1}",
                        anchor_id=f"fragment-{index - 1}",
                    )
                )
            mode = "lines"
        else:
            fragments.append(
                PageFragment(
                    fragment_id="full-text",
                    source_kind="full_text",
                    label="Full Text",
                    text=page.full_text,
                    bbox=None,
                    confidence_notes=None,
                    field_name="fragment_0",
                    anchor_id="fragment-0",
                )
            )
            mode = "full_text"

        crop_count = sum(1 for fragment in fragments if fragment.has_bbox)
        handwritten_count = sum(1 for fragment in fragments if fragment.handwritten_effective)
        return PageWorkspace(
            mode=mode,
            fragments=fragments,
            warning_count=len(self.filter_display_warnings(page.warnings)),
            crop_count=crop_count,
            handwritten_count=handwritten_count,
        )

    def build_page_preview_svg(self, document_id: str, page_number: int, mode: str = "corrected") -> str:
        context = self.get_document_context(document_id, page_number=page_number)
        normalized_mode = "raw" if mode == "raw" else "corrected"
        workspace = context.selected_raw_workspace if normalized_mode == "raw" else context.selected_page_workspace
        page = context.selected_raw_page if normalized_mode == "raw" else context.selected_page
        return self._render_preview_svg(
            document_id=document_id,
            page_number=page_number,
            mode=normalized_mode,
            workspace=workspace,
            warnings=self.filter_display_warnings(page.warnings),
            width=context.page_image_width,
            height=context.page_image_height,
            page_image_path=context.page_image_path,
        )

    def load_fragment_crop(self, document_id: str, page_number: int, fragment_id: str) -> bytes:
        context = self.get_document_context(document_id, page_number=page_number)
        fragment = self._find_fragment(context.selected_page_workspace, fragment_id)
        if fragment is None:
            fragment = self._find_fragment(context.selected_raw_workspace, fragment_id)
        if fragment is None or not fragment.has_bbox:
            raise FileNotFoundError(f"No crop is available for fragment {fragment_id}.")

        left = max(int(fragment.bbox.left or 0) - 24, 0)
        top = max(int(fragment.bbox.top or 0) - 24, 0)
        right = min(int(fragment.bbox.right or context.page_image_width) + 24, context.page_image_width)
        bottom = min(int(fragment.bbox.bottom or context.page_image_height) + 24, context.page_image_height)
        if right <= left or bottom <= top:
            raise FileNotFoundError(f"No crop is available for fragment {fragment_id}.")

        image_path = Path(context.page_image_path)
        with Image.open(image_path) as image:
            crop = image.crop((left, top, right, bottom))
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
        return buffer.getvalue()

    def command_hints(self, document_id: str | None = None) -> dict[str, str]:
        hints = {
            "run": "python main.py run --force",
            "interactive_web": "python main.py interactive --ui web",
            "interactive_tui": "python main.py interactive --ui tui",
            "pipeline_status": "python main.py pipeline-status --status all",
            "review_status": "python main.py review-status",
            "export": "python main.py export",
            "dry_run_cleanup": "python main.py dry-run-artifacts --action inspect",
        }
        if document_id:
            hints["approve"] = f"python main.py approve {document_id} --reviewer <name> --status approved"
            hints["focused_web"] = f"python main.py interactive --ui web --document-id {document_id}"
        else:
            hints["approve"] = "python main.py approve <document_id> --reviewer <name> --status approved"
        return hints

    def filter_display_warnings(self, warnings: list[str]) -> list[str]:
        technical_markers = (
            "no parseable content",
            "not valid json",
            "provider response",
            "parse",
            "parser",
            "dry-run",
            "dry run",
            "recovered parsed",
            "schema",
            "normalizer",
            "json",
        )
        return [
            warning
            for warning in warnings
            if any(marker in warning.lower() for marker in technical_markers)
        ]

    def page_image_path(self, document_id: str, page_number: int) -> str:
        return str(self.data_paths.rendered_pages_dir / document_id / f"page_{page_number:04d}.png")

    def page_image_size(self, document_id: str, page_number: int) -> tuple[int, int]:
        image_path = Path(self.page_image_path(document_id, page_number))
        with Image.open(image_path) as image:
            return image.size

    def report_uri(self, document_id: str) -> str | None:
        path = self.data_paths.reports_dir / document_id / "report.html"
        if not path.exists():
            return None
        return path.resolve().as_uri()

    def _load_review_record_map(self) -> dict[str, ReviewRecord]:
        mapping: dict[str, ReviewRecord] = {}
        for path in sorted(self.data_paths.status_dir.glob("*/review_record.json")):
            record = ReviewRecord.model_validate(self.repository.read_json(path))
            mapping[record.document_id] = record
        return mapping

    def _load_corrected_bundle(self, document_id: str) -> tuple[CorrectedDocument, ReviewRecord]:
        review_record_path = self.data_paths.status_dir / document_id / "review_record.json"
        if not review_record_path.exists():
            raise FileNotFoundError(f"Review record not found for document: {document_id}")
        review_record = ReviewRecord.model_validate(self.repository.read_json(review_record_path))
        corrected = load_corrected_document(Path(review_record.corrected_output_path), self.repository)
        return corrected, review_record

    def _load_raw_records(self, review_record: ReviewRecord) -> list[RawExtractionRecord]:
        raw_dir = Path(review_record.raw_output_path)
        records: list[RawExtractionRecord] = []
        if not raw_dir.exists():
            return records
        for path in sorted(raw_dir.glob("*.json")):
            records.append(RawExtractionRecord.model_validate(self.repository.read_json(path)))
        return records

    def _resolve_page_index(self, corrected: CorrectedDocument, page_number: int | None) -> int:
        if not corrected.extracted_pages:
            raise ValueError("No extracted pages found for this document.")
        if page_number is None:
            return 0
        for index, page in enumerate(corrected.extracted_pages):
            effective_page_number = page.page_number or (index + 1)
            if effective_page_number == page_number:
                return index
        raise ValueError(f"Page {page_number} not found in corrected document.")

    def _page_review_by_number(self, corrected: CorrectedDocument) -> dict[int, PageReviewState]:
        ensure_page_reviews(corrected)
        return {review.page_number: review for review in corrected.page_reviews}

    def _sync_target_output(self, corrected: CorrectedDocument) -> None:
        corrected.target_output.setdefault("document_id", corrected.document_id)
        corrected.target_output["pages"] = [page.model_dump(mode="json") for page in corrected.extracted_pages]

    def _mark_corrected(self, corrected: CorrectedDocument, reviewer: str) -> None:
        corrected.approval_metadata.corrected_at = utc_now()
        corrected.approval_metadata.corrected_by = reviewer or corrected.approval_metadata.corrected_by

    def _set_in_review(self, corrected: CorrectedDocument, review_record: ReviewRecord) -> None:
        if corrected.approval_metadata.review_status not in {ReviewStatus.APPROVED, ReviewStatus.REJECTED}:
            corrected.approval_metadata.review_status = ReviewStatus.IN_REVIEW
        if review_record.review_status not in {ReviewStatus.APPROVED, ReviewStatus.REJECTED}:
            review_record.review_status = ReviewStatus.IN_REVIEW

    def _sync_review_metadata(
        self,
        review_record: ReviewRecord,
        *,
        reviewer: str = "",
        comments: str = "",
        status: ReviewStatus | None = None,
    ) -> None:
        review_record.reviewer = reviewer or review_record.reviewer
        review_record.comments = comments or review_record.comments
        if status is not None:
            review_record.review_status = status

    def _save_corrected_bundle(self, review_record: ReviewRecord, corrected: CorrectedDocument) -> None:
        with self._document_write_lock(corrected.document_id):
            corrected_path = Path(review_record.corrected_output_path)
            review_record_path = self.data_paths.status_dir / corrected.document_id / "review_record.json"
            self.repository.write_yaml(corrected_path, corrected)
            self._refresh_review_report(review_record, corrected)
            self.repository.write_json(review_record_path, review_record)

    def _refresh_review_report(self, review_record: ReviewRecord, corrected: CorrectedDocument) -> None:
        page_image_paths = [
            Path(self.page_image_path(corrected.document_id, page.page_number or index))
            for index, page in enumerate(corrected.extracted_pages, start=1)
        ]
        report_html = render_review_report(
            title=self.report_title,
            document_id=corrected.document_id,
            page_image_paths=page_image_paths,
            raw_records=self._load_raw_records(review_record),
            extractions=corrected.extracted_pages,
            corrected=corrected,
            correction_path=Path(review_record.corrected_output_path),
        )
        self.repository.write_text(Path(review_record.report_path), report_html)

    def _sync_manifest_review_state(
        self,
        document_id: str,
        corrected: CorrectedDocument,
        *,
        manifest_payload: dict[str, Any] | None = None,
    ) -> DocumentManifest:
        manifest_path = self.data_paths.manifests_dir / document_id / "manifest.json"
        payload = manifest_payload if manifest_payload is not None else self.repository.read_json(manifest_path)
        manifest = DocumentManifest.model_validate(payload)
        desired_status = self._pipeline_status_from_review(corrected)
        if manifest.status != desired_status:
            with self._document_write_lock(document_id):
                manifest.status = desired_status
                manifest.updated_at = utc_now()
                self.repository.write_json(manifest_path, manifest)
        return manifest

    def _pipeline_status_from_review(self, corrected: CorrectedDocument) -> PipelineStatus:
        review_status = corrected.approval_metadata.review_status
        if review_status == ReviewStatus.APPROVED:
            return PipelineStatus.APPROVED
        if review_status == ReviewStatus.REJECTED:
            return PipelineStatus.REJECTED
        if review_status == ReviewStatus.NEEDS_RETRY:
            return PipelineStatus.NEEDS_RETRY
        if review_status == ReviewStatus.IN_REVIEW:
            return PipelineStatus.IN_REVIEW
        return PipelineStatus.REVIEW_PENDING

    def _parse_json_object(self, value: str, *, field_name: str) -> dict[str, Any]:
        try:
            payload = json.loads(value) if value.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{field_name} must be a JSON object.")
        return payload

    def _page_is_parse_placeholder(self, page: NormalizedExtraction) -> bool:
        return (
            not page.full_text.strip()
            and not page.structured_content
            and not page.blocks
            and not page.lines
            and not page.tables
            and page.warnings == ["No parseable content returned by provider."]
        )

    def _auto_heal_placeholder_pages(
        self,
        document_id: str,
        corrected: CorrectedDocument,
        review_record: ReviewRecord,
        raw_records: list[RawExtractionRecord],
    ) -> list[int]:
        healed_pages: list[int] = []
        if not corrected.extracted_pages:
            return healed_pages

        previous_consolidated = consolidate_text(corrected.extracted_pages)
        updated = False
        for index, page in enumerate(corrected.extracted_pages, start=1):
            page_number = page.page_number or index
            if not self._page_is_parse_placeholder(page):
                continue
            raw_record = self._raw_record_for_page(raw_records, page_number)
            if raw_record is None:
                continue
            recovered = self._normalize_raw_record(document_id, page_number, raw_record)
            if recovered is None or self._page_is_parse_placeholder(recovered):
                continue
            corrected.extracted_pages[index - 1] = recovered
            self._save_parsed_page(recovered)
            healed_pages.append(page_number)
            updated = True

        if updated:
            self._sync_target_output(corrected)
            if not corrected.consolidated_text or corrected.consolidated_text == previous_consolidated:
                corrected.consolidated_text = consolidate_text(corrected.extracted_pages)
                corrected.target_output["consolidated_text"] = corrected.consolidated_text
            self._save_corrected_bundle(review_record, corrected)
        return healed_pages

    def _normalize_raw_record(
        self,
        document_id: str,
        page_number: int,
        raw_record: RawExtractionRecord,
    ) -> NormalizedExtraction | None:
        try:
            parsed_payload = self.parser_provider.parse_response(
                document_id=document_id,
                page_number=page_number,
                raw_response_text=raw_record.raw_response_text,
                raw_response_json=raw_record.raw_response_json,
            )
            return normalize_provider_payload(document_id, page_number, parsed_payload)
        except Exception:  # noqa: BLE001
            return None

    def _save_parsed_page(self, extraction: NormalizedExtraction) -> None:
        path = self.data_paths.parsed_dir / extraction.document_id / f"page_{(extraction.page_number or 0):04d}.json"
        self.repository.write_json(path, extraction)

    def _load_raw_page(
        self,
        document_id: str,
        page_number: int,
        raw_records: list[RawExtractionRecord],
        *,
        fallback_page: NormalizedExtraction,
    ) -> NormalizedExtraction:
        parsed_path = self.data_paths.parsed_dir / document_id / f"page_{page_number:04d}.json"
        parsed_page: NormalizedExtraction | None = None
        if parsed_path.exists():
            parsed_page = NormalizedExtraction.model_validate(self.repository.read_json(parsed_path))
            if not self._page_is_parse_placeholder(parsed_page):
                return parsed_page

        raw_record = self._raw_record_for_page(raw_records, page_number)
        recovered = self._normalize_raw_record(document_id, page_number, raw_record) if raw_record is not None else None
        if recovered is not None:
            return recovered
        return parsed_page or fallback_page

    def _raw_record_for_page(
        self,
        raw_records: list[RawExtractionRecord],
        page_number: int,
    ) -> RawExtractionRecord | None:
        exact_match = next((record for record in raw_records if record.page_number == page_number), None)
        if exact_match is not None:
            return exact_match
        return next((record for record in raw_records if record.page_number is None), None)

    def _find_fragment(self, workspace: PageWorkspace, fragment_id: str) -> PageFragment | None:
        for fragment in workspace.fragments:
            if fragment.fragment_id == fragment_id:
                return fragment
        return None

    def _parse_bbox_payload(self, value: str) -> dict[str, BoundingBox | None]:
        if not value.strip():
            return {}
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("bbox_payload_json must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("bbox_payload_json must be a JSON object.")

        parsed: dict[str, BoundingBox | None] = {}
        for fragment_id, bbox_payload in payload.items():
            if not isinstance(fragment_id, str) or not fragment_id:
                raise ValueError("bbox_payload_json keys must be fragment ids.")
            if bbox_payload is None:
                parsed[fragment_id] = None
                continue
            if not isinstance(bbox_payload, dict):
                raise ValueError("bbox payload entries must be objects or null.")
            bbox = BoundingBox.model_validate(bbox_payload)
            if None in {bbox.left, bbox.top, bbox.right, bbox.bottom}:
                raise ValueError("bbox payload entries must include left, top, right, and bottom.")
            parsed[fragment_id] = bbox
        return parsed

    def _apply_bbox_updates(
        self,
        document_id: str,
        page_number: int,
        extraction: NormalizedExtraction,
        *,
        workspace: PageWorkspace,
        bbox_payload: dict[str, BoundingBox | None],
    ) -> bool:
        if not bbox_payload:
            return False
        if workspace.mode == "full_text":
            raise ValueError("Bounding box edits are not available for full-text-only pages.")

        page_width, page_height = self.page_image_size(document_id, page_number)
        if workspace.mode == "blocks":
            targets = extraction.blocks
        else:
            targets = extraction.lines

        target_map = {
            fragment.fragment_id: targets[index]
            for index, fragment in enumerate(workspace.fragments)
            if index < len(targets)
        }
        changed = False
        for fragment_id, incoming_bbox in bbox_payload.items():
            if fragment_id not in target_map:
                raise ValueError(f"Unknown fragment id in bbox payload: {fragment_id}")
            normalized_bbox = (
                None
                if incoming_bbox is None
                else self._normalize_bbox(incoming_bbox, page_width=page_width, page_height=page_height)
            )
            target = target_map[fragment_id]
            if not self._bbox_equal(target.bbox, normalized_bbox):
                target.bbox = normalized_bbox
                changed = True
        return changed

    def _normalize_bbox(
        self,
        bbox: BoundingBox,
        *,
        page_width: int,
        page_height: int,
    ) -> BoundingBox:
        if None in {bbox.left, bbox.top, bbox.right, bbox.bottom}:
            raise ValueError("Bounding boxes must include left, top, right, and bottom.")

        left = self._clamp_coordinate(float(bbox.left), 0.0, float(page_width))
        right = self._clamp_coordinate(float(bbox.right), 0.0, float(page_width))
        top = self._clamp_coordinate(float(bbox.top), 0.0, float(page_height))
        bottom = self._clamp_coordinate(float(bbox.bottom), 0.0, float(page_height))
        normalized_left, normalized_right = sorted((left, right))
        normalized_top, normalized_bottom = sorted((top, bottom))
        return BoundingBox(
            left=normalized_left,
            top=normalized_top,
            right=normalized_right,
            bottom=normalized_bottom,
        )

    def _clamp_coordinate(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _bbox_equal(self, first: BoundingBox | None, second: BoundingBox | None) -> bool:
        if first is None or second is None:
            return first is None and second is None
        return (
            first.left == second.left
            and first.top == second.top
            and first.right == second.right
            and first.bottom == second.bottom
        )

    def _render_preview_svg(
        self,
        *,
        document_id: str,
        page_number: int,
        mode: str,
        workspace: PageWorkspace,
        warnings: list[str],
        width: int,
        height: int,
        page_image_path: str,
    ) -> str:
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{escape(document_id)} page {page_number} {escape(mode)} preview</title>",
            f"<desc id=\"desc\">Rendered {escape(mode)} text mapping for page {page_number}.</desc>",
            '<style>.preview-label{font-family:Segoe UI Variable, Noto Naskh Arabic, serif;font-size:14px;font-weight:700;}.preview-note{font-family:Segoe UI Variable, Noto Naskh Arabic, serif;font-size:12px;}</style>',
            f'<rect width="{width}" height="{height}" fill="#fbf7f0" />',
            f'<rect x="6" y="6" width="{max(width - 12, 1)}" height="{max(height - 12, 1)}" rx="18" fill="#fffdfa" stroke="#d7b79d" stroke-width="1.5" />',
        ]

        bbox_fragments = sorted(
            [fragment for fragment in workspace.fragments if fragment.has_bbox],
            key=lambda item: (
                float(item.bbox.top or 0) if item.bbox is not None else 0.0,
                float(item.bbox.left or 0) if item.bbox is not None else 0.0,
            ),
        )
        no_bbox_fragments = [fragment for fragment in workspace.fragments if not fragment.has_bbox]
        if bbox_fragments:
            for index, fragment in enumerate(bbox_fragments, start=1):
                fragment_svg = self._render_bbox_fragment(
                    fragment,
                    index,
                )
                svg_parts.extend(fragment_svg)
        if no_bbox_fragments:
            footer_height = min(168, max(84, 36 + len(no_bbox_fragments) * 22))
            footer_top = max(height - footer_height - 18, 100)
            svg_parts.append(
                f'<rect x="18" y="{footer_top}" width="{max(width - 36, 1)}" height="{footer_height}" rx="18" fill="#fff8ef" stroke="#d7c2b1" stroke-width="1.5" />'
            )
            svg_parts.append(
                f'<text x="34" y="{footer_top + 28}" fill="#8d4a2b" class="preview-label">Fragments Without Geometry</text>'
            )
            for index, fragment in enumerate(no_bbox_fragments, start=1):
                preview = (fragment.text or "No extracted text yet.").replace("\n", " ").strip()
                preview = preview[:84] + ("..." if len(preview) > 84 else "")
                svg_parts.append(
                    f'<text x="34" y="{footer_top + 32 + index * 22}" fill="#5f554f" class="preview-note">{escape(fragment.label)}: {escape(preview)}</text>'
                )
        if warnings:
            warning_badge = f"{len(warnings)} system alert{'s' if len(warnings) != 1 else ''}"
            svg_parts.append(
                f'<rect x="18" y="18" width="{min(240, max(width - 36, 1))}" height="28" rx="14" fill="rgba(154,107,29,0.12)" stroke="#a36b1f" stroke-width="1" />'
            )
            svg_parts.append(
                f'<text x="32" y="37" fill="#8d5f17" class="preview-note">{escape(warning_badge)}</text>'
            )

        if not bbox_fragments and not no_bbox_fragments:
            svg_parts.append(
                f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" fill="#6e5e54" class="preview-label">No mapped content available.</text>'
            )

        svg_parts.append("</svg>")
        return "".join(svg_parts)

    def _render_bbox_fragment(
        self,
        fragment: PageFragment,
        index: int,
    ) -> list[str]:
        assert fragment.bbox is not None
        left = max(int(fragment.bbox.left or 0), 0)
        top = max(int(fragment.bbox.top or 0), 0)
        right = max(int(fragment.bbox.right or 0), left + 40)
        bottom = max(int(fragment.bbox.bottom or 0), top + 28)
        width = max(right - left, 40)
        height = max(bottom - top, 28)
        content_x = left + 6
        content_y = top + 28 if height >= 58 else top + 6
        content_width = max(width - 12, 1)
        content_height = max(bottom - content_y - 6, 1)
        font_size, _ = self._fit_text_box(fragment.text or "Empty fragment", content_width, content_height)
        block_html = self._render_bidi_preview_lines(fragment.text or "Empty fragment")
        label_width = min(max(len(fragment.label) * 8 + 26, 84), max(width, 84))
        return [
            f'<g id="bbox-{escape(fragment.fragment_id)}">',
            f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="12" fill="rgba(141,74,43,0.08)" stroke="#8d4a2b" stroke-width="1.5" stroke-dasharray="8 6" />',
            f'<rect x="{left}" y="{top}" width="{label_width}" height="22" rx="11" fill="rgba(24,18,14,0.78)" />',
            f'<text x="{left + 10}" y="{top + 15}" fill="#fff8ee" class="preview-note">{index}. {escape(fragment.label)}</text>',
            f'<foreignObject x="{content_x}" y="{content_y}" width="{content_width}" height="{content_height}"><div xmlns="http://www.w3.org/1999/xhtml" style="box-sizing:border-box;width:100%;height:100%;padding:0 4px 2px 4px;font-family:Segoe UI Variable, Noto Naskh Arabic, serif;font-size:{font_size}px;line-height:1.32;color:#241712;direction:rtl;text-align:right;word-break:break-word;white-space:pre-wrap;overflow:auto;overscroll-behavior:contain;scrollbar-width:thin;">{block_html}</div></foreignObject>',
            "</g>",
        ]

    def _render_bidi_preview_lines(self, text: str) -> str:
        lines = text.splitlines() or [""]
        rendered_lines = []
        for line in lines:
            direction = "rtl" if self._looks_rtl(line) else "ltr"
            line_html = escape(line) if line else "&#160;"
            rendered_lines.append(
                f'<div dir="{direction}" style="direction:{direction};unicode-bidi:isolate;text-align:right;">{line_html}</div>'
            )
        return "".join(rendered_lines)

    def _looks_rtl(self, value: str) -> bool:
        return any("\u0600" <= character <= "\u08ff" for character in value)

    def _fit_text_box(self, text: str, width: int, original_height: int) -> tuple[int, int]:
        max_font = min(22, max(10, int(min(original_height / 2.2, width / 7.2))))
        min_font = 10
        for font_size in range(max_font, min_font - 1, -1):
            line_height = max(int(font_size * 1.36), font_size + 3)
            line_count = self._estimate_line_count(text, width, font_size)
            required_height = line_count * line_height
            if required_height <= max(original_height, 1):
                return max(font_size, 26), required_height
        fallback_font = min_font
        fallback_line_height = max(int(fallback_font * 1.36), fallback_font + 3)
        fallback_count = self._estimate_line_count(text, width, fallback_font)
        return max(fallback_font, 26), fallback_count * fallback_line_height

    def _estimate_line_count(self, text: str, width: int, font_size: int) -> int:
        characters_per_line = max(int(width / max(font_size * 0.56, 1)), 4)
        wrapper = textwrap.TextWrapper(
            width=characters_per_line,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        line_count = 0
        for paragraph in (text or "").splitlines() or [""]:
            wrapped = wrapper.wrap(paragraph) or [""]
            line_count += len(wrapped)
        return max(line_count, 1)


def serialize_summary(summary: dict[str, Any] | None) -> str:
    return json.dumps(summary or {}, ensure_ascii=False, indent=2, sort_keys=True)


def serialize_page_json(corrected: CorrectedDocument, page_index: int) -> dict[str, str]:
    page = corrected.extracted_pages[page_index]
    return {
        "full_text": page.full_text,
        "structured_content_json": json.dumps(page.structured_content, ensure_ascii=False, indent=2, sort_keys=True),
    }


def serialize_target_output(corrected: CorrectedDocument) -> str:
    return json.dumps(corrected.target_output, ensure_ascii=False, indent=2, sort_keys=True)


def serialize_timestamp(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def generated_label() -> str:
    return utc_now_iso()
