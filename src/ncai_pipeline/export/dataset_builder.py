from __future__ import annotations

import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from ncai_pipeline.config import AppConfig
from ncai_pipeline.export.split_builder import assign_split
from ncai_pipeline.parsing.validators import load_corrected_document
from ncai_pipeline.review.dry_run_policy import corrected_document_is_dry_run
from ncai_pipeline.review.handwriting import block_handwriting_status
from ncai_pipeline.schemas.common import BoundingBox, ReviewStatus
from ncai_pipeline.schemas.export import CuratedTrainingSample, ExportManifest, ExportStats
from ncai_pipeline.schemas.extraction import ExtractedBlock
from ncai_pipeline.schemas.review import CorrectedDocument, ReviewRecord
from ncai_pipeline.storage.paths import DataPaths
from ncai_pipeline.storage.nas import CREATE_ONLY_PUBLISH_STRATEGY, NasOutputIndex
from ncai_pipeline.storage.repository import FileRepository
from ncai_pipeline.utils.time import utc_now_iso


@dataclass
class ApprovedDocument:
    record: ReviewRecord
    corrected: CorrectedDocument
    split: str
    is_dry_run: bool


class DatasetBuilder:
    def __init__(
        self,
        config: AppConfig,
        data_paths: DataPaths,
        repository: FileRepository,
        source_data_paths: DataPaths | None = None,
    ) -> None:
        self.config = config
        self.data_paths = data_paths
        self.source_data_paths = source_data_paths or data_paths
        self.repository = repository
        self._asset_counts: Counter[str] = Counter()
        self._skipped_counts: Counter[str] = Counter()
        self._counted_page_assets: set[Path] = set()
        self._counted_crop_assets: set[Path] = set()

    def load_review_records(self) -> list[ReviewRecord]:
        records: list[ReviewRecord] = []
        if (
            self.config.pipeline.storage_mode == "nas"
            and self.config.nas.publish_strategy == CREATE_ONLY_PUBLISH_STRATEGY
        ):
            paths = NasOutputIndex(self.source_data_paths, self.repository).latest_review_record_paths()
        else:
            paths = sorted(self.source_data_paths.status_dir.glob("*/review_record.json"))
        for path in paths:
            records.append(ReviewRecord.model_validate(self.repository.read_json(path)))
        return records

    def build_samples(
        self,
        include_dry_run: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[CuratedTrainingSample], ExportManifest]:
        self._reset_export_counts()
        documents, review_counts, excluded_dry_run = self._load_approved_documents(
            include_dry_run=include_dry_run,
            progress_callback=progress_callback,
        )
        samples = [self._build_full_sample(document) for document in documents]
        stats = self._stats_for_documents(
            documents,
            dataset_counts={"full": len(samples)},
            review_counts=review_counts,
            excluded_dry_run=excluded_dry_run,
        )
        manifest = ExportManifest(
            export_name=self.config.export.target_format,
            samples=len(samples),
            generated_at=utc_now_iso(),
            stats=stats,
            dataset_types=["full"],
            output_formats=["jsonl"],
        )
        return samples, manifest

    def build_datasets(
        self,
        *,
        dataset_types: list[str] | None = None,
        region_source: str | None = None,
        include_dry_run: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, list[dict[str, Any]]], ExportManifest]:
        self._reset_export_counts()
        documents, review_counts, excluded_dry_run = self._load_approved_documents(
            include_dry_run=include_dry_run,
            progress_callback=progress_callback,
        )
        dataset_ids = normalize_dataset_ids(dataset_types or self.config.export.dataset_types, region_source or self.config.export.region_source)
        datasets = {dataset_id: [] for dataset_id in dataset_ids}
        include_page_assets = any(
            dataset_id in {"full", "page_text", "regions_blocks", "regions_sections"}
            for dataset_id in dataset_ids
        )

        for document in documents:
            for dataset_id in dataset_ids:
                if dataset_id == "full":
                    datasets[dataset_id].append(self._build_full_sample(document).model_dump(mode="json"))
                elif dataset_id == "page_text":
                    datasets[dataset_id].extend(self._build_page_text_samples(document))
                elif dataset_id == "regions_blocks":
                    datasets[dataset_id].extend(self._build_region_block_samples(document))
                elif dataset_id == "regions_sections":
                    datasets[dataset_id].extend(self._build_region_section_samples(document))
                elif dataset_id == "handwritten":
                    datasets[dataset_id].extend(
                        self._build_handwritten_samples(document, include_page_image=include_page_assets)
                    )
                else:
                    raise ValueError(f"Unsupported dataset type: {dataset_id}")

        dataset_counts = {dataset_id: len(records) for dataset_id, records in datasets.items()}
        stats = self._stats_for_documents(
            documents,
            dataset_counts=dataset_counts,
            review_counts=review_counts,
            excluded_dry_run=excluded_dry_run,
        )
        stats.sample_counts = dict(
            Counter(record["source_split"] for records in datasets.values() for record in records if record.get("source_split"))
        )
        manifest = ExportManifest(
            export_name=self.config.export.target_format,
            samples=sum(dataset_counts.values()),
            generated_at=utc_now_iso(),
            stats=stats,
            dataset_types=dataset_ids,
            output_formats=self.config.export.output_formats,
        )
        return datasets, manifest

    def _reset_export_counts(self) -> None:
        self._asset_counts = Counter()
        self._skipped_counts = Counter()
        self._counted_page_assets = set()
        self._counted_crop_assets = set()

    def _load_approved_documents(
        self,
        *,
        include_dry_run: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> tuple[list[ApprovedDocument], Counter[str], int]:
        documents: list[ApprovedDocument] = []
        review_counts: Counter[str] = Counter()
        excluded_dry_run = 0
        records = self.load_review_records()
        total_records = len(records)

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "start",
                    "total_records": total_records,
                    "completed_records": 0,
                    "included_samples": 0,
                }
            )

        for index, record in enumerate(records, start=1):
            review_counts[record.review_status.value] += 1
            corrected = load_corrected_document(Path(record.corrected_output_path), self.repository)
            outcome = "skipped_not_approved"
            if corrected.approval_metadata.review_status != ReviewStatus.APPROVED:
                self._emit_record_progress(progress_callback, record, index, total_records, len(documents), outcome, corrected)
                continue

            is_dry_run = corrected_document_is_dry_run(corrected)
            if is_dry_run and not include_dry_run:
                excluded_dry_run += 1
                outcome = "skipped_dry_run"
                self._emit_record_progress(progress_callback, record, index, total_records, len(documents), outcome, corrected)
                continue

            split = assign_split(record.document_id, self.config.export.split_ratios)
            documents.append(ApprovedDocument(record=record, corrected=corrected, split=split, is_dry_run=is_dry_run))
            self._emit_record_progress(
                progress_callback,
                record,
                index,
                total_records,
                len(documents),
                "included",
                corrected,
                split=split,
            )
        return documents, review_counts, excluded_dry_run

    def _emit_record_progress(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        record: ReviewRecord,
        index: int,
        total_records: int,
        included_samples: int,
        outcome: str,
        corrected: CorrectedDocument,
        *,
        split: str | None = None,
    ) -> None:
        if progress_callback is None:
            return
        payload = {
            "event": "record",
            "document_id": record.document_id,
            "completed_records": index,
            "total_records": total_records,
            "included_samples": included_samples,
            "outcome": outcome,
            "review_status": corrected.approval_metadata.review_status.value,
        }
        if split:
            payload["split"] = split
        progress_callback(payload)

    def _build_full_sample(self, document: ApprovedDocument) -> CuratedTrainingSample:
        corrected = document.corrected
        input_media_paths = [
            self._portable_export_path(self._page_asset_path(corrected.document_id, page.page_number or index))
            for index, page in enumerate(corrected.extracted_pages, start=1)
        ]
        sample = CuratedTrainingSample(
            sample_id=f"{document.record.document_id}_{document.split}",
            document_id=document.record.document_id,
            page_refs=[page.page_number or 0 for page in corrected.extracted_pages],
            input_media_paths=input_media_paths,
            task_instruction=self.config.export.instruction,
            target_output=corrected.target_output or {
                "document_id": corrected.document_id,
                "consolidated_text": corrected.consolidated_text,
            },
            language=(corrected.extracted_pages[0].language if corrected.extracted_pages else "ur") or "ur",
            source_split=document.split,
            approval_metadata=corrected.approval_metadata,
        )
        target_dir = self.data_paths.dry_run_dir if document.is_dry_run else self.data_paths.approved_dir
        self.repository.write_json(target_dir / document.record.document_id / "sample.json", sample)
        return sample

    def _build_page_text_samples(self, document: ApprovedDocument) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for index, page in enumerate(document.corrected.extracted_pages, start=1):
            page_number = page.page_number or index
            samples.append(
                {
                    "sample_id": f"{document.record.document_id}_page_{page_number:04d}_{document.split}",
                    "dataset_type": "page_text",
                    "document_id": document.record.document_id,
                    "page_number": page_number,
                    "image_path": self._portable_export_path(self._page_asset_path(document.record.document_id, page_number)),
                    "full_text": page.full_text,
                    "language": page.language or "ur",
                    "source_split": document.split,
                    "approval_metadata": document.corrected.approval_metadata.model_dump(mode="json"),
                }
            )
        return samples

    def _build_region_block_samples(self, document: ApprovedDocument) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for index, page in enumerate(document.corrected.extracted_pages, start=1):
            page_number = page.page_number or index
            blocks = [self._block_payload(block) for block in page.blocks]
            samples.append(
                {
                    "sample_id": f"{document.record.document_id}_page_{page_number:04d}_blocks_{document.split}",
                    "dataset_type": "regions_blocks",
                    "document_id": document.record.document_id,
                    "page_number": page_number,
                    "image_path": self._portable_export_path(self._page_asset_path(document.record.document_id, page_number)),
                    "target_output": {"blocks": blocks},
                    "blocks": blocks,
                    "language": page.language or "ur",
                    "source_split": document.split,
                    "approval_metadata": document.corrected.approval_metadata.model_dump(mode="json"),
                }
            )
        return samples

    def _build_region_section_samples(self, document: ApprovedDocument) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for index, page in enumerate(document.corrected.extracted_pages, start=1):
            page_number = page.page_number or index
            sections = page.structured_content.get("sections", [])
            samples.append(
                {
                    "sample_id": f"{document.record.document_id}_page_{page_number:04d}_sections_{document.split}",
                    "dataset_type": "regions_sections",
                    "document_id": document.record.document_id,
                    "page_number": page_number,
                    "image_path": self._portable_export_path(self._page_asset_path(document.record.document_id, page_number)),
                    "target_output": {"sections": sections},
                    "sections": sections,
                    "language": page.language or "ur",
                    "source_split": document.split,
                    "approval_metadata": document.corrected.approval_metadata.model_dump(mode="json"),
                }
            )
        return samples

    def _build_handwritten_samples(self, document: ApprovedDocument, *, include_page_image: bool) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for index, page in enumerate(document.corrected.extracted_pages, start=1):
            page_number = page.page_number or index
            source_page_path = (
                self._page_asset_path(document.record.document_id, page_number)
                if include_page_image
                else None
            )
            for block in page.blocks:
                handwriting = self._block_handwriting_status(block)
                if not handwriting["handwritten_effective"]:
                    continue
                if not block.text.strip():
                    self._skipped_counts["handwritten_empty_text"] += 1
                    continue
                if not self._has_bbox(block.bbox):
                    self._skipped_counts["handwritten_missing_bbox"] += 1
                    continue
                crop_path = self._write_handwritten_crop(document.record.document_id, page_number, block.block_id, block.bbox)
                if crop_path is None:
                    self._skipped_counts["handwritten_crop_failed"] += 1
                    continue
                sample = {
                    "sample_id": f"{document.record.document_id}_page_{page_number:04d}_{_safe_name(block.block_id)}_{document.split}",
                    "dataset_type": "handwritten",
                    "document_id": document.record.document_id,
                    "page_number": page_number,
                    "block_id": block.block_id,
                    "image_path": self._portable_export_path(crop_path),
                    "text": block.text,
                    "bbox": _bbox_payload(block.bbox),
                    "handwritten_auto": handwriting["handwritten_auto"],
                    "handwritten_manual": handwriting["handwritten_manual"],
                    "handwritten_effective": handwriting["handwritten_effective"],
                    "handwritten_source": handwriting["handwritten_source"],
                    "language": page.language or "ur",
                    "source_split": document.split,
                    "approval_metadata": document.corrected.approval_metadata.model_dump(mode="json"),
                }
                if source_page_path is not None:
                    sample["page_path"] = self._portable_export_path(source_page_path)
                samples.append(sample)
        return samples

    def _block_payload(self, block: ExtractedBlock) -> dict[str, Any]:
        handwriting = self._block_handwriting_status(block)
        return {
            "block_id": block.block_id,
            "text": block.text,
            "bbox": _bbox_payload(block.bbox),
            "confidence_notes": block.confidence_notes,
            "handwritten_auto": handwriting["handwritten_auto"],
            "handwritten_manual": handwriting["handwritten_manual"],
            "handwritten_effective": handwriting["handwritten_effective"],
            "handwritten_source": handwriting["handwritten_source"],
        }

    def _block_handwriting_status(self, block: ExtractedBlock) -> dict[str, bool | str]:
        return block_handwriting_status(
            block,
            keywords=self.config.export.handwritten.keywords,
            auto_detect=self.config.export.handwritten.auto_detect,
        )

    def _page_asset_path(self, document_id: str, page_number: int) -> Path:
        source = self._source_page_path(document_id, page_number)
        if self.config.export.assets.page_strategy == "reference_working":
            return source
        target = self.data_paths.export_page_assets_dir / document_id / f"page_{page_number:04d}.png"
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if target not in self._counted_page_assets:
                self._asset_counts["page_images"] += 1
                self._counted_page_assets.add(target)
            return target
        self._skipped_counts["missing_page_images"] += 1
        return target

    def _write_handwritten_crop(
        self,
        document_id: str,
        page_number: int,
        block_id: str,
        bbox: BoundingBox | None,
    ) -> Path | None:
        if not self._has_bbox(bbox):
            return None
        source = self._source_page_path(document_id, page_number)
        if not source.exists():
            return None
        target = (
            self.data_paths.export_crop_assets_dir
            / "handwritten"
            / document_id
            / f"page_{page_number:04d}"
            / f"{_safe_name(block_id)}.png"
        )
        try:
            with Image.open(source) as image:
                left, top, right, bottom = self._clamped_crop_box(bbox, image.width, image.height)
                if right <= left or bottom <= top:
                    return None
                target.parent.mkdir(parents=True, exist_ok=True)
                image.crop((left, top, right, bottom)).save(target, format="PNG")
        except OSError:
            return None
        if target not in self._counted_crop_assets:
            self._asset_counts["handwritten_crops"] += 1
            self._counted_crop_assets.add(target)
        return target

    def _portable_export_path(self, path: Path) -> str:
        for root in (self.data_paths.exports_dir, self.source_data_paths.root, self.data_paths.root):
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                continue
        return path.as_posix()

    def _source_page_path(self, document_id: str, page_number: int) -> Path:
        if (
            self.config.pipeline.storage_mode == "nas"
            and self.config.nas.publish_strategy == CREATE_ONLY_PUBLISH_STRATEGY
        ):
            return NasOutputIndex(self.source_data_paths, self.repository).latest_page_image_path(document_id, page_number)
        return self.source_data_paths.rendered_pages_dir / document_id / f"page_{page_number:04d}.png"

    def _clamped_crop_box(self, bbox: BoundingBox | None, width: int, height: int) -> tuple[int, int, int, int]:
        assert bbox is not None
        padding = self.config.export.handwritten.crop_padding
        left = max(int(float(bbox.left or 0)) - padding, 0)
        top = max(int(float(bbox.top or 0)) - padding, 0)
        right = min(int(float(bbox.right or width)) + padding, width)
        bottom = min(int(float(bbox.bottom or height)) + padding, height)
        return left, top, right, bottom

    def _has_bbox(self, bbox: BoundingBox | None) -> bool:
        return bbox is not None and None not in {bbox.left, bbox.top, bbox.right, bbox.bottom}

    def _stats_for_documents(
        self,
        documents: list[ApprovedDocument],
        *,
        dataset_counts: dict[str, int],
        review_counts: Counter[str],
        excluded_dry_run: int,
    ) -> ExportStats:
        split_counts = Counter(document.split for document in documents)
        split_page_counts = Counter(
            {
                split: sum(len(document.corrected.extracted_pages) for document in documents if document.split == split)
                for split in {"train", "val", "test"}
            }
        )
        language_counts: Counter[str] = Counter()
        total_text_length = 0
        for document in documents:
            first_page = document.corrected.extracted_pages[0] if document.corrected.extracted_pages else None
            language_counts[(first_page.language if first_page else "ur") or "ur"] += 1
            total_text_length += len(document.corrected.consolidated_text)
        error_counts = {"dry_run_excluded": excluded_dry_run} if excluded_dry_run else {}
        error_counts.update(dict(self._skipped_counts))
        return ExportStats(
            sample_counts=dict(split_counts),
            page_counts={key: value for key, value in split_page_counts.items() if value},
            dataset_counts=dataset_counts,
            asset_counts=dict(self._asset_counts),
            skipped_counts=dict(self._skipped_counts),
            average_text_length=round(total_text_length / len(documents), 2) if documents else 0.0,
            language_counts=dict(language_counts),
            review_coverage=dict(review_counts),
            error_counts=error_counts,
        )


def normalize_dataset_ids(dataset_types: list[str], region_source: str) -> list[str]:
    normalized: list[str] = []
    for item in dataset_types:
        dataset_type = item.strip().lower()
        if dataset_type == "regions":
            dataset_type = f"regions_{region_source.strip().lower()}"
        if dataset_type not in {"full", "page_text", "regions_blocks", "regions_sections", "handwritten"}:
            raise ValueError(f"Unsupported dataset type: {item}")
        if dataset_type not in normalized:
            normalized.append(dataset_type)
    return normalized


def _bbox_payload(bbox: BoundingBox | None) -> dict[str, float] | None:
    if bbox is None:
        return None
    if None in {bbox.left, bbox.top, bbox.right, bbox.bottom}:
        return None
    return {
        "left": float(bbox.left or 0),
        "top": float(bbox.top or 0),
        "right": float(bbox.right or 0),
        "bottom": float(bbox.bottom or 0),
    }


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "fragment"
