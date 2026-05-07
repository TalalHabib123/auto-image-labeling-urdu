from __future__ import annotations

from pathlib import Path

from ncai_pipeline.config import ReviewSettings
from ncai_pipeline.pipeline.stages import consolidate_text
from ncai_pipeline.review.html_report import render_review_report
from ncai_pipeline.schemas.common import ApprovalMetadata, PageProcessingState, ReviewStatus
from ncai_pipeline.schemas.document import PageArtifact
from ncai_pipeline.schemas.extraction import NormalizedExtraction, RawExtractionRecord
from ncai_pipeline.schemas.review import CorrectedDocument, PageReviewState, ReviewRecord
from ncai_pipeline.storage.repository import FileRepository


class ReviewArtifactBuilder:
    def __init__(self, settings: ReviewSettings, repository: FileRepository, reports_dir: Path, correction_dir: Path) -> None:
        self.settings = settings
        self.repository = repository
        self.reports_dir = reports_dir
        self.correction_dir = correction_dir

    def build(
        self,
        document_id: str,
        page_artifacts: list[PageArtifact],
        raw_records: list[RawExtractionRecord],
        extractions: list[NormalizedExtraction],
        pipeline_version: str,
        raw_output_path: Path,
        is_dry_run: bool,
        page_processing: list[PageProcessingState] | None = None,
        prior_corrected: CorrectedDocument | None = None,
        reset_review_pages: set[int] | None = None,
    ) -> ReviewRecord:
        correction_path = self.correction_dir / document_id / "corrected.yaml"
        consolidated_text = consolidate_text(extractions)
        notes = ["Generated from dry-run placeholder output. Do not include in production exports."] if is_dry_run else []
        reset_review_pages = reset_review_pages or set()
        prior_reviews = {
            review.page_number: review
            for review in (prior_corrected.page_reviews if prior_corrected is not None else [])
        }
        page_reviews = []
        for index, page in enumerate(extractions, start=1):
            page_number = page.page_number or index
            existing_review = prior_reviews.get(page_number)
            if existing_review is not None and page_number not in reset_review_pages:
                page_reviews.append(existing_review)
            else:
                page_reviews.append(PageReviewState(page_number=page_number))

        approval_metadata = (
            prior_corrected.approval_metadata.model_copy(deep=True)
            if prior_corrected is not None
            else ApprovalMetadata(
                source_version=pipeline_version,
                is_dry_run=is_dry_run,
                retention_label="temporary" if is_dry_run else None,
                notes=notes,
            )
        )
        approval_metadata.source_version = approval_metadata.source_version or pipeline_version
        approval_metadata.is_dry_run = is_dry_run
        approval_metadata.retention_label = "temporary" if is_dry_run else approval_metadata.retention_label
        if notes:
            approval_metadata.notes = list({*approval_metadata.notes, *notes})
        if reset_review_pages and approval_metadata.review_status == ReviewStatus.APPROVED:
            approval_metadata.review_status = ReviewStatus.IN_REVIEW

        corrected = CorrectedDocument(
            document_id=document_id,
            approval_metadata=approval_metadata,
            extracted_pages=extractions,
            page_reviews=page_reviews,
            page_processing=page_processing or [],
            consolidated_text=consolidated_text,
            target_output={
                "document_id": document_id,
                "pages": [item.model_dump(mode="json") for item in extractions],
                "consolidated_text": consolidated_text,
            },
        )
        self.repository.write_yaml(correction_path, corrected)

        report_path = self.reports_dir / document_id / "report.html"
        report_html = render_review_report(
            title=self.settings.report_title,
            document_id=document_id,
            page_image_paths=[Path(page.image_path) for page in page_artifacts],
            raw_records=raw_records,
            extractions=extractions,
            corrected=corrected,
            correction_path=correction_path,
        )
        self.repository.write_text(report_path, report_html)

        return ReviewRecord(
            document_id=document_id,
            is_dry_run=is_dry_run,
            retention_label="temporary" if is_dry_run else None,
            corrected_output_path=str(correction_path),
            raw_output_path=str(raw_output_path.resolve()),
            report_path=str(report_path),
        )
