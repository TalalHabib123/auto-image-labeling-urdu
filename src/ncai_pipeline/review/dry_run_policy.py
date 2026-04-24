from __future__ import annotations

from ncai_pipeline.schemas.review import CorrectedDocument, ReviewRecord


def corrected_document_is_dry_run(corrected: CorrectedDocument) -> bool:
    if corrected.approval_metadata.is_dry_run:
        return True
    for page in corrected.extracted_pages:
        if page.structured_content.get("mode") == "dry_run":
            return True
        if any("dry-run" in warning.lower() or "dry run" in warning.lower() for warning in page.warnings):
            return True
        if any("dry-run" in note.lower() or "dry run" in note.lower() for note in page.confidence_notes):
            return True
    return False


def review_record_is_dry_run(record: ReviewRecord) -> bool:
    return record.is_dry_run

