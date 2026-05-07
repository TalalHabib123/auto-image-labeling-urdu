from __future__ import annotations

from pathlib import Path

from ncai_pipeline.schemas.common import PageProcessingState, PageProcessingStatus, ReviewStatus
from ncai_pipeline.schemas.review import CorrectedDocument, PageReviewState
from ncai_pipeline.storage.repository import FileRepository


def ensure_page_reviews(document: CorrectedDocument) -> CorrectedDocument:
    existing_reviews = {review.page_number: review for review in document.page_reviews}
    bootstrapped: list[PageReviewState] = []

    if document.extracted_pages:
        for index, page in enumerate(document.extracted_pages, start=1):
            page_number = page.page_number or index
            bootstrapped.append(existing_reviews.get(page_number, PageReviewState(page_number=page_number)))
    else:
        bootstrapped.extend(sorted(existing_reviews.values(), key=lambda item: item.page_number))

    document.page_reviews = bootstrapped
    return document


def ensure_page_processing(document: CorrectedDocument) -> CorrectedDocument:
    existing = {state.page_number: state for state in document.page_processing}
    bootstrapped: list[PageProcessingState] = []
    seen_pages: set[int] = set()
    for index, page in enumerate(document.extracted_pages, start=1):
        page_number = page.page_number or index
        seen_pages.add(page_number)
        bootstrapped.append(
            existing.get(
                page_number,
                PageProcessingState(
                    page_number=page_number,
                    status=PageProcessingStatus.OCR_SUCCEEDED,
                ),
            )
        )
    for page_number, state in sorted(existing.items()):
        if page_number not in seen_pages:
            bootstrapped.append(state)
    document.page_processing = bootstrapped
    return document


def load_corrected_document(path: Path, repository: FileRepository) -> CorrectedDocument:
    payload = repository.read_yaml(path)
    document = ensure_page_processing(ensure_page_reviews(CorrectedDocument.model_validate(payload)))
    if document.approval_metadata.review_status == ReviewStatus.APPROVED and not document.approval_metadata.reviewer:
        raise ValueError("Approved corrected documents must include a reviewer.")
    return document
