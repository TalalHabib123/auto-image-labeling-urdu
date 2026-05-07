from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from ncai_pipeline.schemas.common import ApprovalMetadata, BaseSchemaModel, PageProcessingState, ReviewStatus
from ncai_pipeline.schemas.extraction import NormalizedExtraction


class PageVerificationStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    NEEDS_FIX = "needs_fix"


class PageReviewState(BaseSchemaModel):
    page_number: int
    status: PageVerificationStatus = PageVerificationStatus.PENDING
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    comments: str | None = None


class ReviewRecord(BaseSchemaModel):
    document_id: str
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewer: str | None = None
    comments: str | None = None
    is_dry_run: bool = False
    retention_label: str | None = None
    corrected_output_path: str
    raw_output_path: str
    report_path: str


class CorrectedDocument(BaseSchemaModel):
    document_id: str
    approval_metadata: ApprovalMetadata = Field(default_factory=ApprovalMetadata)
    extracted_pages: list[NormalizedExtraction] = Field(default_factory=list)
    page_reviews: list[PageReviewState] = Field(default_factory=list)
    page_processing: list[PageProcessingState] = Field(default_factory=list)
    consolidated_text: str = ""
    target_output: dict[str, object] = Field(default_factory=dict)
