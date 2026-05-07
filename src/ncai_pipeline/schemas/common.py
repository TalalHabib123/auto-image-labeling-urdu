from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PipelineStatus(str, Enum):
    DISCOVERED = "discovered"
    NORMALIZED = "normalized"
    EXTRACTED = "extracted"
    REVIEW_PENDING = "review_pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_RETRY = "needs_retry"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_RETRY = "needs_retry"


class ScopeType(str, Enum):
    PAGE = "page"
    DOCUMENT = "document"


class PageProcessingStatus(str, Enum):
    PENDING = "pending"
    OCR_SUCCEEDED = "ocr_succeeded"
    SKIPPED_EMPTY = "skipped_empty"
    NEEDS_RETRY = "needs_retry"
    FAILED = "failed"


class BaseSchemaModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class BoundingBox(BaseSchemaModel):
    left: float | None = None
    top: float | None = None
    right: float | None = None
    bottom: float | None = None


class TokenUsage(BaseSchemaModel):
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    raw: dict[str, Any] | None = None


class ApprovalMetadata(BaseSchemaModel):
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    review_status: ReviewStatus = ReviewStatus.PENDING
    comments: str | None = None
    source_version: str | None = None
    corrected_by: str | None = None
    corrected_at: datetime | None = None
    is_dry_run: bool = False
    retention_label: str | None = None
    notes: list[str] = Field(default_factory=list)


class PageProcessingState(BaseSchemaModel):
    page_number: int
    status: PageProcessingStatus = PageProcessingStatus.PENDING
    image_path: str | None = None
    width: int | None = None
    height: int | None = None
    blank_detection: dict[str, Any] = Field(default_factory=dict)
    error_summary: str | None = None
    error_category: str | None = None
    http_status_code: int | None = None
    attempt_count: int | None = None
    retry_after_seconds: float | None = None
    raw_response_path: str | None = None
    parsed_path: str | None = None
    processed_at: datetime | None = None
    updated_at: datetime | None = None
