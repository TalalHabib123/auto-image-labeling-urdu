from __future__ import annotations

from typing import Any

from pydantic import Field

from ncai_pipeline.schemas.common import BaseSchemaModel, BoundingBox, ScopeType, TokenUsage


class ExtractedLine(BaseSchemaModel):
    line_id: str
    text: str
    bbox: BoundingBox | None = None
    confidence_notes: str | None = None


class ExtractedBlock(BaseSchemaModel):
    block_id: str
    text: str
    bbox: BoundingBox | None = None
    confidence_notes: str | None = None
    lines: list[ExtractedLine] = Field(default_factory=list)


class ExtractedTable(BaseSchemaModel):
    table_id: str
    rows: list[list[str]] = Field(default_factory=list)
    confidence_notes: str | None = None


class RawExtractionRecord(BaseSchemaModel):
    document_id: str
    scope_type: ScopeType = ScopeType.PAGE
    page_number: int | None = None
    provider_name: str
    model_name: str
    request_prompt_version: str
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    raw_response_text: str | None = None
    raw_response_json: dict[str, Any] | None = None
    token_usage: TokenUsage | None = None
    latency_ms: int | None = None
    status: str
    error_details: str | None = None
    error_category: str | None = None
    http_status_code: int | None = None
    retry_after_seconds: float | None = None
    attempt_count: int = 1
    retry_count: int = 0
    waited_seconds_total: float = 0.0
    response_headers: dict[str, Any] | None = None
    provider_error_payload: dict[str, Any] | None = None


class NormalizedExtraction(BaseSchemaModel):
    document_id: str
    page_number: int | None = None
    full_text: str = ""
    structured_content: dict[str, Any] = Field(default_factory=dict)
    blocks: list[ExtractedBlock] = Field(default_factory=list)
    lines: list[ExtractedLine] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence_notes: list[str] = Field(default_factory=list)
    parser_version: str
    language: str | None = None
