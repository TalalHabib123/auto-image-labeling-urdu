from __future__ import annotations

from datetime import datetime

from pydantic import Field

from ncai_pipeline.schemas.common import BaseSchemaModel, PipelineStatus


class DocumentManifest(BaseSchemaModel):
    document_id: str
    source_path: str
    file_name: str
    file_type: str
    sha256: str
    page_count: int | None = None
    status: PipelineStatus = PipelineStatus.DISCOVERED
    created_at: datetime
    updated_at: datetime
    pipeline_version: str
    duplicate_of: str | None = None
    error_details: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PageArtifact(BaseSchemaModel):
    document_id: str
    page_number: int
    image_path: str
    width: int
    height: int
    render_metadata: dict[str, object] = Field(default_factory=dict)

