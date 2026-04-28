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
    storage_mode: str | None = None
    source_root: str | None = None
    source_relative_path: str | None = None
    source_size_bytes: int | None = None
    source_mtime_utc: datetime | None = None
    source_sha256: str | None = None
    scratch_source_path: str | None = None
    scratch_source_sha256: str | None = None
    nas_share_label: str | None = None
    published_at: datetime | None = None
    artifact_version: int | None = None
    completion_marker_path: str | None = None


class PageArtifact(BaseSchemaModel):
    document_id: str
    page_number: int
    image_path: str
    width: int
    height: int
    render_metadata: dict[str, object] = Field(default_factory=dict)
