from __future__ import annotations

from pydantic import Field

from ncai_pipeline.schemas.common import ApprovalMetadata, BaseSchemaModel


class CuratedTrainingSample(BaseSchemaModel):
    sample_id: str
    document_id: str
    page_refs: list[int] = Field(default_factory=list)
    input_media_paths: list[str] = Field(default_factory=list)
    task_instruction: str
    target_output: dict[str, object] = Field(default_factory=dict)
    language: str = "ur"
    source_split: str
    approval_metadata: ApprovalMetadata = Field(default_factory=ApprovalMetadata)


class ExportStats(BaseSchemaModel):
    sample_counts: dict[str, int] = Field(default_factory=dict)
    page_counts: dict[str, int] = Field(default_factory=dict)
    average_text_length: float = 0.0
    language_counts: dict[str, int] = Field(default_factory=dict)
    review_coverage: dict[str, int] = Field(default_factory=dict)
    error_counts: dict[str, int] = Field(default_factory=dict)


class ExportManifest(BaseSchemaModel):
    export_name: str
    samples: int
    generated_at: str
    stats: ExportStats
    files: dict[str, str] = Field(default_factory=dict)

