from __future__ import annotations

from pathlib import Path

from ncai_pipeline.config import PipelineSettings
from ncai_pipeline.ingestion.image_loader import normalize_image
from ncai_pipeline.ingestion.pdf_loader import render_pdf
from ncai_pipeline.schemas.document import DocumentManifest, PageArtifact
from ncai_pipeline.schemas.extraction import NormalizedExtraction


def normalize_document(manifest: DocumentManifest, rendered_pages_dir: Path, settings: PipelineSettings) -> list[PageArtifact]:
    source_path = Path(manifest.scratch_source_path or manifest.source_path)
    if manifest.file_type == "image":
        return normalize_image(manifest.document_id, source_path, rendered_pages_dir)
    if manifest.file_type == "pdf":
        return render_pdf(manifest.document_id, source_path, rendered_pages_dir, settings.pdf_render_scale)
    raise ValueError(f"Unsupported manifest file type: {manifest.file_type}")


def consolidate_text(extractions: list[NormalizedExtraction]) -> str:
    return "\n\n".join(item.full_text.strip() for item in extractions if item.full_text.strip())
