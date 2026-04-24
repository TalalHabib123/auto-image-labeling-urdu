from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from ncai_pipeline.ingestion.page_renderer import save_page_image
from ncai_pipeline.schemas.document import PageArtifact


def normalize_image(document_id: str, source_path: Path, target_dir: Path) -> list[PageArtifact]:
    with Image.open(source_path) as image:
        normalized = ImageOps.exif_transpose(image)
        target_path = target_dir / document_id / "page_0001.png"
        width, height = save_page_image(normalized, target_path)
    return [
        PageArtifact(
            document_id=document_id,
            page_number=1,
            image_path=str(target_path),
            width=width,
            height=height,
            render_metadata={"source_format": source_path.suffix.lower(), "normalized": True},
        )
    ]

