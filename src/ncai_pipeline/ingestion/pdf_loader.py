from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from ncai_pipeline.ingestion.page_renderer import save_page_image
from ncai_pipeline.schemas.document import PageArtifact


def render_pdf(document_id: str, source_path: Path, target_dir: Path, scale: float) -> list[PageArtifact]:
    pdf = pdfium.PdfDocument(str(source_path))
    artifacts: list[PageArtifact] = []
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            bitmap = page.render(scale=scale)
            pil_image: Image.Image = bitmap.to_pil()
            target_path = target_dir / document_id / f"page_{index + 1:04d}.png"
            width, height = save_page_image(pil_image, target_path)
            artifacts.append(
                PageArtifact(
                    document_id=document_id,
                    page_number=index + 1,
                    image_path=str(target_path),
                    width=width,
                    height=height,
                    render_metadata={"source_format": "pdf", "scale": scale},
                )
            )
            if hasattr(bitmap, "close"):
                bitmap.close()
            if hasattr(page, "close"):
                page.close()
    finally:
        pdf.close()
    return artifacts
