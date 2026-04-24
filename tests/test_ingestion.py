from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from PIL import Image

from ncai_pipeline.config import PipelineSettings
from ncai_pipeline.ingestion.pdf_loader import render_pdf
from ncai_pipeline.ingestion.scanner import discover_documents


def create_test_pdf(path: Path) -> None:
    image = Image.new("RGB", (64, 64), color="white")
    image.save(path, format="PDF")


class IngestionTests(unittest.TestCase):
    def test_discover_documents_filters_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            Image.new("RGB", (32, 20), color="white").save(input_dir / "sample.png")
            (input_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

            settings = PipelineSettings(input_dir=str(input_dir))
            manifests = discover_documents(input_dir, settings)

            self.assertEqual(len(manifests), 1)
            manifest = manifests[0]
            self.assertEqual(manifest.file_type, "image")
            self.assertTrue(manifest.document_id.startswith("sample_"))
            self.assertEqual(manifest.source_path, str(input_dir / "sample.png"))

    def test_pdf_render_creates_png_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "fixture.pdf"
            create_test_pdf(pdf_path)
            target_dir = Path(temp_dir) / "rendered"
            artifacts = render_pdf(
                document_id="fixture_doc",
                source_path=pdf_path,
                target_dir=target_dir,
                scale=1.5,
            )

            self.assertEqual(len(artifacts), 1)
            self.assertTrue(Path(artifacts[0].image_path).exists())
            self.assertGreater(artifacts[0].width, 0)
            self.assertGreater(artifacts[0].height, 0)


if __name__ == "__main__":
    unittest.main()
