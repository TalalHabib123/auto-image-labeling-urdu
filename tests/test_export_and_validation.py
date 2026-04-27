from __future__ import annotations

import json
import logging
import tempfile
import unittest
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from PIL import Image
import yaml
from typer.testing import CliRunner

from ncai_pipeline.cli import app
from ncai_pipeline.config import AppConfig
from ncai_pipeline.export.dataset_builder import DatasetBuilder
from ncai_pipeline.parsing.normalizer import normalize_provider_payload
from ncai_pipeline.parsing.validators import load_corrected_document
from ncai_pipeline.pipeline.dry_run_manager import DryRunArtifactManager
from ncai_pipeline.schemas.common import ApprovalMetadata, ReviewStatus
from ncai_pipeline.schemas.review import CorrectedDocument, ReviewRecord
from ncai_pipeline.storage.paths import DataPaths
from ncai_pipeline.storage.repository import FileRepository


def reset_logging() -> None:
    logging.shutdown()
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        handler.close()
        root_logger.removeHandler(handler)


class ExportAndValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_load_corrected_document_requires_reviewer_when_approved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = FileRepository()
            correction_path = Path(temp_dir) / "corrected.yaml"
            corrected = CorrectedDocument(
                document_id="doc1",
                approval_metadata=ApprovalMetadata(review_status=ReviewStatus.APPROVED),
            )
            repository.write_yaml(correction_path, corrected)

            with self.assertRaises(ValueError):
                load_corrected_document(correction_path, repository)

    def test_export_config_defaults_to_full_jsonl(self) -> None:
        config = AppConfig()

        self.assertEqual(config.export.dataset_types, ["full"])
        self.assertEqual(config.export.output_formats, ["jsonl"])
        self.assertEqual(config.export.region_source, "blocks")
        self.assertEqual(config.export.assets.page_strategy, "shared_copy")
        self.assertTrue(config.export.handwritten.auto_detect)

    def test_dataset_builder_exports_approved_only_and_isolates_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_paths = DataPaths(root / "data")
            data_paths.ensure()
            repository = FileRepository()

            extraction = normalize_provider_payload(
                "doc1",
                1,
                {
                    "language": "ur",
                    "full_text": "متن",
                    "structured_content": {"language": "ur"},
                    "blocks": [],
                    "lines": [],
                    "tables": [],
                    "warnings": [],
                    "confidence_notes": [],
                },
            )
            corrected = CorrectedDocument(
                document_id="doc1",
                approval_metadata=ApprovalMetadata(
                    review_status=ReviewStatus.APPROVED,
                    reviewer="qa",
                    is_dry_run=True,
                ),
                extracted_pages=[extraction],
                consolidated_text="متن",
                target_output={"full_text": "متن"},
            )
            correction_path = data_paths.correction_files_dir / "doc1" / "corrected.yaml"
            repository.write_yaml(correction_path, corrected)
            repository.write_json(
                data_paths.status_dir / "doc1" / "review_record.json",
                ReviewRecord(
                    document_id="doc1",
                    review_status=ReviewStatus.APPROVED,
                    reviewer="qa",
                    comments=None,
                    corrected_output_path=str(correction_path),
                    raw_output_path=str(data_paths.raw_responses_dir / "doc1"),
                    report_path=str(data_paths.reports_dir / "doc1" / "report.html"),
                ),
            )
            (data_paths.rendered_pages_dir / "doc1").mkdir(parents=True, exist_ok=True)
            (data_paths.rendered_pages_dir / "doc1" / "page_0001.png").write_bytes(b"fake")

            config = AppConfig.model_validate(
                {
                    "pipeline": {
                        "version": "test",
                        "input_dir": str(data_paths.input_dir),
                        "output_root": str(data_paths.root),
                        "process_mode": "per_page",
                        "skip_completed": True,
                        "prompt_version": "ocr_v1",
                    },
                    "provider": {
                        "name": "gemini",
                        "model_name": "fake-model",
                        "timeout_seconds": 5,
                        "retry_limit": 1,
                        "retry_backoff_seconds": 0.01,
                        "rate_limit_per_minute": 1000,
                        "api_base_url": "https://example.invalid",
                    },
                    "review": {"correction_format": "yaml", "report_title": "Test Review"},
                    "export": {
                        "instruction": "Test instruction",
                        "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                        "target_format": "jsonl",
                    },
                    "logging": {"level": "INFO", "structured": True},
                }
            )

            builder = DatasetBuilder(config, data_paths, repository)
            samples, manifest = builder.build_samples()

            self.assertEqual(len(samples), 0)
            self.assertEqual(manifest.samples, 0)
            self.assertEqual(manifest.stats.error_counts, {"dry_run_excluded": 1})

            included_samples, included_manifest = builder.build_samples(include_dry_run=True)
            self.assertEqual(len(included_samples), 1)
            self.assertEqual(included_samples[0].document_id, "doc1")
            self.assertEqual(included_manifest.samples, 1)
            self.assertTrue((data_paths.dry_run_dir / "doc1" / "sample.json").exists())

    def test_dry_run_manager_keep_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_paths = DataPaths(root / "data")
            data_paths.ensure()
            repository = FileRepository()

            corrected = CorrectedDocument(
                document_id="doc1",
                approval_metadata=ApprovalMetadata(is_dry_run=True, review_status=ReviewStatus.PENDING),
                consolidated_text="",
            )
            correction_path = data_paths.correction_files_dir / "doc1" / "corrected.yaml"
            repository.write_yaml(correction_path, corrected)
            review_record_path = data_paths.status_dir / "doc1" / "review_record.json"
            repository.write_json(
                review_record_path,
                ReviewRecord(
                    document_id="doc1",
                    is_dry_run=True,
                    review_status=ReviewStatus.PENDING,
                    corrected_output_path=str(correction_path),
                    raw_output_path=str(data_paths.raw_responses_dir / "doc1"),
                    report_path=str(data_paths.reports_dir / "doc1" / "report.html"),
                ),
            )
            (data_paths.approved_dir / "doc1").mkdir(parents=True, exist_ok=True)
            (data_paths.approved_dir / "doc1" / "sample.json").write_text("{}", encoding="utf-8")
            (data_paths.manifests_dir / "doc1").mkdir(parents=True, exist_ok=True)
            (data_paths.manifests_dir / "doc1" / "manifest.json").write_text("{}", encoding="utf-8")

            manager = DryRunArtifactManager(data_paths, repository)
            keep_summary = manager.keep(retention_label="archive")
            self.assertEqual(keep_summary["count"], 1)
            kept_record = ReviewRecord.model_validate(repository.read_json(review_record_path))
            self.assertEqual(kept_record.retention_label, "archive")
            self.assertTrue((data_paths.dry_run_dir / "doc1" / "sample.json").exists())

            delete_summary = manager.delete()
            self.assertEqual(delete_summary["count"], 1)
            self.assertFalse((data_paths.correction_files_dir / "doc1").exists())
            self.assertFalse((data_paths.status_dir / "doc1").exists())
            self.assertFalse((data_paths.dry_run_dir / "doc1").exists())

    def test_export_command_shows_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_paths = DataPaths(root / "data")
            data_paths.ensure()
            repository = FileRepository()

            extraction = normalize_provider_payload(
                "doc1",
                1,
                {
                    "language": "ur",
                    "full_text": "متن",
                    "structured_content": {"language": "ur"},
                    "blocks": [],
                    "lines": [],
                    "tables": [],
                    "warnings": [],
                    "confidence_notes": [],
                },
            )
            corrected = CorrectedDocument(
                document_id="doc1",
                approval_metadata=ApprovalMetadata(
                    review_status=ReviewStatus.APPROVED,
                    reviewer="qa",
                ),
                extracted_pages=[extraction],
                consolidated_text="متن",
                target_output={"full_text": "متن"},
            )
            correction_path = data_paths.correction_files_dir / "doc1" / "corrected.yaml"
            repository.write_yaml(correction_path, corrected)
            repository.write_json(
                data_paths.status_dir / "doc1" / "review_record.json",
                ReviewRecord(
                    document_id="doc1",
                    review_status=ReviewStatus.APPROVED,
                    reviewer="qa",
                    comments=None,
                    corrected_output_path=str(correction_path),
                    raw_output_path=str(data_paths.raw_responses_dir / "doc1"),
                    report_path=str(data_paths.reports_dir / "doc1" / "report.html"),
                ),
            )
            (data_paths.rendered_pages_dir / "doc1").mkdir(parents=True, exist_ok=True)
            (data_paths.rendered_pages_dir / "doc1" / "page_0001.png").write_bytes(b"fake")

            config = AppConfig.model_validate(
                {
                    "pipeline": {
                        "version": "test",
                        "input_dir": str(data_paths.input_dir),
                        "output_root": str(data_paths.root),
                        "process_mode": "per_page",
                        "skip_completed": True,
                        "prompt_version": "ocr_v1",
                    },
                    "provider": {
                        "name": "gemini",
                        "model_name": "fake-model",
                        "timeout_seconds": 5,
                        "retry_limit": 1,
                        "retry_backoff_seconds": 0.01,
                        "retry_max_backoff_seconds": 0.05,
                        "rate_limit_per_minute": 1000,
                        "rate_limit_max_retries": 2,
                        "rate_limit_wait_budget_seconds": 0.05,
                        "api_base_url": "https://example.invalid",
                    },
                    "review": {"correction_format": "yaml", "report_title": "Test Review"},
                    "export": {
                        "instruction": "Test instruction",
                        "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                        "target_format": "jsonl",
                    },
                    "logging": {"level": "INFO", "structured": True},
                }
            )
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

            result = self.runner.invoke(app, ["export", "--config", str(config_path)])
            reset_logging()

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Export progress: 0/1 review records scanned | included=0", result.output)
            self.assertIn("outcome=included", result.output)
            self.assertIn("Exported 1 approved sample(s)", result.output)

    def test_export_command_writes_all_dataset_types_formats_and_handwritten_crops(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_paths = DataPaths(root / "data")
            data_paths.ensure()
            repository = FileRepository()

            extraction = normalize_provider_payload(
                "doc1",
                1,
                {
                    "language": "ur",
                    "full_text": "Ù…ØªÙ†",
                    "structured_content": {"language": "ur", "sections": [{"title": "body", "content": "Ù…ØªÙ†"}]},
                    "blocks": [
                        {
                            "block_id": "b1",
                            "text": "Ù…ØªÙ†",
                            "bbox": [20, 10, 80, 100],
                            "confidence_notes": "contains handwritten amount",
                        },
                        {
                            "block_id": "b2",
                            "text": "printed",
                            "bbox": [90, 10, 130, 100],
                            "confidence_notes": "high",
                        },
                        {
                            "block_id": "b3",
                            "text": "missing bbox",
                            "bbox": None,
                            "confidence_notes": "high",
                            "handwritten_manual": True,
                        },
                        {
                            "block_id": "b4",
                            "text": "",
                            "bbox": [140, 10, 170, 100],
                            "confidence_notes": "high",
                            "handwritten_manual": True,
                        },
                    ],
                    "lines": [],
                    "tables": [],
                    "warnings": [],
                    "confidence_notes": [],
                },
            )
            corrected = CorrectedDocument(
                document_id="doc1",
                approval_metadata=ApprovalMetadata(
                    review_status=ReviewStatus.APPROVED,
                    reviewer="qa",
                ),
                extracted_pages=[extraction],
                consolidated_text="Ù…ØªÙ†",
                target_output={"full_text": "Ù…ØªÙ†"},
            )
            correction_path = data_paths.correction_files_dir / "doc1" / "corrected.yaml"
            repository.write_yaml(correction_path, corrected)
            repository.write_json(
                data_paths.status_dir / "doc1" / "review_record.json",
                ReviewRecord(
                    document_id="doc1",
                    review_status=ReviewStatus.APPROVED,
                    reviewer="qa",
                    comments=None,
                    corrected_output_path=str(correction_path),
                    raw_output_path=str(data_paths.raw_responses_dir / "doc1"),
                    report_path=str(data_paths.reports_dir / "doc1" / "report.html"),
                ),
            )
            rendered_dir = data_paths.rendered_pages_dir / "doc1"
            rendered_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (220, 220), color="white").save(rendered_dir / "page_0001.png")

            config = AppConfig.model_validate(
                {
                    "pipeline": {
                        "version": "test",
                        "input_dir": str(data_paths.input_dir),
                        "output_root": str(data_paths.root),
                    },
                    "provider": {"name": "gemini", "model_name": "fake-model", "api_base_url": "https://example.invalid"},
                    "review": {"correction_format": "yaml", "report_title": "Test Review"},
                    "export": {
                        "instruction": "Test instruction",
                        "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                        "target_format": "jsonl",
                    },
                    "logging": {"level": "INFO", "structured": True},
                }
            )
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

            result = self.runner.invoke(
                app,
                [
                    "export",
                    "--config",
                    str(config_path),
                    "--dataset-type",
                    "full",
                    "--dataset-type",
                    "page_text",
                    "--dataset-type",
                    "regions",
                    "--region-source",
                    "blocks",
                    "--dataset-type",
                    "regions_sections",
                    "--dataset-type",
                    "handwritten",
                    "--format",
                    "jsonl",
                    "--format",
                    "json",
                    "--format",
                    "csv",
                    "--format",
                    "xlsx",
                ],
            )
            reset_logging()

            self.assertEqual(result.exit_code, 0, result.output)
            manifest = repository.read_json(data_paths.exports_dir / "export_manifest.json")
            self.assertEqual(manifest["stats"]["dataset_counts"]["page_text"], 1)
            self.assertEqual(manifest["stats"]["dataset_counts"]["regions_blocks"], 1)
            self.assertEqual(manifest["stats"]["dataset_counts"]["regions_sections"], 1)
            self.assertEqual(manifest["stats"]["dataset_counts"]["handwritten"], 1)
            self.assertEqual(manifest["stats"]["skipped_counts"]["handwritten_missing_bbox"], 1)
            self.assertEqual(manifest["stats"]["skipped_counts"]["handwritten_empty_text"], 1)

            self.assertTrue((data_paths.export_page_assets_dir / "doc1" / "page_0001.png").exists())
            crop_path = data_paths.export_crop_assets_dir / "handwritten" / "doc1" / "page_0001" / "b1.png"
            self.assertTrue(crop_path.exists())

            rows = []
            for handwritten_file in manifest["files"]["handwritten"]["jsonl"].values():
                rows.extend(
                    json.loads(line)
                    for line in Path(handwritten_file).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            self.assertEqual(rows[0]["block_id"], "b1")
            self.assertEqual(rows[0]["handwritten_source"], "auto")
            self.assertEqual(Path(rows[0]["crop_path"]), crop_path)

            self.assertTrue((data_paths.export_datasets_dir / "page_text" / "csv" / "train.csv").exists())
            xlsx_path = data_paths.export_datasets_dir / "page_text" / "xlsx" / "train.xlsx"
            self.assertTrue(xlsx_path.exists())
            self.assertTrue(zipfile.is_zipfile(xlsx_path))
            self.assertTrue((data_paths.export_target_dir / "train.jsonl").exists())
            self.assertTrue((data_paths.export_target_dir / "export_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
