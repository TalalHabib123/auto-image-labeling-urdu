from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from PIL import Image

from ncai_pipeline.config import AppConfig, PipelineSettings
from ncai_pipeline.cli import _write_exports
from ncai_pipeline.ingestion.scanner import discover_documents
from ncai_pipeline.pipeline.orchestrator import PipelineOrchestrator
from ncai_pipeline.providers.base import (
    ExtractionProvider,
    MediaPart,
    ProviderHealth,
    ProviderInvocationResult,
    ProviderRequestEnvelope,
)
from ncai_pipeline.schemas.common import ReviewStatus, TokenUsage
from ncai_pipeline.schemas.review import CorrectedDocument, ReviewRecord
from ncai_pipeline.storage.paths import DataPaths
from ncai_pipeline.storage.repository import FileRepository
from ncai_pipeline.config import EnvironmentSettings, PromptRegistry


class NasFakeProvider(ExtractionProvider):
    provider_name = "fake"

    def __init__(self) -> None:
        self.invocations = 0

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(provider_name=self.provider_name, healthy=True)

    def prepare_input(self, page_paths: list[Path]) -> list[MediaPart]:
        return [
            MediaPart(
                page_number=index,
                path=str(path),
                mime_type="image/png",
                data_base64="",
            )
            for index, path in enumerate(page_paths, start=1)
        ]

    def build_request(
        self,
        document_id: str,
        page_number: int | None,
        prompt_version: str,
        prompt_bundle: dict[str, str],
        media: list[MediaPart],
        mode: str,
    ) -> ProviderRequestEnvelope:
        return ProviderRequestEnvelope(
            document_id=document_id,
            page_number=page_number,
            prompt_version=prompt_version,
            model_name="fake-model",
            request_payload={"media": [item.path for item in media]},
        )

    def invoke(self, request: ProviderRequestEnvelope, dry_run: bool = False) -> ProviderInvocationResult:
        self.invocations += 1
        payload = {
            "page_number": request.page_number or 1,
            "language": "ur",
            "full_text": "nas text",
            "structured_content": {"language": "ur"},
            "blocks": [],
            "lines": [],
            "tables": [],
            "warnings": [],
            "confidence_notes": [],
        }
        return ProviderInvocationResult(
            status="succeeded",
            raw_response_text=json.dumps(payload),
            raw_response_json={"payload": payload},
            token_usage=TokenUsage(prompt_tokens=1, output_tokens=1, total_tokens=2),
            latency_ms=1,
        )

    def parse_response(
        self,
        document_id: str,
        page_number: int | None,
        raw_response_text: str | None,
        raw_response_json: dict | None,
    ) -> dict:
        return json.loads(raw_response_text or "{}")


def nas_config(root: Path, input_root: Path, output_root: Path, scratch_root: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "pipeline": {
                "version": "test",
                "storage_mode": "nas",
                "input_dir": str(root / "ignored_local_input"),
                "output_root": str(root / "ignored_local_output"),
                "process_mode": "per_page",
                "skip_completed": True,
                "prompt_version": "ocr_v1",
            },
            "nas": {
                "input_root": str(input_root),
                "output_root": str(output_root),
                "scratch_root": str(scratch_root),
                "share_label": "temp_nas",
                "input_readiness": {"enabled": True, "stable_check_seconds": 0},
                "scratch": {"cleanup_on_success": True},
                "locks": {"enabled": True, "stale_after_minutes": 240, "heartbeat_seconds": 60},
            },
            "provider": {
                "name": "gemini",
                "model_name": "fake-model",
                "api_base_url": "https://example.invalid",
            },
            "review": {"correction_format": "yaml", "report_title": "NAS Review"},
            "export": {"instruction": "Test", "target_format": "jsonl"},
        }
    )


class NasStorageModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompts = PromptRegistry.model_validate(
            {
                "prompts": {
                    "ocr_v1": {
                        "system_instruction": "system",
                        "extraction_prompt": "prompt",
                    }
                }
            }
        )

    def test_nas_config_validation_requires_roots(self) -> None:
        with self.assertRaises(ValueError):
            AppConfig.model_validate({"pipeline": {"storage_mode": "nas"}})

    def test_nas_config_rejects_scratch_under_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ValueError):
                nas_config(root, root / "input", root / "output", root / "output" / "scratch")

    def test_scanner_records_nested_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "nas_input"
            nested = input_root / "department_a" / "forms"
            nested.mkdir(parents=True)
            Image.new("RGB", (24, 16), color="white").save(nested / "scan.png")

            manifests = discover_documents(
                input_root,
                PipelineSettings(storage_mode="nas"),
                source_root=input_root,
                storage_metadata={"storage_mode": "nas", "nas_share_label": "temp_nas"},
            )

            self.assertEqual(len(manifests), 1)
            self.assertEqual(manifests[0].source_relative_path, "department_a/forms/scan.png")
            self.assertEqual(manifests[0].storage_mode, "nas")
            self.assertEqual(manifests[0].nas_share_label, "temp_nas")

    def test_nas_run_uses_scratch_and_publishes_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "nas_input"
            output_root = root / "nas_output"
            scratch_root = root / "local_scratch"
            input_root.mkdir()
            output_root.mkdir()
            Image.new("RGB", (40, 20), color="white").save(input_root / "doc.png")
            config = nas_config(root, input_root, output_root, scratch_root)
            provider = NasFakeProvider()
            orchestrator = PipelineOrchestrator(
                config,
                self.prompts,
                EnvironmentSettings(),
                DataPaths(output_root),
                provider,
                FileRepository(),
            )

            summary = orchestrator.run(dry_run=True)

            self.assertEqual(summary["storage_mode"], "nas")
            self.assertEqual(summary["processed_documents"], 1)
            data_paths = DataPaths(output_root)
            manifest_paths = list(data_paths.manifests_dir.glob("*/manifest.json"))
            self.assertEqual(len(manifest_paths), 1)
            manifest = FileRepository().read_json(manifest_paths[0])
            document_id = manifest["document_id"]
            self.assertEqual(manifest["source_path"], str(input_root / "doc.png"))
            self.assertTrue(manifest["scratch_source_path"].endswith("doc.png"))
            self.assertEqual(manifest["source_relative_path"], "doc.png")
            self.assertTrue((data_paths.completion_markers_dir / f"{document_id}.complete.json").exists())
            self.assertTrue((data_paths.correction_files_dir / document_id / "corrected.yaml").exists())
            self.assertTrue((data_paths.run_manifests_dir / f"{summary['run_id']}.json").exists())
            self.assertFalse((scratch_root / "active" / str(summary["run_id"])).exists())

    def test_nas_export_builds_in_scratch_and_publishes_manifest_last(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "nas_input"
            output_root = root / "nas_output"
            scratch_root = root / "local_scratch"
            input_root.mkdir()
            output_root.mkdir()
            Image.new("RGB", (40, 20), color="white").save(input_root / "doc.png")
            config = nas_config(root, input_root, output_root, scratch_root)
            provider = NasFakeProvider()
            orchestrator = PipelineOrchestrator(
                config,
                self.prompts,
                EnvironmentSettings(),
                DataPaths(output_root),
                provider,
                FileRepository(),
            )
            orchestrator.run(dry_run=False)
            data_paths = DataPaths(output_root)
            repository = FileRepository()
            review_record_path = next(data_paths.status_dir.glob("*/review_record.json"))
            review_record = ReviewRecord.model_validate(repository.read_json(review_record_path))
            corrected_path = Path(review_record.corrected_output_path)
            corrected = CorrectedDocument.model_validate(repository.read_yaml(corrected_path))
            corrected.approval_metadata.review_status = ReviewStatus.APPROVED
            corrected.approval_metadata.reviewer = "qa"
            review_record.review_status = ReviewStatus.APPROVED
            review_record.reviewer = "qa"
            repository.write_yaml(corrected_path, corrected)
            repository.write_json(review_record_path, review_record)

            sample_count, _, _ = _write_exports(config, data_paths, repository)

            self.assertEqual(sample_count, 1)
            self.assertTrue((data_paths.exports_dir / "export_manifest.json").exists())
            self.assertTrue((data_paths.export_target_dir / "train.jsonl").exists())
            self.assertTrue((data_paths.approved_dir / review_record.document_id / "sample.json").exists())
            self.assertFalse(any((scratch_root / "active").glob("*")) if (scratch_root / "active").exists() else False)


if __name__ == "__main__":
    unittest.main()
