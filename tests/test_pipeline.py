from __future__ import annotations

import base64
import json
import logging
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import httpx
import yaml

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from PIL import Image
from typer.testing import CliRunner

from ncai_pipeline.cli import app
from ncai_pipeline.config import AppConfig, EnvironmentSettings, PromptRegistry
from ncai_pipeline.parsing.normalizer import normalize_provider_payload
from ncai_pipeline.pipeline.orchestrator import PipelineOrchestrator
from ncai_pipeline.providers.base import (
    ExtractionProvider,
    MediaPart,
    ProviderHealth,
    ProviderInvocationResult,
    ProviderRequestEnvelope,
)
from ncai_pipeline.providers.gemini_provider import GeminiProvider
from ncai_pipeline.schemas.common import TokenUsage
from ncai_pipeline.storage.paths import DataPaths
from ncai_pipeline.storage.repository import FileRepository


def create_test_pdf(path: Path) -> None:
    image = Image.new("RGB", (64, 64), color="white")
    image.save(path, format="PDF")


def create_multi_page_pdf(path: Path) -> None:
    first = Image.new("RGB", (64, 64), color="white")
    second = Image.new("RGB", (64, 64), color="gray")
    first.save(path, format="PDF", save_all=True, append_images=[second])


def reset_logging() -> None:
    logging.shutdown()
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        handler.close()
        root_logger.removeHandler(handler)


class FakeProvider(ExtractionProvider):
    provider_name = "fake"

    def __init__(self) -> None:
        self.last_media_count = 0

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(provider_name=self.provider_name, healthy=True)

    def prepare_input(self, page_paths: list[Path]) -> list[MediaPart]:
        self.last_media_count = len(page_paths)
        return [
            MediaPart(
                page_number=index,
                path=str(path),
                mime_type="image/png",
                data_base64=base64.b64encode(path.read_bytes()).decode("utf-8"),
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
            request_payload={"mode": mode, "media_count": len(media)},
            request_metadata={"mode": mode, "media_count": len(media)},
        )

    def invoke(self, request: ProviderRequestEnvelope, dry_run: bool = False) -> ProviderInvocationResult:
        if request.request_metadata["mode"] == "per_document":
            payload = {
                "pages": [
                    {
                        "page_number": index + 1,
                        "language": "ur",
                        "full_text": f"page {index + 1}",
                        "structured_content": {"language": "ur"},
                        "blocks": [],
                        "lines": [],
                        "tables": [],
                        "warnings": [],
                        "confidence_notes": [],
                    }
                    for index in range(request.request_metadata["media_count"])
                ]
            }
        else:
            payload = {
                "page_number": request.page_number,
                "language": "ur",
                "full_text": f"page {request.page_number}",
                "structured_content": {"language": "ur"},
                "blocks": [],
                "lines": [],
                "tables": [],
                "warnings": [],
                "confidence_notes": [],
            }
        return ProviderInvocationResult(
            status="succeeded",
            raw_response_text=str(payload).replace("'", '"'),
            raw_response_json={"candidates": [{"content": {"parts": [{"text": str(payload).replace("'", '"')}]}}]},
            token_usage=TokenUsage(prompt_tokens=1, output_tokens=1, total_tokens=2),
            latency_ms=5,
            retry_count=0,
        )

    def parse_response(
        self,
        document_id: str,
        page_number: int | None,
        raw_response_text: str | None,
        raw_response_json: dict | None,
    ) -> dict:
        import json

        return json.loads(raw_response_text or "{}")


class FailingProvider(ExtractionProvider):
    provider_name = "fake-failure"

    def __init__(self, result: ProviderInvocationResult) -> None:
        self.result = result

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
            request_payload={"mode": mode},
            request_metadata={"mode": mode},
        )

    def invoke(self, request: ProviderRequestEnvelope, dry_run: bool = False) -> ProviderInvocationResult:
        return self.result.model_copy(update={"page_number": request.page_number} if hasattr(self.result, "page_number") else {})

    def parse_response(
        self,
        document_id: str,
        page_number: int | None,
        raw_response_text: str | None,
        raw_response_json: dict | None,
    ) -> dict:
        return {}


class SequencedProvider(ExtractionProvider):
    provider_name = "fake-sequence"

    def __init__(self, results: list[ProviderInvocationResult]) -> None:
        self.results = list(results)

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
            request_payload={"mode": mode},
            request_metadata={"mode": mode},
        )

    def invoke(self, request: ProviderRequestEnvelope, dry_run: bool = False) -> ProviderInvocationResult:
        return self.results.pop(0)

    def parse_response(
        self,
        document_id: str,
        page_number: int | None,
        raw_response_text: str | None,
        raw_response_json: dict | None,
    ) -> dict:
        import json

        return json.loads(raw_response_text or "{}")


def build_config(temp_root: Path, input_dir: Path, process_mode: str = "per_page") -> AppConfig:
    return AppConfig.model_validate(
        {
            "pipeline": {
                "version": "test",
                "input_dir": str(input_dir),
                "output_root": str(temp_root / "data"),
                "process_mode": process_mode,
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


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = EnvironmentSettings()
        self.runner = CliRunner()
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

    def test_orchestrator_creates_review_artifacts_and_skips_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            Image.new("RGB", (40, 20), color="white").save(input_dir / "doc.png")
            config = build_config(root, input_dir, process_mode="per_page")
            data_paths = DataPaths(Path(config.pipeline.output_root))
            provider = FakeProvider()
            orchestrator = PipelineOrchestrator(config, self.prompts, self.environment, data_paths, provider, FileRepository())

            first_summary = orchestrator.run()
            second_summary = orchestrator.run()

            self.assertEqual(first_summary["processed_documents"], 1)
            self.assertEqual(second_summary["skipped_documents"], 1)
            correction_files = list((data_paths.correction_files_dir).glob("*/corrected.yaml"))
            self.assertEqual(len(correction_files), 1)

    def test_document_mode_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            target_pdf = input_dir / "fixture.pdf"
            create_test_pdf(target_pdf)
            config = build_config(root, input_dir, process_mode="per_document")
            data_paths = DataPaths(Path(config.pipeline.output_root))
            provider = FakeProvider()
            orchestrator = PipelineOrchestrator(config, self.prompts, self.environment, data_paths, provider, FileRepository())

            summary = orchestrator.run()

            self.assertEqual(summary["processed_documents"], 1)
            parsed_files = list(data_paths.parsed_dir.glob("**/*.json"))
            self.assertTrue(parsed_files)

    def test_orchestrator_records_needs_retry_and_writes_status_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            Image.new("RGB", (40, 20), color="white").save(input_dir / "doc.png")
            config = build_config(root, input_dir)
            data_paths = DataPaths(Path(config.pipeline.output_root))
            provider = FailingProvider(
                ProviderInvocationResult(
                    status="failed",
                    error_details="HTTP 429 | rate limited",
                    error_category="rate_limited",
                    http_status_code=429,
                    retry_after_seconds=5.0,
                    attempt_count=2,
                    retry_count=1,
                    waited_seconds_total=5.0,
                    response_headers={"retry-after": "5"},
                    provider_error_payload={"error": {"message": "rate limited"}},
                )
            )
            orchestrator = PipelineOrchestrator(config, self.prompts, self.environment, data_paths, provider, FileRepository())

            summary = orchestrator.run()

            self.assertEqual(summary["needs_retry_documents"], 1)
            self.assertEqual(len(summary["documents_needing_retry"]), 1)
            manifest = FileRepository().read_json(data_paths.manifests_dir / next(data_paths.manifests_dir.iterdir()).name / "manifest.json")
            self.assertEqual(manifest["status"], "needs_retry")
            self.assertTrue((data_paths.status_dir / summary["run_id"] / "run_manifest.json").exists())
            self.assertTrue((data_paths.status_dir / "latest_run.json").exists())

    def test_partial_page_success_then_failure_marks_document_needs_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            create_multi_page_pdf(input_dir / "fixture.pdf")
            config = build_config(root, input_dir, process_mode="per_page")
            data_paths = DataPaths(Path(config.pipeline.output_root))
            success_payload = {
                "page_number": 1,
                "language": "ur",
                "full_text": "page 1",
                "structured_content": {"language": "ur"},
                "blocks": [],
                "lines": [],
                "tables": [],
                "warnings": [],
                "confidence_notes": [],
            }
            provider = SequencedProvider(
                [
                    ProviderInvocationResult(
                        status="succeeded",
                        raw_response_text='{"page_number": 1, "language": "ur", "full_text": "page 1", "structured_content": {"language": "ur"}, "blocks": [], "lines": [], "tables": [], "warnings": [], "confidence_notes": []}',
                        raw_response_json={"candidates": [{"content": {"parts": [{"text": '{"page_number": 1, "language": "ur", "full_text": "page 1", "structured_content": {"language": "ur"}, "blocks": [], "lines": [], "tables": [], "warnings": [], "confidence_notes": []}'}]}}]},
                        token_usage=TokenUsage(prompt_tokens=1, output_tokens=1, total_tokens=2),
                        latency_ms=5,
                    ),
                    ProviderInvocationResult(
                        status="failed",
                        error_details="HTTP 429 | rate limited",
                        error_category="rate_limited",
                        http_status_code=429,
                        attempt_count=2,
                        retry_count=1,
                    ),
                ]
            )
            orchestrator = PipelineOrchestrator(config, self.prompts, self.environment, data_paths, provider, FileRepository())

            summary = orchestrator.run()

            manifest_paths = list(data_paths.manifests_dir.glob("*/manifest.json"))
            manifest = FileRepository().read_json(manifest_paths[0])
            self.assertEqual(manifest["status"], "needs_retry")
            self.assertEqual(summary["needs_retry_documents"], 1)
            raw_responses = list(data_paths.raw_responses_dir.glob("**/*.json"))
            self.assertGreaterEqual(len(raw_responses), 2)

    def test_orchestrator_records_failed_document_for_non_retryable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            Image.new("RGB", (40, 20), color="white").save(input_dir / "doc.png")
            config = build_config(root, input_dir)
            data_paths = DataPaths(Path(config.pipeline.output_root))
            provider = FailingProvider(
                ProviderInvocationResult(
                    status="failed",
                    error_details="HTTP 400 | invalid request",
                    error_category="http_error",
                    http_status_code=400,
                    attempt_count=1,
                )
            )
            orchestrator = PipelineOrchestrator(config, self.prompts, self.environment, data_paths, provider, FileRepository())

            summary = orchestrator.run()

            self.assertEqual(summary["failed_documents"], 1)
            self.assertEqual(len(summary["documents_failed"]), 1)

    def test_gemini_429_with_retry_after_uses_wait_budget_and_captures_diagnostics(self) -> None:
        config = build_config(Path("."), Path("data/input"))
        provider = GeminiProvider(
            settings=config.provider.model_copy(
                update={
                    "retry_limit": 0,
                    "retry_backoff_seconds": 0.5,
                    "retry_max_backoff_seconds": 5.0,
                    "rate_limit_per_minute": 1000,
                    "rate_limit_max_retries": 2,
                    "rate_limit_wait_budget_seconds": 1.0,
                }
            ),
            environment=EnvironmentSettings(GEMINI_PROJECT_API="secret-value-123"),
        )
        request = ProviderRequestEnvelope(
            document_id="doc-1",
            page_number=1,
            prompt_version="ocr_v1",
            model_name="gemini-3-flash-preview",
            request_payload={"contents": []},
        )
        http_request = httpx.Request("POST", "https://example.test")
        response = httpx.Response(
            429,
            headers={"Retry-After": "3"},
            json={"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota exceeded"}},
            request=http_request,
        )
        fake_time = [100.0]

        def fake_monotonic() -> float:
            return fake_time[0]

        def fake_sleep(seconds: float) -> None:
            fake_time[0] += seconds

        with patch("ncai_pipeline.providers.gemini_provider.httpx.post", side_effect=[response, response]), patch(
            "ncai_pipeline.providers.gemini_provider.time.sleep",
            side_effect=fake_sleep,
        ), patch("ncai_pipeline.providers.gemini_provider.time.monotonic", side_effect=fake_monotonic):
            result = provider.invoke(request)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_category, "rate_limited")
        self.assertEqual(result.http_status_code, 429)
        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(result.retry_after_seconds, 3.0)
        self.assertEqual(result.waited_seconds_total, 1.0)
        self.assertEqual(result.provider_error_payload, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota exceeded"}})
        self.assertEqual(result.response_headers, {"retry-after": "3"})

    def test_gemini_network_errors_are_categorized(self) -> None:
        config = build_config(Path("."), Path("data/input"))
        provider = GeminiProvider(
            settings=config.provider.model_copy(update={"retry_limit": 0, "rate_limit_per_minute": 1000}),
            environment=EnvironmentSettings(GEMINI_PROJECT_API="secret-value-123"),
        )
        request = ProviderRequestEnvelope(
            document_id="doc-1",
            page_number=1,
            prompt_version="ocr_v1",
            model_name="gemini-3-flash-preview",
            request_payload={"contents": []},
        )

        with patch(
            "ncai_pipeline.providers.gemini_provider.httpx.post",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            result = provider.invoke(request)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_category, "network")

    def test_gemini_error_sanitizer_redacts_api_key(self) -> None:
        provider = GeminiProvider(
            settings=build_config(Path("."), Path("data/input")).provider,
            environment=EnvironmentSettings(GEMINI_PROJECT_API="secret-value-123"),
        )
        sanitized = provider._sanitize_error(
            "Client error for url 'https://example.test/path?key=secret-value-123'"
        )
        self.assertNotIn("secret-value-123", sanitized)
        self.assertIn("***REDACTED***", sanitized)

    def test_gemini_parse_response_accepts_top_level_list_for_page_mode(self) -> None:
        provider = GeminiProvider(
            settings=build_config(Path("."), Path("data/input")).provider,
            environment=EnvironmentSettings(),
        )
        raw_payload = [
            {
                "page_number": 1,
                "language": "ur",
                "full_text": "page 1",
                "structured_content": {"language": "ur"},
                "blocks": [
                    {"block_id": "b1", "text": "page 1", "bbox": [10, 20, 50, 120], "confidence_notes": "high"}
                ],
                "lines": [],
                "tables": [],
                "warnings": ["Recovered"],
                "confidence_notes": [],
            }
        ]

        parsed = provider.parse_response(
            document_id="doc-1",
            page_number=1,
            raw_response_text=json.dumps(raw_payload),
            raw_response_json={"candidates": [{"content": {"parts": [{"text": json.dumps(raw_payload)}]}}]},
        )

        self.assertEqual(parsed["page_number"], 1)
        self.assertEqual(parsed["full_text"], "page 1")
        self.assertEqual(parsed["warnings"], ["Recovered"])

    def test_normalizer_accepts_bbox_arrays(self) -> None:
        normalized = normalize_provider_payload(
            "doc-1",
            1,
            {
                "page_number": 1,
                "language": "ur",
                "full_text": "page 1",
                "structured_content": {"language": "ur"},
                "blocks": [
                    {"block_id": "b1", "text": "page 1", "bbox": [10, 20, 50, 120], "confidence_notes": "high"}
                ],
                "lines": [],
                "tables": [],
                "warnings": [],
                "confidence_notes": [],
            },
        )

        self.assertIsNotNone(normalized.blocks[0].bbox)
        self.assertEqual(normalized.blocks[0].bbox.left, 20.0)
        self.assertEqual(normalized.blocks[0].bbox.top, 10.0)
        self.assertEqual(normalized.blocks[0].bbox.right, 120.0)
        self.assertEqual(normalized.blocks[0].bbox.bottom, 50.0)

    def test_valid_list_payload_does_not_emit_no_parseable_warning(self) -> None:
        provider = GeminiProvider(
            settings=build_config(Path("."), Path("data/input")).provider,
            environment=EnvironmentSettings(),
        )
        raw_payload = [
            {
                "page_number": 1,
                "language": "ur",
                "full_text": "page 1",
                "structured_content": {"language": "ur"},
                "blocks": [],
                "lines": [],
                "tables": [],
                "warnings": [],
                "confidence_notes": [],
            }
        ]

        parsed = provider.parse_response(
            document_id="doc-1",
            page_number=1,
            raw_response_text=json.dumps(raw_payload),
            raw_response_json=None,
        )
        normalized = normalize_provider_payload("doc-1", 1, parsed)

        self.assertEqual(normalized.full_text, "page 1")
        self.assertNotIn("No parseable content returned by provider.", normalized.warnings)

    def test_cli_run_reports_needs_retry_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            Image.new("RGB", (40, 20), color="white").save(input_dir / "doc.png")
            config = build_config(root, input_dir)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
            provider = FailingProvider(
                ProviderInvocationResult(
                    status="failed",
                    error_details="HTTP 429 | rate limited",
                    error_category="rate_limited",
                    http_status_code=429,
                    attempt_count=2,
                    retry_count=1,
                )
            )

            with patch("ncai_pipeline.cli.build_provider", return_value=provider):
                result = self.runner.invoke(app, ["run", "--config", str(config_path)])
            reset_logging()

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Run progress: 0/1 documents completed", result.output)
            self.assertIn("processing", result.output)
            self.assertIn("Documents needing retry: 1", result.output)

    def test_cli_run_reports_failed_documents_and_exits_non_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            Image.new("RGB", (40, 20), color="white").save(input_dir / "doc.png")
            config = build_config(root, input_dir)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
            provider = FailingProvider(
                ProviderInvocationResult(
                    status="failed",
                    error_details="HTTP 400 | invalid request",
                    error_category="http_error",
                    http_status_code=400,
                    attempt_count=1,
                )
            )

            with patch("ncai_pipeline.cli.build_provider", return_value=provider):
                result = self.runner.invoke(app, ["run", "--config", str(config_path)])
            reset_logging()

            self.assertEqual(result.exit_code, 1)
            self.assertIn("Run progress: 0/1 documents completed", result.output)
            self.assertIn("Documents failed: 1", result.output)

    def test_pipeline_status_lists_documents_needing_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            Image.new("RGB", (40, 20), color="white").save(input_dir / "doc.png")
            config = build_config(root, input_dir)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
            provider = FailingProvider(
                ProviderInvocationResult(
                    status="failed",
                    error_details="HTTP 429 | rate limited",
                    error_category="rate_limited",
                    http_status_code=429,
                    attempt_count=2,
                    retry_count=1,
                )
            )

            with patch("ncai_pipeline.cli.build_provider", return_value=provider):
                run_result = self.runner.invoke(app, ["run", "--config", str(config_path)])
            reset_logging()
            self.assertEqual(run_result.exit_code, 0)

            status_result = self.runner.invoke(app, ["pipeline-status", "--config", str(config_path), "--status", "needs_retry"])
            reset_logging()

            self.assertEqual(status_result.exit_code, 0)
            self.assertIn("Matching documents: 1", status_result.output)
            self.assertIn("needs_retry", status_result.output)


if __name__ == "__main__":
    unittest.main()
