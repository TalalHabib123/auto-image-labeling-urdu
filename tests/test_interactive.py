from __future__ import annotations

import io
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
from urllib.parse import urlencode

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from PIL import Image
from typer.testing import CliRunner

from ncai_pipeline.cli import app
from ncai_pipeline.interactive.service import InteractiveReviewService
from ncai_pipeline.interactive.tui import run_terminal_dashboard
from ncai_pipeline.interactive.web import InteractiveWebController
from ncai_pipeline.parsing.normalizer import normalize_provider_payload
from ncai_pipeline.parsing.validators import load_corrected_document
from ncai_pipeline.schemas.common import ApprovalMetadata, ReviewStatus, ScopeType, TokenUsage
from ncai_pipeline.schemas.extraction import RawExtractionRecord
from ncai_pipeline.schemas.review import CorrectedDocument, PageVerificationStatus, ReviewRecord
from ncai_pipeline.storage.paths import DataPaths
from ncai_pipeline.storage.repository import FileRepository


def reset_logging() -> None:
    logging.shutdown()
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        handler.close()
        root_logger.removeHandler(handler)


def _placeholder_payload() -> dict[str, object]:
    return {
        "page_number": 1,
        "language": None,
        "full_text": "",
        "structured_content": {},
        "blocks": [],
        "lines": [],
        "tables": [],
        "warnings": ["No parseable content returned by provider."],
        "confidence_notes": [],
    }


def _raw_page_payload(workspace_mode: str = "blocks") -> dict[str, object]:
    if workspace_mode == "full_text":
        return {
            "page_number": 1,
            "language": "ur",
            "full_text": "Fallback full text from raw response",
            "structured_content": {"language": "ur", "sections": [{"title": "body", "content": "Fallback"}]},
            "blocks": [],
            "lines": [],
            "tables": [],
            "warnings": [],
            "confidence_notes": [],
        }
    if workspace_mode == "lines":
        return {
            "page_number": 1,
            "language": "ur",
            "full_text": "Ø§ÙˆÙ„ÛŒÙ†\nØ¯ÙˆÙ…ÛŒÙ†",
            "structured_content": {"language": "ur", "sections": [{"title": "body", "content": "Ø§ÙˆÙ„ÛŒÙ†"}]},
            "blocks": [],
            "lines": [
                {
                    "line_id": "l1",
                    "text": "Ø§ÙˆÙ„ÛŒÙ†",
                    "bbox": [20, 40, 80, 220],
                    "confidence_notes": "high",
                },
                {
                    "line_id": "l2",
                    "text": "Ø¯ÙˆÙ…ÛŒÙ†",
                    "bbox": [90, 60, 150, 260],
                    "confidence_notes": "medium",
                },
            ],
            "tables": [],
            "warnings": [],
            "confidence_notes": [],
        }

    return {
        "page_number": 1,
        "language": "ur",
        "full_text": "اصل متن\nدرست حصہ",
        "structured_content": {"language": "ur", "sections": [{"title": "body", "content": "اصل متن"}]},
        "blocks": [
            {
                "block_id": "b1",
                "text": "اصل متن",
                "bbox": [20, 40, 120, 220],
                "confidence_notes": "high",
            },
            {
                "block_id": "b2",
                "text": "درست حصہ",
                "bbox": [160, 60, 280, 300],
                "confidence_notes": "medium",
            },
        ],
        "lines": [],
        "tables": [],
        "warnings": ["Handwritten amount detected."],
        "confidence_notes": [],
    }


def build_interactive_fixture(
    root: Path,
    *,
    placeholder: bool = False,
    manual_full_text: str | None = None,
    workspace_mode: str = "blocks",
) -> tuple[DataPaths, FileRepository]:
    data_paths = DataPaths(root / "data")
    data_paths.ensure()
    repository = FileRepository()

    raw_page_payload = _raw_page_payload(workspace_mode=workspace_mode)
    normalized_from_raw = normalize_provider_payload("doc1", 1, raw_page_payload)
    placeholder_extraction = normalize_provider_payload("doc1", 1, _placeholder_payload())

    corrected_page = placeholder_extraction if placeholder else normalized_from_raw.model_copy(deep=True)
    if manual_full_text is not None:
        corrected_page = normalized_from_raw.model_copy(deep=True)
        corrected_page.full_text = manual_full_text
        if corrected_page.blocks:
            corrected_page.blocks[0].text = manual_full_text

    corrected = CorrectedDocument(
        document_id="doc1",
        approval_metadata=ApprovalMetadata(review_status=ReviewStatus.PENDING, source_version="test"),
        extracted_pages=[corrected_page],
        consolidated_text=corrected_page.full_text,
        target_output={
            "document_id": "doc1",
            "pages": [corrected_page.model_dump(mode="json")],
            "consolidated_text": corrected_page.full_text,
        },
    )
    correction_path = data_paths.correction_files_dir / "doc1" / "corrected.yaml"
    repository.write_yaml(correction_path, corrected)
    repository.write_json(
        data_paths.status_dir / "doc1" / "review_record.json",
        ReviewRecord(
            document_id="doc1",
            review_status=ReviewStatus.PENDING,
            corrected_output_path=str(correction_path),
            raw_output_path=str((data_paths.raw_responses_dir / "doc1").resolve()),
            report_path=str(data_paths.reports_dir / "doc1" / "report.html"),
        ),
    )
    repository.write_json(
        data_paths.manifests_dir / "doc1" / "manifest.json",
        {
            "document_id": "doc1",
            "source_path": "data\\input\\doc1.pdf",
            "file_name": "doc1.pdf",
            "file_type": "pdf",
            "sha256": "abc",
            "page_count": 1,
            "status": "review_pending",
            "created_at": "2026-04-17T00:00:00Z",
            "updated_at": "2026-04-17T00:00:00Z",
            "pipeline_version": "test",
            "duplicate_of": None,
            "error_details": None,
            "warnings": [],
        },
    )

    raw_text = json.dumps([raw_page_payload], ensure_ascii=False)
    raw_record = RawExtractionRecord(
        document_id="doc1",
        scope_type=ScopeType.PAGE,
        page_number=1,
        provider_name="gemini",
        model_name="gemini-3-flash-preview",
        request_prompt_version="ocr_v1",
        request_metadata={"mode": "per_page"},
        raw_response_text=raw_text,
        raw_response_json={"candidates": [{"content": {"parts": [{"text": raw_text}]}}]},
        token_usage=TokenUsage(prompt_tokens=1, output_tokens=1, total_tokens=2),
        latency_ms=10,
        status="succeeded",
    )
    repository.write_json(data_paths.raw_responses_dir / "doc1" / "page_0001.json", raw_record)
    repository.write_json(
        data_paths.parsed_dir / "doc1" / "page_0001.json",
        placeholder_extraction if placeholder else normalized_from_raw,
    )

    repository.write_text(data_paths.reports_dir / "doc1" / "report.html", "<html><body>report</body></html>")
    repository.write_json(data_paths.status_dir / "latest_run.json", {"run_id": "run_test", "processed_documents": 1})
    repository.write_json(
        data_paths.export_target_dir / "export_manifest.json",
        {
            "samples": 1,
            "export_name": "jsonl",
            "generated_at": "2026-04-17T00:00:00Z",
            "stats": {
                "sample_counts": {},
                "page_counts": {},
                "average_text_length": 0.0,
                "language_counts": {},
                "review_coverage": {},
                "error_counts": {},
            },
            "files": {},
        },
    )

    image_dir = data_paths.rendered_pages_dir / "doc1"
    image_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (420, 640), color="white").save(image_dir / "page_0001.png")
    return data_paths, repository


def encode_form(payload: dict[str, object]) -> bytes:
    return urlencode(payload).encode("utf-8")


class InteractiveModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_load_corrected_document_bootstraps_page_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir))
            document = load_corrected_document(data_paths.correction_files_dir / "doc1" / "corrected.yaml", repository)
            self.assertEqual(len(document.page_reviews), 1)
            self.assertEqual(document.page_reviews[0].page_number, 1)
            self.assertEqual(document.page_reviews[0].status.value, "pending")

    def test_service_auto_heals_placeholder_page_from_raw_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir), placeholder=True)
            service = InteractiveReviewService(data_paths, repository)

            context = service.get_document_context("doc1", page_number=1)

            corrected = load_corrected_document(data_paths.correction_files_dir / "doc1" / "corrected.yaml", repository)
            manifest = repository.read_json(data_paths.manifests_dir / "doc1" / "manifest.json")
            parsed = repository.read_json(data_paths.parsed_dir / "doc1" / "page_0001.json")
            self.assertEqual(context.selected_page.full_text, "اصل متن\nدرست حصہ")
            self.assertEqual(corrected.extracted_pages[0].full_text, "اصل متن\nدرست حصہ")
            self.assertEqual(parsed["full_text"], "اصل متن\nدرست حصہ")
            self.assertEqual(corrected.extracted_pages[0].blocks[0].bbox.left, 40.0)
            self.assertEqual(manifest["status"], "review_pending")
            self.assertIsNotNone(context.auto_heal_note)

    def test_service_does_not_auto_heal_non_placeholder_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir), manual_full_text="Manual corrected text")
            service = InteractiveReviewService(data_paths, repository)

            context = service.get_document_context("doc1", page_number=1)

            corrected = load_corrected_document(data_paths.correction_files_dir / "doc1" / "corrected.yaml", repository)
            self.assertEqual(context.selected_page.full_text, "Manual corrected text")
            self.assertEqual(corrected.extracted_pages[0].full_text, "Manual corrected text")
            self.assertIsNone(context.auto_heal_note)

    def test_web_controller_renders_document_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir), placeholder=True)
            controller = InteractiveWebController(InteractiveReviewService(data_paths, repository), "127.0.0.1", 8765)

            document = controller.handle_request("GET", "/documents/doc1?page=1&preview=raw")

            self.assertEqual(document.status_code, 200)
            self.assertIn(b"Mapped Preview", document.body)
            self.assertIn(b"Blocks Workspace", document.body)
            self.assertIn(b"System Alerts", document.body)
            self.assertIn(b"Adjust Bounding Boxes", document.body)
            self.assertIn("اصل متن".encode("utf-8"), document.body)

    def test_preview_and_crop_routes_return_image_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir), placeholder=True)
            controller = InteractiveWebController(InteractiveReviewService(data_paths, repository), "127.0.0.1", 8765)

            preview = controller.handle_request("GET", "/documents/doc1/pages/1/preview?mode=raw")
            crop = controller.handle_request("GET", "/documents/doc1/pages/1/crops/block-1")

            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview.content_type, "image/svg+xml; charset=utf-8")
            self.assertIn(b"<svg", preview.body)
            self.assertNotIn(b"<image href=", preview.body)
            self.assertIn(b"<g id=\"bbox-block-1\"><rect x=\"40\" y=\"20\" width=\"180\" height=\"100\"", preview.body)
            self.assertIn(b"<foreignObject", preview.body)
            self.assertIn(b"overflow:auto", preview.body)
            self.assertEqual(crop.status_code, 200)
            self.assertEqual(crop.content_type, "image/png")
            self.assertTrue(crop.body.startswith(b"\x89PNG"))

    def test_preview_bbox_text_uses_line_level_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(
                Path(temp_dir),
                manual_full_text="ABC 123\nABC اردو 123",
            )
            controller = InteractiveWebController(InteractiveReviewService(data_paths, repository), "127.0.0.1", 8765)

            preview = controller.handle_request("GET", "/documents/doc1/pages/1/preview?mode=corrected")
            body = preview.body.decode("utf-8")

            self.assertEqual(preview.status_code, 200)
            self.assertIn('dir="ltr"', body)
            self.assertIn('dir="rtl"', body)
            self.assertIn("ABC 123", body)
            self.assertIn("ABC اردو 123", body)
            self.assertIn("box-sizing:border-box", body)

    def test_bbox_save_route_persists_bbox_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir))
            controller = InteractiveWebController(InteractiveReviewService(data_paths, repository), "127.0.0.1", 8765)

            response = controller.handle_request(
                "POST",
                "/documents/doc1/pages/1/bbox-save",
                body=encode_form(
                    {
                        "bbox_payload_json": json.dumps({"block-1": {"left": 90, "top": 110, "right": 210, "bottom": 230}}),
                        "reviewer": "qa",
                    }
                ),
            )

            corrected = load_corrected_document(data_paths.correction_files_dir / "doc1" / "corrected.yaml", repository)
            self.assertEqual(response.status_code, 303)
            self.assertIn("Bounding%20boxes%20saved.", response.headers["Location"])
            self.assertEqual(corrected.extracted_pages[0].blocks[0].bbox.left, 90.0)
            self.assertEqual(corrected.extracted_pages[0].blocks[0].bbox.bottom, 230.0)
            self.assertEqual(corrected.target_output["pages"][0]["blocks"][0]["bbox"]["right"], 210.0)

    def test_commit_route_updates_blocks_full_text_and_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir))
            service = InteractiveReviewService(data_paths, repository)
            controller = InteractiveWebController(service, "127.0.0.1", 8765)

            response = controller.handle_request(
                "POST",
                "/documents/doc1/pages/1/commit",
                body=encode_form(
                    {
                        "fragment_0": "ترمیم اول",
                        "fragment_1": "ترمیم دوم",
                        "structured_content_json": json.dumps({"language": "ur", "sections": [{"title": "body"}]}, ensure_ascii=False),
                        "reviewer": "qa",
                        "status": "verified",
                        "comments": "done",
                    }
                ),
            )

            corrected = load_corrected_document(data_paths.correction_files_dir / "doc1" / "corrected.yaml", repository)
            manifest = repository.read_json(data_paths.manifests_dir / "doc1" / "manifest.json")
            self.assertEqual(response.status_code, 303)
            self.assertIn("preview=corrected", response.headers["Location"])
            self.assertEqual(corrected.extracted_pages[0].blocks[0].text, "ترمیم اول")
            self.assertEqual(corrected.extracted_pages[0].blocks[1].text, "ترمیم دوم")
            self.assertEqual(corrected.extracted_pages[0].full_text, "ترمیم اول\n\nترمیم دوم")
            self.assertEqual(corrected.page_reviews[0].status.value, "verified")
            self.assertEqual(manifest["status"], "in_review")
            self.assertEqual(corrected.target_output["pages"][0]["full_text"], "ترمیم اول\n\nترمیم دوم")

    def test_commit_route_updates_block_bboxes_and_target_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir))
            controller = InteractiveWebController(InteractiveReviewService(data_paths, repository), "127.0.0.1", 8765)

            response = controller.handle_request(
                "POST",
                "/documents/doc1/pages/1/commit",
                body=encode_form(
                    {
                        "fragment_0": "اصل متن",
                        "fragment_1": "درست حصہ",
                        "structured_content_json": json.dumps({"language": "ur"}, ensure_ascii=False),
                        "bbox_payload_json": json.dumps({"block-1": {"left": 80, "top": 100, "right": 180, "bottom": 200}}),
                        "reviewer": "qa",
                        "status": "verified",
                    }
                ),
            )

            corrected = load_corrected_document(data_paths.correction_files_dir / "doc1" / "corrected.yaml", repository)
            self.assertEqual(response.status_code, 303)
            self.assertEqual(corrected.extracted_pages[0].blocks[0].bbox.left, 80.0)
            self.assertEqual(corrected.extracted_pages[0].blocks[0].bbox.top, 100.0)
            self.assertEqual(corrected.extracted_pages[0].blocks[0].bbox.right, 180.0)
            self.assertEqual(corrected.extracted_pages[0].blocks[0].bbox.bottom, 200.0)
            self.assertEqual(corrected.target_output["pages"][0]["blocks"][0]["bbox"]["left"], 80.0)
            self.assertEqual(corrected.target_output["pages"][0]["blocks"][0]["bbox"]["bottom"], 200.0)

    def test_commit_route_updates_line_bboxes_in_lines_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir), workspace_mode="lines")
            controller = InteractiveWebController(InteractiveReviewService(data_paths, repository), "127.0.0.1", 8765)

            controller.handle_request(
                "POST",
                "/documents/doc1/pages/1/commit",
                body=encode_form(
                    {
                        "fragment_0": "اولین",
                        "fragment_1": "دومین",
                        "structured_content_json": json.dumps({"language": "ur"}, ensure_ascii=False),
                        "bbox_payload_json": json.dumps({"line-1": {"left": 25, "top": 35, "right": 200, "bottom": 95}}),
                        "reviewer": "qa",
                        "status": "verified",
                    }
                ),
            )

            corrected = load_corrected_document(data_paths.correction_files_dir / "doc1" / "corrected.yaml", repository)
            self.assertEqual(corrected.extracted_pages[0].lines[0].bbox.left, 25.0)
            self.assertEqual(corrected.extracted_pages[0].lines[0].bbox.top, 35.0)
            self.assertEqual(corrected.target_output["pages"][0]["lines"][0]["bbox"]["right"], 200.0)

    def test_commit_route_can_clear_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir))
            controller = InteractiveWebController(InteractiveReviewService(data_paths, repository), "127.0.0.1", 8765)

            controller.handle_request(
                "POST",
                "/documents/doc1/pages/1/commit",
                body=encode_form(
                    {
                        "fragment_0": "اصل متن",
                        "fragment_1": "درست حصہ",
                        "structured_content_json": json.dumps({"language": "ur"}, ensure_ascii=False),
                        "bbox_payload_json": json.dumps({"block-1": None}),
                        "reviewer": "qa",
                        "status": "verified",
                    }
                ),
            )

            corrected = load_corrected_document(data_paths.correction_files_dir / "doc1" / "corrected.yaml", repository)
            self.assertIsNone(corrected.extracted_pages[0].blocks[0].bbox)
            self.assertIsNone(corrected.target_output["pages"][0]["blocks"][0]["bbox"])

    def test_commit_route_clamps_and_normalizes_bbox_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir))
            controller = InteractiveWebController(InteractiveReviewService(data_paths, repository), "127.0.0.1", 8765)

            controller.handle_request(
                "POST",
                "/documents/doc1/pages/1/commit",
                body=encode_form(
                    {
                        "fragment_0": "اصل متن",
                        "fragment_1": "درست حصہ",
                        "structured_content_json": json.dumps({"language": "ur"}, ensure_ascii=False),
                        "bbox_payload_json": json.dumps({"block-1": {"left": 500, "top": 700, "right": -20, "bottom": 10}}),
                        "reviewer": "qa",
                        "status": "verified",
                    }
                ),
            )

            corrected = load_corrected_document(data_paths.correction_files_dir / "doc1" / "corrected.yaml", repository)
            bbox = corrected.extracted_pages[0].blocks[0].bbox
            assert bbox is not None
            self.assertEqual((bbox.left, bbox.top, bbox.right, bbox.bottom), (0.0, 10.0, 420.0, 640.0))

    def test_crop_route_uses_corrected_bbox_after_save(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir))
            controller = InteractiveWebController(InteractiveReviewService(data_paths, repository), "127.0.0.1", 8765)

            controller.handle_request(
                "POST",
                "/documents/doc1/pages/1/commit",
                body=encode_form(
                    {
                        "fragment_0": "اصل متن",
                        "fragment_1": "درست حصہ",
                        "structured_content_json": json.dumps({"language": "ur"}, ensure_ascii=False),
                        "bbox_payload_json": json.dumps({"block-1": {"left": 80, "top": 100, "right": 180, "bottom": 200}}),
                        "reviewer": "qa",
                        "status": "verified",
                    }
                ),
            )

            crop = controller.handle_request("GET", "/documents/doc1/pages/1/crops/block-1")
            with Image.open(io.BytesIO(crop.body)) as image:
                self.assertEqual(image.size, (148, 148))

    def test_saving_bbox_changes_refreshes_static_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir))
            controller = InteractiveWebController(InteractiveReviewService(data_paths, repository), "127.0.0.1", 8765)

            controller.handle_request(
                "POST",
                "/documents/doc1/pages/1/commit",
                body=encode_form(
                    {
                        "fragment_0": "اصل متن",
                        "fragment_1": "درست حصہ",
                        "structured_content_json": json.dumps({"language": "ur"}, ensure_ascii=False),
                        "bbox_payload_json": json.dumps({"block-1": {"left": 88, "top": 110, "right": 188, "bottom": 210}}),
                        "reviewer": "qa",
                        "status": "verified",
                    }
                ),
            )

            report_html = repository.read_text(data_paths.reports_dir / "doc1" / "report.html")
            self.assertIn("Correction Snapshot", report_html)
            self.assertIn("\"left\": 88.0", report_html)

    def test_saving_bbox_changes_moves_approved_document_back_to_in_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir))
            service = InteractiveReviewService(data_paths, repository)
            controller = InteractiveWebController(service, "127.0.0.1", 8765)

            service.set_page_verification("doc1", 1, status=PageVerificationStatus.VERIFIED, reviewer="qa")
            service.set_document_review_status("doc1", status=ReviewStatus.APPROVED, reviewer="qa")

            controller.handle_request(
                "POST",
                "/documents/doc1/pages/1/commit",
                body=encode_form(
                    {
                        "fragment_0": "اصل متن",
                        "fragment_1": "درست حصہ",
                        "structured_content_json": json.dumps({"language": "ur"}, ensure_ascii=False),
                        "bbox_payload_json": json.dumps({"block-1": {"left": 95, "top": 115, "right": 205, "bottom": 225}}),
                        "reviewer": "qa",
                        "status": "verified",
                    }
                ),
            )

            corrected = load_corrected_document(data_paths.correction_files_dir / "doc1" / "corrected.yaml", repository)
            manifest = repository.read_json(data_paths.manifests_dir / "doc1" / "manifest.json")
            self.assertEqual(corrected.approval_metadata.review_status, ReviewStatus.IN_REVIEW)
            self.assertEqual(manifest["status"], "in_review")

    def test_workspace_falls_back_to_full_text_when_no_blocks_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir), workspace_mode="full_text")
            service = InteractiveReviewService(data_paths, repository)

            context = service.get_document_context("doc1", page_number=1)

            self.assertEqual(context.selected_page_workspace.mode, "full_text")
            self.assertEqual(context.selected_page_workspace.crop_count, 0)
            self.assertEqual(len(context.selected_page_workspace.fragments), 1)
            self.assertEqual(context.selected_page_workspace.fragments[0].text, "Fallback full text from raw response")

    def test_full_text_page_disables_bbox_editor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir), workspace_mode="full_text")
            controller = InteractiveWebController(InteractiveReviewService(data_paths, repository), "127.0.0.1", 8765)

            document = controller.handle_request("GET", "/documents/doc1?page=1&preview=corrected")

            self.assertEqual(document.status_code, 200)
            self.assertIn(b"Adjust Bounding Boxes", document.body)
            self.assertIn(b"No bbox-editable fragments exist for this page.", document.body)

    def test_web_commit_persists_manual_handwritten_block_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir))
            service = InteractiveReviewService(data_paths, repository)
            controller = InteractiveWebController(service, "127.0.0.1", 8765)

            response = controller.handle_request(
                "POST",
                "/documents/doc1/pages/1/commit",
                body=(
                    "fragment_0=manual+handwritten&"
                    "fragment_1=printed&"
                    "handwritten_0=true&"
                    "structured_content_json=%7B%22language%22%3A%22ur%22%7D&"
                    "reviewer=qa&status=verified&comments="
                ).encode("utf-8"),
            )

            corrected = load_corrected_document(data_paths.correction_files_dir / "doc1" / "corrected.yaml", repository)
            context = service.get_document_context("doc1", page_number=1)
            self.assertEqual(response.status_code, 303)
            self.assertTrue(corrected.extracted_pages[0].blocks[0].handwritten_manual)
            self.assertFalse(corrected.extracted_pages[0].blocks[1].handwritten_manual)
            self.assertEqual(context.selected_page_workspace.handwritten_count, 1)
            self.assertEqual(context.page_summaries[0].handwritten_count, 1)

    def test_document_approval_updates_manifest_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir))
            service = InteractiveReviewService(data_paths, repository)
            controller = InteractiveWebController(service, "127.0.0.1", 8765)

            service.set_page_verification("doc1", 1, status=PageVerificationStatus.VERIFIED, reviewer="qa")
            response = controller.handle_request(
                "POST",
                "/documents/doc1/approve",
                body="reviewer=qa&status=approved&comments=ship&redirect_page=1".encode("utf-8"),
            )

            manifest = repository.read_json(data_paths.manifests_dir / "doc1" / "manifest.json")
            corrected = load_corrected_document(data_paths.correction_files_dir / "doc1" / "corrected.yaml", repository)
            self.assertEqual(response.status_code, 303)
            self.assertEqual(manifest["status"], "approved")
            self.assertEqual(corrected.approval_metadata.review_status, ReviewStatus.APPROVED)

    def test_non_system_warnings_are_hidden_from_workspace_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir))
            service = InteractiveReviewService(data_paths, repository)

            context = service.get_document_context("doc1", page_number=1)

            self.assertEqual(context.page_summaries[0].warning_count, 0)
            self.assertEqual(service.filter_display_warnings(context.selected_page.warnings), [])

    def test_tui_can_list_and_open_document_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, repository = build_interactive_fixture(Path(temp_dir))
            service = InteractiveReviewService(data_paths, repository)
            output = io.StringIO()
            choices = iter(["l", "d", "doc1", "q"])

            run_terminal_dashboard(
                service,
                document_id=None,
                input_func=lambda prompt: next(choices),
                output_func=lambda text: output.write(text + "\n"),
            )

            rendered = output.getvalue()
            self.assertIn("NCAI Interactive Terminal Dashboard", rendered)
            self.assertIn("doc1", rendered)
            self.assertIn("Command hints:", rendered)

    def test_cli_interactive_web_and_tui_commands_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_paths, _ = build_interactive_fixture(Path(temp_dir))
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "pipeline:",
                        f"  input_dir: '{data_paths.input_dir.as_posix()}'",
                        f"  output_root: '{data_paths.root.as_posix()}'",
                        "provider:",
                        "  name: \"gemini\"",
                        "review:",
                        "  correction_format: \"yaml\"",
                        "  report_title: \"Test Review\"",
                        "export:",
                        "  instruction: \"Test instruction\"",
                        "  split_ratios:",
                        "    train: 0.8",
                        "    val: 0.1",
                        "    test: 0.1",
                        "  target_format: \"jsonl\"",
                        "logging:",
                        "  level: \"INFO\"",
                        "  structured: true",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("ncai_pipeline.cli.serve_web_ui", return_value="http://127.0.0.1:8765") as serve_web:
                web_result = self.runner.invoke(app, ["interactive", "--ui", "web", "--config", str(config_path), "--no-open-browser"])
            reset_logging()
            with patch("ncai_pipeline.cli.run_terminal_dashboard", return_value=None) as run_tui:
                tui_result = self.runner.invoke(app, ["interactive", "--ui", "tui", "--config", str(config_path)])
            reset_logging()

            self.assertEqual(web_result.exit_code, 0)
            self.assertEqual(tui_result.exit_code, 0)
            serve_web.assert_called_once()
            run_tui.assert_called_once()


if __name__ == "__main__":
    unittest.main()
