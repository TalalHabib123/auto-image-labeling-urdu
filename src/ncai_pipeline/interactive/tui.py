from __future__ import annotations

import webbrowser
from typing import Callable

from ncai_pipeline.interactive.service import InteractiveReviewService
from ncai_pipeline.interactive.web import start_background_server


def run_terminal_dashboard(
    service: InteractiveReviewService,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    document_id: str | None = None,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> None:
    server = None
    thread = None
    base_url = f"http://{host}:{port}"

    def ensure_server() -> str:
        nonlocal server, thread, base_url
        if server is None:
            server, thread, base_url = start_background_server(service, host, port)
            output_func(f"Started local web review server at {base_url}")
        return base_url

    try:
        while True:
            dashboard = service.get_dashboard_state()
            output_func("")
            output_func("NCAI Interactive Terminal Dashboard")
            output_func("Pipeline counts: " + str(dashboard["pipeline_counts"]))
            output_func("Review counts: " + str(dashboard["review_counts"]))
            latest_run = dashboard["latest_run"] or {}
            latest_export = dashboard["latest_export"] or {}
            output_func(f"Latest run id: {latest_run.get('run_id', '-')}")
            output_func(f"Latest export samples: {latest_export.get('samples', '-')}")

            if document_id:
                try:
                    context = service.get_document_context(document_id)
                    output_func(
                        f"Focused document: {context.queue_entry.document_id} | "
                        f"pipeline={context.queue_entry.pipeline_status} | "
                        f"review={context.queue_entry.review_status}"
                    )
                    output_func(f"Static report: {context.queue_entry.report_path or '-'}")
                except FileNotFoundError as exc:
                    output_func(str(exc))

            output_func("Options: [l]ist queue  [d]ocument details  [w]eb review  [r]eport  [q]uit")
            choice = input_func("Choice: ").strip().lower()
            if choice in {"q", "quit", "exit"}:
                break
            if choice == "l":
                for item in dashboard["documents"]:
                    output_func(
                        f"- {item.document_id} | pipeline={item.pipeline_status} | "
                        f"review={item.review_status} | source={item.source_path}"
                    )
                continue
            if choice == "d":
                target_id = document_id or input_func("Document ID: ").strip()
                if not target_id:
                    output_func("No document selected.")
                    continue
                try:
                    context = service.get_document_context(target_id)
                except (FileNotFoundError, ValueError) as exc:
                    output_func(str(exc))
                    continue
                output_func(f"Document: {context.queue_entry.document_id}")
                output_func(f"Pipeline: {context.queue_entry.pipeline_status}")
                output_func(f"Review: {context.queue_entry.review_status}")
                output_func(f"Correction file: {context.queue_entry.correction_path or '-'}")
                output_func(f"Static report: {context.queue_entry.report_path or '-'}")
                output_func("Command hints:")
                for label, command in context.command_hints.items():
                    output_func(f"  {label}: {command}")
                continue
            if choice == "w":
                target_id = document_id or input_func("Document ID: ").strip()
                if not target_id:
                    output_func("No document selected.")
                    continue
                url = f"{ensure_server()}/documents/{target_id}"
                webbrowser.open(url)
                output_func(f"Opened web review: {url}")
                continue
            if choice == "r":
                target_id = document_id or input_func("Document ID: ").strip()
                if not target_id:
                    output_func("No document selected.")
                    continue
                report_uri = service.report_uri(target_id)
                if not report_uri:
                    output_func("Static report not found for that document.")
                    continue
                webbrowser.open(report_uri)
                output_func(f"Opened static report: {report_uri}")
                continue
            output_func("Unknown option.")
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
            if thread is not None:
                thread.join(timeout=2)
