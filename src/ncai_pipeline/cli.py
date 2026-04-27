from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Callable

import typer

from ncai_pipeline.config import AppConfig, load_app_config, load_environment, load_prompts
from ncai_pipeline.export.dataset_builder import DatasetBuilder, normalize_dataset_ids
from ncai_pipeline.export.writers import write_records
from ncai_pipeline.interactive.service import InteractiveReviewService
from ncai_pipeline.interactive.tui import run_terminal_dashboard
from ncai_pipeline.interactive.web import serve_web_ui
from ncai_pipeline.observability.logging_config import configure_logging
from ncai_pipeline.parsing.validators import load_corrected_document
from ncai_pipeline.pipeline.dry_run_manager import DryRunArtifactManager
from ncai_pipeline.pipeline.orchestrator import PipelineOrchestrator
from ncai_pipeline.providers.factory import build_provider
from ncai_pipeline.schemas.common import ReviewStatus
from ncai_pipeline.schemas.review import ReviewRecord
from ncai_pipeline.storage.paths import DataPaths
from ncai_pipeline.storage.repository import FileRepository


app = typer.Typer(add_completion=False, help="NCAI OCR extraction and curation pipeline")


def _load_runtime(config_path: str | None = None) -> tuple[AppConfig, DataPaths, FileRepository]:
    config = load_app_config(config_path)
    data_paths = DataPaths(Path(config.pipeline.output_root))
    repository = FileRepository()
    data_paths.ensure()
    configure_logging(config.logging.level, data_paths.status_dir / "pipeline.log")
    return config, data_paths, repository


def _build_review_service(config: AppConfig, data_paths: DataPaths, repository: FileRepository) -> InteractiveReviewService:
    return InteractiveReviewService(data_paths, repository, report_title=config.review.report_title)


def _write_exports(
    config: AppConfig,
    data_paths: DataPaths,
    repository: FileRepository,
    include_dry_run: bool = False,
    dataset_types: list[str] | None = None,
    output_formats: list[str] | None = None,
    region_source: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[int, dict[str, Any], object]:
    builder = DatasetBuilder(config, data_paths, repository)
    requested_dataset_types = dataset_types or config.export.dataset_types
    requested_formats = _normalize_output_formats(output_formats or config.export.output_formats)
    requested_region_source = region_source or config.export.region_source
    datasets, manifest = builder.build_datasets(
        dataset_types=requested_dataset_types,
        region_source=requested_region_source,
        include_dry_run=include_dry_run,
        progress_callback=progress_callback,
    )
    dataset_ids = normalize_dataset_ids(requested_dataset_types, requested_region_source)
    files: dict[str, Any] = {}
    split_names = ["train", "val", "test"]
    extension_map = {"jsonl": "jsonl", "json": "json", "csv": "csv", "xlsx": "xlsx"}

    for dataset_id in dataset_ids:
        files[dataset_id] = {}
        records = datasets.get(dataset_id, [])
        for output_format in requested_formats:
            format_dir = data_paths.export_datasets_dir / dataset_id / output_format
            format_files: dict[str, str] = {}
            for split_name in split_names:
                split_records = [record for record in records if record.get("source_split") == split_name]
                file_path = format_dir / f"{split_name}.{extension_map[output_format]}"
                write_records(file_path, split_records, output_format)
                format_files[split_name] = str(file_path)
            files[dataset_id][output_format] = format_files
            format_manifest = manifest.model_copy(deep=True)
            format_manifest.dataset_id = dataset_id
            format_manifest.output_format = output_format
            format_manifest.dataset_types = [dataset_id]
            format_manifest.output_formats = [output_format]
            format_manifest.samples = len(records)
            format_manifest.files = format_files
            repository.write_json(format_dir / "export_manifest.json", format_manifest)

    manifest.dataset_types = dataset_ids
    manifest.output_formats = requested_formats
    manifest.files = files
    manifest.stats.format_counts = {
        output_format: sum(len(records) for records in datasets.values())
        for output_format in requested_formats
    }
    repository.write_json(data_paths.exports_dir / "export_manifest.json", manifest)

    if "full" in datasets and "jsonl" in requested_formats:
        _write_legacy_full_jsonl_aliases(data_paths, repository, datasets["full"], manifest)

    return manifest.samples, files, manifest


def _normalize_output_formats(output_formats: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in output_formats:
        output_format = item.strip().lower()
        if output_format == "excel":
            output_format = "xlsx"
        if output_format not in {"jsonl", "json", "csv", "xlsx"}:
            raise typer.BadParameter("format must be one of: jsonl, json, csv, xlsx")
        if output_format not in normalized:
            normalized.append(output_format)
    return normalized


def _write_legacy_full_jsonl_aliases(
    data_paths: DataPaths,
    repository: FileRepository,
    records: list[dict[str, Any]],
    manifest: object,
) -> None:
    files: dict[str, str] = {}
    for split_name in ["train", "val", "test"]:
        split_records = [record for record in records if record.get("source_split") == split_name]
        file_path = data_paths.export_target_dir / f"{split_name}.jsonl"
        write_records(file_path, split_records, "jsonl")
        files[split_name] = str(file_path)
    legacy_manifest = manifest.model_copy(deep=True) if hasattr(manifest, "model_copy") else manifest
    legacy_manifest.files = files
    legacy_manifest.dataset_types = ["full"]
    legacy_manifest.output_formats = ["jsonl"]
    legacy_manifest.dataset_id = "full"
    legacy_manifest.output_format = "jsonl"
    legacy_manifest.samples = len(records)
    legacy_manifest.stats.dataset_counts = {"full": len(records)}
    legacy_manifest.stats.format_counts = {"jsonl": len(records)}
    repository.write_json(data_paths.export_target_dir / "export_manifest.json", legacy_manifest)


def _format_document_entry(entry: dict[str, Any]) -> str:
    detail_parts = [
        entry["document_id"],
        entry.get("final_status"),
        entry.get("source_path"),
    ]
    if entry.get("page_number") is not None:
        detail_parts.append(f"page={entry['page_number']}")
    if entry.get("error_category"):
        detail_parts.append(f"category={entry['error_category']}")
    if entry.get("http_status_code") is not None:
        detail_parts.append(f"http={entry['http_status_code']}")
    if entry.get("attempt_count") is not None:
        detail_parts.append(f"attempts={entry['attempt_count']}")
    if entry.get("retry_after_seconds") is not None:
        detail_parts.append(f"retry_after={entry['retry_after_seconds']}")
    if entry.get("error_summary"):
        detail_parts.append(f"error={entry['error_summary']}")
    return " | ".join(str(part) for part in detail_parts if part not in (None, ""))


def _emit_run_summary(summary: dict[str, Any], data_paths: DataPaths) -> None:
    typer.echo(f"Run ID: {summary['run_id']}")
    typer.echo(
        "Counts: "
        f"discovered={summary.get('discovered_documents', 0)} "
        f"processed={summary.get('processed_documents', 0)} "
        f"skipped={summary.get('skipped_documents', 0)} "
        f"succeeded={summary.get('succeeded_documents', 0)} "
        f"needs_retry={summary.get('needs_retry_documents', 0)} "
        f"failed={summary.get('failed_documents', 0)}"
    )
    typer.echo(
        f"Performance: average_latency_ms={summary.get('average_latency_ms', 0)} "
        f"retry_count={summary.get('retry_count', 0)}"
    )

    retry_docs = summary.get("documents_needing_retry", [])
    typer.echo(f"Documents needing retry: {len(retry_docs)}")
    for entry in retry_docs:
        typer.echo(f"- {_format_document_entry(entry)}")

    failed_docs = summary.get("documents_failed", [])
    typer.echo(f"Documents failed: {len(failed_docs)}")
    for entry in failed_docs:
        typer.echo(f"- {_format_document_entry(entry)}")

    typer.echo(f"Latest run status: {data_paths.status_dir / 'latest_run.json'}")
    typer.echo(f"Run manifest: {data_paths.status_dir / summary['run_id'] / 'run_manifest.json'}")


def _emit_run_progress(update: dict[str, Any]) -> None:
    event = update["event"]
    if event == "start":
        typer.echo(f"Run progress: 0/{update['total_documents']} documents completed")
        return
    if event == "processing":
        typer.echo(
            f"Run progress: {update['completed_documents']}/{update['total_documents']} | "
            f"processing {update['document_id']}"
        )
        return
    if event in {"skip", "completed"}:
        status = update.get("status", event)
        parts = [
            f"Run progress: {update['completed_documents']}/{update['total_documents']}",
            update["document_id"],
            f"status={status}",
        ]
        if update.get("error_summary"):
            parts.append(f"error={update['error_summary']}")
        typer.echo(" | ".join(parts))


def _emit_export_progress(update: dict[str, Any]) -> None:
    event = update["event"]
    if event == "start":
        typer.echo(f"Export progress: 0/{update['total_records']} review records scanned | included=0")
        return
    if event == "record":
        parts = [
            f"Export progress: {update['completed_records']}/{update['total_records']}",
            update["document_id"],
            f"outcome={update['outcome']}",
            f"included={update['included_samples']}",
        ]
        if update.get("split"):
            parts.append(f"split={update['split']}")
        typer.echo(" | ".join(parts))


@app.command()
def run(
    config_path: Annotated[str | None, typer.Option("--config")] = None,
    input_dir: Annotated[str | None, typer.Option("--input-dir")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    config, data_paths, repository = _load_runtime(config_path)
    environment = load_environment()
    prompts = load_prompts()
    provider = build_provider(config, environment)
    orchestrator = PipelineOrchestrator(config, prompts, environment, data_paths, provider, repository)
    summary = orchestrator.run(
        input_dir=input_dir,
        force=force or config.pipeline.force_reprocess,
        dry_run=dry_run or config.pipeline.dry_run,
        progress_callback=_emit_run_progress,
    )
    _emit_run_summary(summary, data_paths)
    if summary.get("documents_failed"):
        raise typer.Exit(code=1)


@app.command("review-status")
def review_status(
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    _, data_paths, repository = _load_runtime(config_path)
    counter: Counter[str] = Counter()
    for path in sorted(data_paths.status_dir.glob("*/review_record.json")):
        record = ReviewRecord.model_validate(repository.read_json(path))
        counter[record.review_status.value] += 1
    typer.echo(f"Review status counts: {dict(counter)}")


@app.command()
def interactive(
    ui: Annotated[str, typer.Option("--ui", help="web or tui")] = "web",
    config_path: Annotated[str | None, typer.Option("--config")] = None,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8765,
    open_browser: Annotated[bool, typer.Option("--open-browser/--no-open-browser")] = True,
    document_id: Annotated[str | None, typer.Option("--document-id")] = None,
) -> None:
    config, data_paths, repository = _load_runtime(config_path)
    service = _build_review_service(config, data_paths, repository)
    normalized_ui = ui.strip().lower()
    if normalized_ui == "web":
        typer.echo(f"Starting interactive web review at http://{host}:{port}")
        if document_id:
            typer.echo(f"Focused document: {document_id}")
        serve_web_ui(service, host, port, open_browser=open_browser, document_id=document_id)
        return
    if normalized_ui == "tui":
        typer.echo("Starting interactive terminal dashboard")
        run_terminal_dashboard(service, host=host, port=port, document_id=document_id)
        return
    raise typer.BadParameter("ui must be one of: web, tui")


@app.command("pipeline-status")
def pipeline_status(
    status: Annotated[str, typer.Option("--status", help="all or a specific pipeline status")] = "all",
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    _, data_paths, repository = _load_runtime(config_path)
    manifests = []
    counts: Counter[str] = Counter()
    for path in sorted(data_paths.manifests_dir.glob("*/manifest.json")):
        manifest = repository.read_json(path)
        counts[manifest["status"]] += 1
        manifests.append(manifest)

    requested_status = status.strip().lower()
    valid_statuses = {"all", "needs_retry", "failed", "review_pending", "approved", "rejected", "skipped"}
    if requested_status not in valid_statuses:
        raise typer.BadParameter("status must be one of: all, needs_retry, failed, review_pending, approved, rejected, skipped")

    typer.echo(f"Pipeline status counts: {dict(counts)}")
    filtered = manifests if requested_status == "all" else [item for item in manifests if item["status"] == requested_status]
    typer.echo(f"Matching documents: {len(filtered)}")
    for manifest in filtered:
        details = [manifest["document_id"], manifest["status"], manifest["source_path"]]
        if manifest.get("error_details"):
            details.append(f"error={manifest['error_details']}")
        typer.echo(f"- {' | '.join(details)}")


@app.command()
def approve(
    document_id: Annotated[str, typer.Argument()],
    reviewer: Annotated[str, typer.Option("--reviewer")] = "",
    status: Annotated[ReviewStatus, typer.Option("--status")] = ReviewStatus.APPROVED,
    comments: Annotated[str, typer.Option("--comments")] = "",
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    config, data_paths, repository = _load_runtime(config_path)
    correction_path = data_paths.correction_files_dir / document_id / "corrected.yaml"
    if not correction_path.exists():
        raise typer.BadParameter(f"Correction file not found for document: {document_id}")
    load_corrected_document(correction_path, repository)
    service = _build_review_service(config, data_paths, repository)
    try:
        service.set_document_review_status(
            document_id,
            status=status,
            reviewer=reviewer,
            comments=comments,
            require_verified_pages=False,
            auto_verify_pages_on_approve=(status == ReviewStatus.APPROVED),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Updated {document_id} to {status.value}")


@app.command()
def export(
    config_path: Annotated[str | None, typer.Option("--config")] = None,
    include_dry_run: Annotated[bool, typer.Option("--include-dry-run")] = False,
    dataset_types: Annotated[list[str] | None, typer.Option("--dataset-type", help="Dataset type: full, page_text, regions, regions_blocks, regions_sections, handwritten")] = None,
    output_formats: Annotated[list[str] | None, typer.Option("--format", help="Output format: jsonl, json, csv, xlsx")] = None,
    region_source: Annotated[str | None, typer.Option("--region-source", help="Region source for --dataset-type regions: blocks or sections")] = None,
) -> None:
    config, data_paths, repository = _load_runtime(config_path)
    sample_count, _, _ = _write_exports(
        config,
        data_paths,
        repository,
        include_dry_run=include_dry_run,
        dataset_types=dataset_types,
        output_formats=output_formats,
        region_source=region_source,
        progress_callback=_emit_export_progress,
    )
    typer.echo(f"Exported {sample_count} approved sample(s) to {data_paths.export_datasets_dir}")


@app.command("dry-run-artifacts")
def dry_run_artifacts(
    action: Annotated[str, typer.Option("--action", help="inspect, keep, or delete")] = "inspect",
    document_id: Annotated[str | None, typer.Option("--document-id")] = None,
    retention_label: Annotated[str, typer.Option("--retention-label")] = "kept",
    rebuild_export: Annotated[bool, typer.Option("--rebuild-export/--no-rebuild-export")] = True,
    include_dry_run_in_export: Annotated[bool, typer.Option("--include-dry-run-in-export")] = False,
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    config, data_paths, repository = _load_runtime(config_path)
    manager = DryRunArtifactManager(data_paths, repository)
    normalized_action = action.strip().lower()

    if normalized_action == "inspect":
        artifacts = manager.discover(document_id)
        typer.echo(
            {
                "action": "inspect",
                "count": len(artifacts),
                "documents": [artifact.document_id for artifact in artifacts],
            }
        )
        return

    if normalized_action == "keep":
        summary = manager.keep(document_id=document_id, retention_label=retention_label)
    elif normalized_action == "delete":
        summary = manager.delete(document_id=document_id)
    else:
        raise typer.BadParameter("action must be one of: inspect, keep, delete")

    if rebuild_export:
        sample_count, _, manifest = _write_exports(
            config,
            data_paths,
            repository,
            include_dry_run=include_dry_run_in_export,
            progress_callback=_emit_export_progress,
        )
        summary["rebuild_export"] = True
        summary["exported_samples"] = sample_count
        summary["export_error_counts"] = manifest.stats.error_counts # type: ignore
    else:
        summary["rebuild_export"] = False

    typer.echo(summary)


def main() -> None:
    app()
