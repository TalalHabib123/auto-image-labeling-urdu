from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Any, Callable, Iterator

from ncai_pipeline.config import AppConfig, EnvironmentSettings, PromptRegistry
from ncai_pipeline.ingestion.blank_detector import detect_blank_page
from ncai_pipeline.ingestion.scanner import discover_documents
from ncai_pipeline.observability.metrics import PipelineMetrics
from ncai_pipeline.observability.tracing import create_run_context
from ncai_pipeline.parsing.validators import load_corrected_document
from ncai_pipeline.parsing.normalizer import normalize_provider_payload
from ncai_pipeline.pipeline.stages import normalize_document
from ncai_pipeline.pipeline.state_manager import StateManager
from ncai_pipeline.providers.base import ExtractionProvider
from ncai_pipeline.review.artifact_builder import ReviewArtifactBuilder
from ncai_pipeline.review.html_report import render_review_report
from ncai_pipeline.schemas.common import PageProcessingState, PageProcessingStatus, PipelineStatus, ScopeType
from ncai_pipeline.schemas.document import DocumentManifest, PageArtifact
from ncai_pipeline.schemas.extraction import NormalizedExtraction, RawExtractionRecord
from ncai_pipeline.schemas.review import CorrectedDocument, ReviewRecord
from ncai_pipeline.storage.locks import LockConflictError, document_lock, status_lock
from ncai_pipeline.storage.manifests import ManifestStore
from ncai_pipeline.storage.nas import (
    CREATE_ONLY_PUBLISH_STRATEGY,
    NAS_ERROR_CATEGORIES,
    NasOutputIndex,
    NasStorageError,
    cleanup_successful_scratch,
    copy_source_to_scratch,
    document_completion_payload,
    healthcheck_nas_paths,
    ignored_source_path,
    make_input_readiness_checker,
    nas_document_is_complete,
    nas_document_is_complete_create_only,
    publish_document_artifacts,
    publish_document_artifacts_create_only,
    required_document_artifacts,
    retain_failed_scratch,
    retry_operation,
)
from ncai_pipeline.storage.paths import DataPaths, NasDocumentVersionPaths
from ncai_pipeline.storage.raw_store import RawArtifactStore
from ncai_pipeline.storage.repository import FileRepository
from ncai_pipeline.storage.runtime import resolve_storage_runtime
from ncai_pipeline.utils.time import utc_now


LOGGER = logging.getLogger(__name__)


class DocumentProcessingError(RuntimeError):
    def __init__(
        self,
        *,
        final_status: PipelineStatus,
        failed_scope: str,
        page_number: int | None,
        error_summary: str,
        error_category: str | None = None,
        http_status_code: int | None = None,
        attempt_count: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(error_summary)
        self.final_status = final_status
        self.failed_scope = failed_scope
        self.page_number = page_number
        self.error_summary = error_summary
        self.error_category = error_category
        self.http_status_code = http_status_code
        self.attempt_count = attempt_count
        self.retry_after_seconds = retry_after_seconds

    def to_summary_entry(self, manifest: DocumentManifest) -> dict[str, Any]:
        return {
            "document_id": manifest.document_id,
            "source_path": manifest.source_path,
            "final_status": self.final_status.value,
            "failed_scope": self.failed_scope,
            "page_number": self.page_number,
            "error_category": self.error_category,
            "http_status_code": self.http_status_code,
            "attempt_count": self.attempt_count,
            "retry_after_seconds": self.retry_after_seconds,
            "error_summary": self.error_summary,
        }


class PipelineOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        prompts: PromptRegistry,
        environment: EnvironmentSettings,
        data_paths: DataPaths,
        provider: ExtractionProvider,
        repository: FileRepository | None = None,
    ) -> None:
        self.config = config
        self.prompts = prompts
        self.environment = environment
        self.provider = provider
        self.repository = repository or FileRepository()
        self.storage_runtime = resolve_storage_runtime(config)
        initial_paths = self.storage_runtime.final_paths if self.storage_runtime.is_nas else data_paths
        self._configure_paths(initial_paths)
        self.final_paths = self.data_paths
        self.state_manager = StateManager()

    def _configure_paths(self, data_paths: DataPaths) -> None:
        self.data_paths = data_paths
        self.manifests = ManifestStore(self.repository, self.data_paths.manifests_dir)
        self.raw_store = RawArtifactStore(
            self.repository,
            self.data_paths.raw_requests_dir,
            self.data_paths.raw_responses_dir,
            self.data_paths.parsed_dir,
        )
        self.review_builder = ReviewArtifactBuilder(
            self.config.review,
            self.repository,
            self.data_paths.reports_dir,
            self.data_paths.correction_files_dir,
        )

    @contextmanager
    def _using_data_paths(self, data_paths: DataPaths) -> Iterator[None]:
        previous_paths = self.data_paths
        self._configure_paths(data_paths)
        try:
            yield
        finally:
            self._configure_paths(previous_paths)

    def run(
        self,
        input_dir: str | None = None,
        force: bool = False,
        dry_run: bool | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, object]:
        run_context = create_run_context()
        metrics = PipelineMetrics()
        documents_needing_retry: list[dict[str, Any]] = []
        documents_failed: list[dict[str, Any]] = []
        effective_dry_run = self.config.pipeline.dry_run if dry_run is None else dry_run
        prompt = self.prompts.get_prompt(self.config.pipeline.prompt_version)
        prompt_bundle = prompt.model_dump(mode="json")

        if self.storage_runtime.is_nas:
            return self._run_nas(
                run_id=run_context.run_id,
                prompt_bundle=prompt_bundle,
                effective_dry_run=effective_dry_run,
                force=force,
                progress_callback=progress_callback,
            )

        self.data_paths.ensure()
        source_dir = Path(input_dir or self.config.pipeline.input_dir)
        manifests = discover_documents(source_dir, self.config.pipeline)
        metrics.increment("discovered_documents", len(manifests))
        total_documents = len(manifests)
        completed_documents = 0

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "start",
                    "total_documents": total_documents,
                    "completed_documents": completed_documents,
                }
            )

        for manifest in manifests:
            existing = self.manifests.load(manifest.document_id)
            if self.state_manager.should_skip(existing, force, self.config.pipeline.skip_completed):
                metrics.increment("skipped_documents")
                completed_documents += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "skip",
                            "document_id": manifest.document_id,
                            "source_path": manifest.source_path,
                            "completed_documents": completed_documents,
                            "total_documents": total_documents,
                            "status": PipelineStatus.SKIPPED.value,
                            "page_counts": self._page_status_counts(existing.page_processing if existing else []),
                        }
                    )
                continue
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "processing",
                        "document_id": manifest.document_id,
                        "source_path": manifest.source_path,
                        "completed_documents": completed_documents,
                        "total_documents": total_documents,
                    }
                )
            try:
                processed_manifest = self._process_document(
                    manifest,
                    prompt_bundle,
                    effective_dry_run,
                    metrics,
                    force=force,
                    prior_manifest=existing,
                    progress_callback=progress_callback,
                )
                self._record_document_outcome(
                    processed_manifest,
                    metrics,
                    documents_needing_retry,
                    documents_failed,
                )
                completed_documents += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "completed",
                            "document_id": processed_manifest.document_id,
                            "source_path": processed_manifest.source_path,
                            "completed_documents": completed_documents,
                            "total_documents": total_documents,
                            "status": processed_manifest.status.value,
                            "page_counts": self._page_status_counts(processed_manifest.page_processing),
                        }
                    )
            except DocumentProcessingError as exc:
                manifest = self.state_manager.update_status(manifest, exc.final_status, exc.error_summary)
                self.manifests.save(manifest)
                entry = exc.to_summary_entry(manifest)
                if exc.final_status == PipelineStatus.NEEDS_RETRY:
                    documents_needing_retry.append(entry)
                    metrics.increment("needs_retry_documents")
                else:
                    documents_failed.append(entry)
                    metrics.increment("failed_documents")
                LOGGER.warning(
                    "Document %s finished with %s%s",
                    manifest.document_id,
                    exc.final_status.value,
                    f" (page {exc.page_number})" if exc.page_number is not None else "",
                )
                completed_documents += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "completed",
                            "document_id": manifest.document_id,
                            "source_path": manifest.source_path,
                            "completed_documents": completed_documents,
                            "total_documents": total_documents,
                            "status": exc.final_status.value,
                            "error_summary": exc.error_summary,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Document processing failed for %s", manifest.document_id)
                failure_status = self._classify_failure(exc)
                manifest = self.state_manager.update_status(manifest, failure_status, str(exc))
                self.manifests.save(manifest)
                entry = {
                    "document_id": manifest.document_id,
                    "source_path": manifest.source_path,
                    "final_status": failure_status.value,
                    "failed_scope": "document",
                    "page_number": None,
                    "error_category": "unhandled",
                    "http_status_code": None,
                    "attempt_count": None,
                    "retry_after_seconds": None,
                    "error_summary": str(exc),
                }
                if failure_status == PipelineStatus.FAILED:
                    documents_failed.append(entry)
                    metrics.increment("failed_documents")
                else:
                    documents_needing_retry.append(entry)
                    metrics.increment("needs_retry_documents")
                LOGGER.warning("Document %s finished with %s", manifest.document_id, failure_status.value)
                completed_documents += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "completed",
                            "document_id": manifest.document_id,
                            "source_path": manifest.source_path,
                            "completed_documents": completed_documents,
                            "total_documents": total_documents,
                            "status": failure_status.value,
                            "error_summary": str(exc),
                        }
                    )

        summary = metrics.summary()
        summary["run_id"] = run_context.run_id
        summary["storage_mode"] = "local"
        summary["documents_needing_retry"] = documents_needing_retry
        summary["documents_failed"] = documents_failed
        self.repository.write_json(self.data_paths.status_dir / run_context.run_id / "run_manifest.json", summary)
        self.repository.write_json(self.data_paths.run_manifests_dir / f"{run_context.run_id}.json", summary)
        self.repository.write_json(self.data_paths.status_dir / "latest_run.json", summary)
        return summary

    def _run_nas(
        self,
        *,
        run_id: str,
        prompt_bundle: dict[str, str],
        effective_dry_run: bool,
        force: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, object]:
        healthcheck_nas_paths(self.config.nas, self.repository)
        runtime = self.storage_runtime.for_run(run_id)
        assert runtime.scratch_paths is not None
        assert runtime.scratch_root is not None
        final_paths = runtime.final_paths
        scratch_paths = runtime.scratch_paths
        create_only_publish = self.config.nas.publish_strategy == CREATE_ONLY_PUBLISH_STRATEGY
        final_paths.ensure()
        scratch_paths.ensure()
        scratch_paths.sources_dir.mkdir(parents=True, exist_ok=True)

        metrics = PipelineMetrics()
        documents_needing_retry: list[dict[str, Any]] = []
        documents_failed: list[dict[str, Any]] = []
        readiness_checker = make_input_readiness_checker(self.config.nas.input_readiness)
        manifests = discover_documents(
            runtime.input_root,
            self.config.pipeline,
            source_root=runtime.input_root,
            ignored_path=lambda path: ignored_source_path(
                path,
                input_root=runtime.input_root,
                extra_roots=[runtime.output_root, runtime.scratch_root],
            ),
            readiness_checker=readiness_checker,
            storage_metadata=runtime.metadata,
        )
        metrics.increment("discovered_documents", len(manifests))
        total_documents = len(manifests)
        completed_documents = 0
        final_manifest_store = ManifestStore(self.repository, final_paths.manifests_dir)
        output_index = NasOutputIndex(final_paths, self.repository) if create_only_publish else None

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "start",
                    "total_documents": total_documents,
                    "completed_documents": completed_documents,
                }
            )

        for manifest in manifests:
            required_artifacts = required_document_artifacts(manifest.document_id)
            if create_only_publish:
                assert output_index is not None
                existing_payload = output_index.latest_manifest(manifest.document_id)
                existing = DocumentManifest.model_validate(existing_payload) if existing_payload else None
                is_complete = nas_document_is_complete_create_only(
                    final_paths,
                    manifest.document_id,
                    self.repository,
                    source_sha256=manifest.source_sha256 or manifest.sha256,
                )
            else:
                existing = final_manifest_store.load(manifest.document_id)
                is_complete = nas_document_is_complete(
                    final_paths,
                    manifest.document_id,
                    self.repository,
                    required_artifacts=required_artifacts,
                )
            if self.state_manager.should_skip(existing, force, self.config.pipeline.skip_completed) and is_complete:
                metrics.increment("skipped_documents")
                completed_documents += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "skip",
                            "document_id": manifest.document_id,
                            "source_path": manifest.source_path,
                            "completed_documents": completed_documents,
                            "total_documents": total_documents,
                            "status": PipelineStatus.SKIPPED.value,
                            "page_counts": self._page_status_counts(existing.page_processing if existing else []),
                        }
                    )
                continue

            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "processing",
                        "document_id": manifest.document_id,
                        "source_path": manifest.source_path,
                        "completed_documents": completed_documents,
                        "total_documents": total_documents,
                    }
                )

            try:
                with document_lock(
                    final_paths.locks_dir,
                    manifest.document_id,
                    self.repository,
                    run_id=run_id,
                    stale_after_minutes=self.config.nas.locks.stale_after_minutes,
                    enabled=self.config.nas.locks.enabled,
                ):
                    scratch_source, scratch_sha256 = retry_operation(
                        self.config.nas,
                        lambda: copy_source_to_scratch(
                            Path(manifest.source_path),
                            scratch_paths,
                            manifest.document_id,
                            self.repository,
                            verify_checksum=self.config.nas.checksums.verify_after_copy,
                            expected_sha256=manifest.source_sha256 or manifest.sha256,
                        ),
                    )
                    manifest.scratch_source_path = str(scratch_source)
                    manifest.scratch_source_sha256 = scratch_sha256
                    prior_corrected, prior_raw_records = self._load_prior_review_bundle(
                        final_paths,
                        manifest.document_id,
                        create_only_publish=create_only_publish,
                    )
                    if prior_corrected is None:
                        prior_corrected, prior_raw_records = self._load_failed_scratch_review_bundle(
                            runtime.scratch_root,
                            manifest.document_id,
                            scratch_sha256 or manifest.source_sha256 or manifest.sha256,
                        )
                    with self._using_data_paths(scratch_paths):
                        processed_manifest = self._process_document(
                            manifest,
                            prompt_bundle,
                            effective_dry_run,
                            metrics,
                            force=force,
                            prior_manifest=existing,
                            prior_corrected=prior_corrected,
                            prior_raw_records=prior_raw_records,
                            progress_callback=progress_callback,
                        )
                    if create_only_publish:
                        self._prepare_nas_artifacts_for_publish_create_only(
                            processed_manifest,
                            run_id=run_id,
                            scratch_paths=scratch_paths,
                            final_paths=final_paths,
                        )
                        publish_document_artifacts_create_only(
                            run_id=run_id,
                            document_id=manifest.document_id,
                            scratch_paths=scratch_paths,
                            final_paths=final_paths,
                            repository=self.repository,
                            verify_checksums=self.config.nas.checksums.verify_after_publish,
                        )
                    else:
                        self._prepare_nas_artifacts_for_publish(
                            processed_manifest,
                            scratch_paths=scratch_paths,
                            final_paths=final_paths,
                        )
                        completion_payload = document_completion_payload(
                            document_id=manifest.document_id,
                            run_id=run_id,
                            final_paths=final_paths,
                            required_artifacts=required_artifacts,
                            checksums_verified=self.config.nas.checksums.verify_after_publish,
                        )
                        publish_document_artifacts(
                            run_id=run_id,
                            document_id=manifest.document_id,
                            scratch_paths=scratch_paths,
                            final_paths=final_paths,
                            repository=self.repository,
                            required_artifacts=required_artifacts,
                            completion_payload=completion_payload,
                            verify_checksums=self.config.nas.checksums.verify_after_publish,
                        )
                completed_documents += 1
                self._record_document_outcome(
                    processed_manifest,
                    metrics,
                    documents_needing_retry,
                    documents_failed,
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "completed",
                            "document_id": processed_manifest.document_id,
                            "source_path": processed_manifest.source_path,
                            "completed_documents": completed_documents,
                            "total_documents": total_documents,
                            "status": processed_manifest.status.value,
                            "page_counts": self._page_status_counts(processed_manifest.page_processing),
                        }
                    )
            except (DocumentProcessingError, LockConflictError, NasStorageError) as exc:
                manifest = self._nas_failure_manifest(
                    manifest,
                    scratch_paths=scratch_paths,
                    exc=exc,
                )
                if not create_only_publish:
                    final_manifest_store.save(manifest)
                entry = self._nas_failure_entry(manifest, exc)
                if manifest.status == PipelineStatus.NEEDS_RETRY:
                    documents_needing_retry.append(entry)
                    metrics.increment("needs_retry_documents")
                else:
                    documents_failed.append(entry)
                    metrics.increment("failed_documents")
                completed_documents += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "completed",
                            "document_id": manifest.document_id,
                            "source_path": manifest.source_path,
                            "completed_documents": completed_documents,
                            "total_documents": total_documents,
                            "status": manifest.status.value,
                            "error_summary": manifest.error_details,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("NAS document processing failed for %s", manifest.document_id)
                manifest = self._nas_failure_manifest(manifest, scratch_paths=scratch_paths, exc=exc)
                if not create_only_publish:
                    final_manifest_store.save(manifest)
                entry = self._nas_failure_entry(manifest, exc)
                if manifest.status == PipelineStatus.FAILED:
                    documents_failed.append(entry)
                    metrics.increment("failed_documents")
                else:
                    documents_needing_retry.append(entry)
                    metrics.increment("needs_retry_documents")
                completed_documents += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "completed",
                            "document_id": manifest.document_id,
                            "source_path": manifest.source_path,
                            "completed_documents": completed_documents,
                            "total_documents": total_documents,
                            "status": manifest.status.value,
                            "error_summary": manifest.error_details,
                        }
                    )

        summary = metrics.summary()
        summary["run_id"] = run_id
        summary["storage_mode"] = "nas"
        summary["nas"] = runtime.metadata or {}
        summary["documents_needing_retry"] = documents_needing_retry
        summary["documents_failed"] = documents_failed
        self._write_run_summary(final_paths, run_id, summary)
        try:
            if documents_needing_retry or documents_failed:
                retain_failed_scratch(runtime.scratch_root, run_id, self.repository)
            else:
                cleanup_successful_scratch(runtime.scratch_root, run_id, self.config.nas, self.repository)
        except NasStorageError as exc:
            LOGGER.warning("NAS scratch cleanup issue: %s", exc)
            summary.setdefault("warnings", []).append(str(exc))  # type: ignore[union-attr]
        self._configure_paths(final_paths)
        return summary

    def _process_document(
        self,
        manifest: DocumentManifest,
        prompt_bundle: dict[str, str],
        dry_run: bool,
        metrics: PipelineMetrics,
        *,
        force: bool = False,
        prior_manifest: DocumentManifest | None = None,
        prior_corrected: CorrectedDocument | None = None,
        prior_raw_records: list[RawExtractionRecord] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> DocumentManifest:
        prior_corrected = prior_corrected or self._load_local_prior_corrected(manifest.document_id)
        prior_raw_records = prior_raw_records or self._load_raw_records_from_review_bundle(manifest.document_id)
        for raw_record in prior_raw_records:
            if raw_record.scope_type == ScopeType.PAGE:
                self.raw_store.save_raw_response(raw_record)

        manifest = self.state_manager.update_status(manifest, PipelineStatus.DISCOVERED)
        manifest.page_processing = list(prior_manifest.page_processing) if prior_manifest is not None else list(manifest.page_processing)
        self.manifests.save(manifest)

        page_artifacts = normalize_document(manifest, self.data_paths.rendered_pages_dir, self.config.pipeline)
        manifest.page_count = len(page_artifacts)
        manifest = self.state_manager.update_status(manifest, PipelineStatus.NORMALIZED)
        self.manifests.save(manifest)

        if self.config.pipeline.process_mode == "per_document":
            raw_records, normalized_records = self._process_document_scope(
                manifest,
                page_artifacts,
                prompt_bundle,
                dry_run,
                metrics,
            )
            page_processing = [
                PageProcessingState(
                    page_number=page.page_number,
                    status=PageProcessingStatus.OCR_SUCCEEDED,
                    image_path=page.image_path,
                    width=page.width,
                    height=page.height,
                    processed_at=utc_now(),
                    updated_at=utc_now(),
                )
                for page in page_artifacts
            ]
            reset_review_pages = {page.page_number for page in page_artifacts}
        else:
            raw_records, normalized_records, page_processing, reset_review_pages = self._process_page_scope(
                manifest,
                page_artifacts,
                prompt_bundle,
                dry_run,
                metrics,
                force=force,
                prior_corrected=prior_corrected,
                prior_raw_records=prior_raw_records,
                progress_callback=progress_callback,
            )

        manifest = self.state_manager.update_status(manifest, PipelineStatus.EXTRACTED)
        manifest.page_processing = page_processing
        self.manifests.save(manifest)
        review_record = self.review_builder.build(
            document_id=manifest.document_id,
            page_artifacts=page_artifacts,
            raw_records=raw_records,
            extractions=normalized_records,
            pipeline_version=manifest.pipeline_version,
            raw_output_path=self.data_paths.raw_responses_dir / manifest.document_id,
            is_dry_run=dry_run,
            page_processing=page_processing,
            prior_corrected=prior_corrected,
            reset_review_pages=reset_review_pages,
        )
        self.repository.write_json(self.data_paths.status_dir / manifest.document_id / "review_record.json", review_record)
        manifest = self.state_manager.update_status(manifest, self._final_status_from_page_processing(page_processing))
        manifest.page_processing = page_processing
        self.manifests.save(manifest)
        return manifest

    def _write_run_summary(self, data_paths: DataPaths, run_id: str, summary: dict[str, object]) -> None:
        if self.config.pipeline.storage_mode == "nas" and self.config.nas.publish_strategy == CREATE_ONLY_PUBLISH_STRATEGY:
            self.repository.create_json(data_paths.run_manifests_dir / f"{run_id}.json", summary)
            return
        lock_enabled = self.config.pipeline.storage_mode == "nas" and self.config.nas.locks.enabled
        stale_after = self.config.nas.locks.stale_after_minutes if self.config.pipeline.storage_mode == "nas" else 240
        with status_lock(
            data_paths.locks_dir,
            self.repository,
            run_id=run_id,
            stale_after_minutes=stale_after,
            enabled=lock_enabled,
        ):
            self.repository.write_json(data_paths.status_dir / run_id / "run_manifest.json", summary)
            self.repository.write_json(data_paths.run_manifests_dir / f"{run_id}.json", summary)
            self.repository.write_json(data_paths.status_dir / "latest_run.json", summary)

    def _prepare_nas_artifacts_for_publish(
        self,
        manifest: DocumentManifest,
        *,
        scratch_paths: DataPaths,
        final_paths: DataPaths,
    ) -> None:
        manifest.published_at = utc_now()
        manifest.artifact_version = 1
        manifest.completion_marker_path = str(final_paths.completion_markers_dir / f"{manifest.document_id}.complete.json")
        self.repository.write_json(scratch_paths.manifests_dir / manifest.document_id / "manifest.json", manifest)

        review_record_path = scratch_paths.status_dir / manifest.document_id / "review_record.json"
        review_record = ReviewRecord.model_validate(self.repository.read_json(review_record_path))
        review_record.corrected_output_path = str(final_paths.correction_files_dir / manifest.document_id / "corrected.yaml")
        review_record.raw_output_path = str(final_paths.raw_responses_dir / manifest.document_id)
        review_record.report_path = str(final_paths.reports_dir / manifest.document_id / "report.html")
        self.repository.write_json(review_record_path, review_record)

        corrected = CorrectedDocument.model_validate(
            self.repository.read_yaml(scratch_paths.correction_files_dir / manifest.document_id / "corrected.yaml")
        )
        raw_records = self._load_raw_records_for_publish(scratch_paths, manifest.document_id)
        page_image_paths = [
            final_paths.rendered_pages_dir / manifest.document_id / f"page_{page_number:04d}.png"
            for page_number in range(1, (manifest.page_count or len(corrected.extracted_pages)) + 1)
        ]
        report_html = render_review_report(
            title=self.config.review.report_title,
            document_id=manifest.document_id,
            page_image_paths=page_image_paths,
            raw_records=raw_records,
            extractions=corrected.extracted_pages,
            corrected=corrected,
            correction_path=Path(review_record.corrected_output_path),
        )
        self.repository.write_text(scratch_paths.reports_dir / manifest.document_id / "report.html", report_html)

    def _prepare_nas_artifacts_for_publish_create_only(
        self,
        manifest: DocumentManifest,
        *,
        run_id: str,
        scratch_paths: DataPaths,
        final_paths: DataPaths,
    ) -> None:
        version_paths = NasDocumentVersionPaths(final_paths.root, manifest.document_id, run_id)
        manifest.published_at = utc_now()
        manifest.artifact_version = 1
        manifest.completion_marker_path = str(version_paths.completion_marker_path)
        self.repository.write_json(scratch_paths.manifests_dir / manifest.document_id / "manifest.json", manifest)

        review_record_path = scratch_paths.status_dir / manifest.document_id / "review_record.json"
        review_record = ReviewRecord.model_validate(self.repository.read_json(review_record_path))
        review_record.corrected_output_path = str(version_paths.correction_path)
        review_record.raw_output_path = str(version_paths.raw_responses_dir)
        review_record.report_path = str(version_paths.report_path)
        self.repository.write_json(review_record_path, review_record)

        corrected = CorrectedDocument.model_validate(
            self.repository.read_yaml(scratch_paths.correction_files_dir / manifest.document_id / "corrected.yaml")
        )
        raw_records = self._load_raw_records_for_publish(scratch_paths, manifest.document_id)
        page_image_paths = [
            version_paths.rendered_pages_dir / f"page_{page_number:04d}.png"
            for page_number in range(1, (manifest.page_count or len(corrected.extracted_pages)) + 1)
        ]
        report_html = render_review_report(
            title=self.config.review.report_title,
            document_id=manifest.document_id,
            page_image_paths=page_image_paths,
            raw_records=raw_records,
            extractions=corrected.extracted_pages,
            corrected=corrected,
            correction_path=version_paths.correction_path,
        )
        self.repository.write_text(scratch_paths.reports_dir / manifest.document_id / "report.html", report_html)

    def _load_raw_records_for_publish(self, scratch_paths: DataPaths, document_id: str) -> list[RawExtractionRecord]:
        raw_dir = scratch_paths.raw_responses_dir / document_id
        if not raw_dir.exists():
            return []
        return [
            RawExtractionRecord.model_validate(self.repository.read_json(path))
            for path in sorted(raw_dir.glob("*.json"))
        ]

    def _nas_failure_manifest(
        self,
        manifest: DocumentManifest,
        *,
        scratch_paths: DataPaths,
        exc: Exception,
    ) -> DocumentManifest:
        scratch_manifest_path = scratch_paths.manifests_dir / manifest.document_id / "manifest.json"
        if scratch_manifest_path.exists():
            try:
                manifest = DocumentManifest.model_validate(self.repository.read_json(scratch_manifest_path))
            except Exception:  # noqa: BLE001
                pass
        status = self._status_for_nas_exception(exc)
        return self.state_manager.update_status(manifest, status, str(exc))

    def _nas_failure_entry(self, manifest: DocumentManifest, exc: Exception) -> dict[str, Any]:
        if isinstance(exc, DocumentProcessingError):
            return exc.to_summary_entry(manifest)
        category = self._category_for_exception(exc)
        return {
            "document_id": manifest.document_id,
            "source_path": manifest.source_path,
            "final_status": manifest.status.value,
            "failed_scope": "document",
            "page_number": None,
            "error_category": category,
            "http_status_code": None,
            "attempt_count": None,
            "retry_after_seconds": None,
            "error_summary": str(exc),
        }

    def _status_for_nas_exception(self, exc: Exception) -> PipelineStatus:
        if isinstance(exc, DocumentProcessingError):
            return exc.final_status
        if isinstance(exc, LockConflictError):
            return PipelineStatus.NEEDS_RETRY
        if isinstance(exc, NasStorageError):
            return PipelineStatus.NEEDS_RETRY
        return self._classify_failure(exc)

    def _category_for_exception(self, exc: Exception) -> str:
        if isinstance(exc, DocumentProcessingError) and exc.error_category:
            return exc.error_category
        if isinstance(exc, LockConflictError):
            return "nas_lock_conflict"
        if isinstance(exc, NasStorageError):
            return exc.category
        return "unhandled"

    def _load_local_prior_corrected(self, document_id: str) -> CorrectedDocument | None:
        review_record_path = self.data_paths.status_dir / document_id / "review_record.json"
        if not review_record_path.exists():
            return None
        try:
            review_record = ReviewRecord.model_validate(self.repository.read_json(review_record_path))
            return load_corrected_document(Path(review_record.corrected_output_path), self.repository)
        except Exception:  # noqa: BLE001
            return None

    def _load_raw_records_from_review_bundle(self, document_id: str) -> list[RawExtractionRecord]:
        review_record_path = self.data_paths.status_dir / document_id / "review_record.json"
        if not review_record_path.exists():
            return []
        try:
            review_record = ReviewRecord.model_validate(self.repository.read_json(review_record_path))
        except Exception:  # noqa: BLE001
            return []
        return self._load_raw_records_from_path(Path(review_record.raw_output_path))

    def _load_prior_review_bundle(
        self,
        data_paths: DataPaths,
        document_id: str,
        *,
        create_only_publish: bool,
    ) -> tuple[CorrectedDocument | None, list[RawExtractionRecord]]:
        if create_only_publish:
            review_record_path = NasOutputIndex(data_paths, self.repository).latest_review_record_path(document_id)
        else:
            review_record_path = data_paths.status_dir / document_id / "review_record.json"
        if review_record_path is None or not review_record_path.exists():
            return None, []
        try:
            review_record = ReviewRecord.model_validate(self.repository.read_json(review_record_path))
            corrected = load_corrected_document(Path(review_record.corrected_output_path), self.repository)
            raw_records = self._load_raw_records_from_path(Path(review_record.raw_output_path))
            return corrected, raw_records
        except Exception:  # noqa: BLE001
            return None, []

    def _load_failed_scratch_review_bundle(
        self,
        scratch_root: Path,
        document_id: str,
        source_sha256: str | None,
    ) -> tuple[CorrectedDocument | None, list[RawExtractionRecord]]:
        failed_root = scratch_root / "failed"
        if not failed_root.exists():
            return None, []

        candidates: list[tuple[float, DataPaths]] = []
        for manifest_path in failed_root.glob(f"*/working/manifests/{document_id}/manifest.json"):
            try:
                payload = self.repository.read_json(manifest_path)
                failed_manifest = DocumentManifest.model_validate(payload)
            except Exception:  # noqa: BLE001
                continue

            failed_sha256 = failed_manifest.source_sha256 or failed_manifest.sha256 or failed_manifest.scratch_source_sha256
            if source_sha256 and failed_sha256 and failed_sha256 != source_sha256:
                continue

            try:
                run_root = manifest_path.parents[3]
            except IndexError:
                continue
            candidates.append((manifest_path.stat().st_mtime, DataPaths(run_root)))

        for _, failed_paths in sorted(candidates, key=lambda item: item[0], reverse=True):
            review_record_path = failed_paths.status_dir / document_id / "review_record.json"
            corrected_path = failed_paths.correction_files_dir / document_id / "corrected.yaml"
            if not review_record_path.exists() or not corrected_path.exists():
                continue
            try:
                corrected = load_corrected_document(corrected_path, self.repository)
                raw_records = self._load_raw_records_from_path(failed_paths.raw_responses_dir / document_id)
            except Exception:  # noqa: BLE001
                continue
            LOGGER.info("Reusing prior successful OCR pages from failed NAS scratch for %s", document_id)
            return corrected, raw_records

        return None, []

    def _load_raw_records_from_path(self, raw_dir: Path) -> list[RawExtractionRecord]:
        if not raw_dir.exists():
            return []
        records: list[RawExtractionRecord] = []
        for path in sorted(raw_dir.glob("*.json")):
            try:
                records.append(RawExtractionRecord.model_validate(self.repository.read_json(path)))
            except Exception:  # noqa: BLE001
                continue
        return records

    def _page_status_from_invocation(self, error_category: str | None) -> PageProcessingStatus:
        if error_category in {"rate_limited", "network", "timeout", "server_error"}:
            return PageProcessingStatus.NEEDS_RETRY
        return PageProcessingStatus.FAILED

    def _page_is_parse_placeholder(self, page: NormalizedExtraction) -> bool:
        return (
            not page.full_text.strip()
            and not page.structured_content
            and not page.blocks
            and not page.lines
            and not page.tables
            and page.warnings == ["No parseable content returned by provider."]
        )

    def _emit_page_progress(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        manifest: DocumentManifest,
        state: PageProcessingState,
        page_index: int,
        total_pages: int,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": "page",
                "document_id": manifest.document_id,
                "source_path": manifest.source_path,
                "page_number": state.page_number,
                "page_index": page_index,
                "total_pages": total_pages,
                "status": state.status.value,
                "error_category": state.error_category,
                "error_summary": state.error_summary,
            }
        )

    def _final_status_from_page_processing(self, page_processing: list[PageProcessingState]) -> PipelineStatus:
        statuses = {state.status for state in page_processing}
        if PageProcessingStatus.NEEDS_RETRY in statuses:
            return PipelineStatus.NEEDS_RETRY
        if PageProcessingStatus.FAILED in statuses:
            return PipelineStatus.FAILED
        if statuses and statuses == {PageProcessingStatus.SKIPPED_EMPTY}:
            return PipelineStatus.SKIPPED
        return PipelineStatus.REVIEW_PENDING

    def _page_status_counts(self, page_processing: list[PageProcessingState]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for state in page_processing:
            counts[state.status.value] = counts.get(state.status.value, 0) + 1
        counts["reviewable"] = counts.get(PageProcessingStatus.OCR_SUCCEEDED.value, 0)
        return counts

    def _record_document_outcome(
        self,
        manifest: DocumentManifest,
        metrics: PipelineMetrics,
        documents_needing_retry: list[dict[str, Any]],
        documents_failed: list[dict[str, Any]],
    ) -> None:
        metrics.increment("processed_documents")
        for status, key in [
            (PageProcessingStatus.OCR_SUCCEEDED, "ocr_succeeded_pages_total"),
            (PageProcessingStatus.SKIPPED_EMPTY, "skipped_empty_pages_total"),
            (PageProcessingStatus.NEEDS_RETRY, "needs_retry_pages_total"),
            (PageProcessingStatus.FAILED, "failed_pages_total"),
        ]:
            metrics.increment(key, sum(1 for state in manifest.page_processing if state.status == status))
        if manifest.status == PipelineStatus.NEEDS_RETRY:
            metrics.increment("needs_retry_documents")
            documents_needing_retry.extend(self._page_issue_entries(manifest, PageProcessingStatus.NEEDS_RETRY))
            failed_entries = self._page_issue_entries(manifest, PageProcessingStatus.FAILED)
            if failed_entries and failed_entries[0].get("page_number") is not None:
                documents_failed.extend(failed_entries)
            return
        if manifest.status == PipelineStatus.FAILED:
            metrics.increment("failed_documents")
            documents_failed.extend(self._page_issue_entries(manifest, PageProcessingStatus.FAILED))
            return
        if manifest.status == PipelineStatus.SKIPPED:
            metrics.increment("skipped_documents")
            return
        metrics.increment("succeeded_documents")
        metrics.increment("pending_review_documents")

    def _page_issue_entries(
        self,
        manifest: DocumentManifest,
        status: PageProcessingStatus,
    ) -> list[dict[str, Any]]:
        entries = []
        for state in manifest.page_processing:
            if state.status != status:
                continue
            entries.append(
                {
                    "document_id": manifest.document_id,
                    "source_path": manifest.source_path,
                    "final_status": manifest.status.value,
                    "failed_scope": ScopeType.PAGE.value,
                    "page_number": state.page_number,
                    "error_category": state.error_category,
                    "http_status_code": state.http_status_code,
                    "attempt_count": state.attempt_count,
                    "retry_after_seconds": state.retry_after_seconds,
                    "error_summary": state.error_summary,
                }
            )
        if not entries:
            entries.append(
                {
                    "document_id": manifest.document_id,
                    "source_path": manifest.source_path,
                    "final_status": manifest.status.value,
                    "failed_scope": "document",
                    "page_number": None,
                    "error_category": manifest.status.value,
                    "http_status_code": None,
                    "attempt_count": None,
                    "retry_after_seconds": None,
                    "error_summary": manifest.error_details,
                }
            )
        return entries

    def _process_page_scope(
        self,
        manifest: DocumentManifest,
        page_artifacts: list[PageArtifact],
        prompt_bundle: dict[str, str],
        dry_run: bool,
        metrics: PipelineMetrics,
        *,
        force: bool,
        prior_corrected: CorrectedDocument | None,
        prior_raw_records: list[RawExtractionRecord],
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> tuple[list[RawExtractionRecord], list[NormalizedExtraction], list[PageProcessingState], set[int]]:
        raw_by_page = {record.page_number: record for record in prior_raw_records if record.scope_type == ScopeType.PAGE}
        normalized_by_page = {
            (page.page_number or index): page.model_copy(deep=True)
            for index, page in enumerate(prior_corrected.extracted_pages, start=1)
        } if prior_corrected is not None else {}
        prior_state_by_page = {
            state.page_number: state
            for state in (prior_corrected.page_processing if prior_corrected is not None else manifest.page_processing)
        }
        page_processing: list[PageProcessingState] = []
        reset_review_pages: set[int] = set()
        total_pages = len(page_artifacts)

        for page_index, page in enumerate(page_artifacts, start=1):
            page_path = Path(page.image_path)
            processed_at = utc_now()
            try:
                blank_result = detect_blank_page(page_path, self.config.pipeline.blank_page_detection)
                blank_payload = blank_result.model_payload()
            except Exception as exc:  # noqa: BLE001
                status = PageProcessingStatus.NEEDS_RETRY if self._classify_failure(exc) == PipelineStatus.NEEDS_RETRY else PageProcessingStatus.FAILED
                state = PageProcessingState(
                    page_number=page.page_number,
                    status=status,
                    image_path=page.image_path,
                    width=page.width,
                    height=page.height,
                    blank_detection={},
                    error_summary=str(exc),
                    error_category="blank_detection_error",
                    processed_at=processed_at,
                    updated_at=processed_at,
                )
                page_processing.append(state)
                metrics.increment(f"{status.value}_pages")
                self._emit_page_progress(progress_callback, manifest, state, page_index, total_pages)
                continue

            previous_state = prior_state_by_page.get(page.page_number)
            if blank_result.is_blank:
                state = PageProcessingState(
                    page_number=page.page_number,
                    status=PageProcessingStatus.SKIPPED_EMPTY,
                    image_path=page.image_path,
                    width=page.width,
                    height=page.height,
                    blank_detection=blank_payload,
                    processed_at=processed_at,
                    updated_at=processed_at,
                )
                page_processing.append(state)
                metrics.increment("skipped_empty_pages")
                self._emit_page_progress(progress_callback, manifest, state, page_index, total_pages)
                continue

            can_reuse = (
                not force
                and previous_state is not None
                and previous_state.status == PageProcessingStatus.OCR_SUCCEEDED
                and page.page_number in normalized_by_page
                and not self._page_is_parse_placeholder(normalized_by_page[page.page_number])
            )
            if can_reuse:
                prior_state = previous_state.model_copy(deep=True)
                prior_state.image_path = page.image_path
                prior_state.width = page.width
                prior_state.height = page.height
                prior_state.blank_detection = blank_payload
                prior_state.updated_at = processed_at
                normalized = normalized_by_page[page.page_number]
                try:
                    self.raw_store.save_parsed(normalized)
                except Exception as exc:  # noqa: BLE001
                    status = PageProcessingStatus.NEEDS_RETRY if self._classify_failure(exc) == PipelineStatus.NEEDS_RETRY else PageProcessingStatus.FAILED
                    state = PageProcessingState(
                        page_number=page.page_number,
                        status=status,
                        image_path=page.image_path,
                        width=page.width,
                        height=page.height,
                        blank_detection=blank_payload,
                        error_summary=str(exc),
                        error_category=self._category_for_exception(exc),
                        processed_at=processed_at,
                        updated_at=processed_at,
                    )
                    page_processing.append(state)
                    metrics.increment(f"{status.value}_pages")
                    self._emit_page_progress(progress_callback, manifest, state, page_index, total_pages)
                    continue
                page_processing.append(prior_state)
                metrics.increment("reused_ocr_pages")
                self._emit_page_progress(progress_callback, manifest, prior_state, page_index, total_pages)
                continue

            raw_response_path: Path | None = None
            invocation = None
            try:
                media = self.provider.prepare_input([page_path])
                request = self.provider.build_request(
                    document_id=manifest.document_id,
                    page_number=page.page_number,
                    prompt_version=self.config.pipeline.prompt_version,
                    prompt_bundle=prompt_bundle,
                    media=media,
                    mode=self.config.pipeline.process_mode,
                )
                self.raw_store.save_request(manifest.document_id, page.page_number, request.model_dump(mode="json"))
                invocation = self.provider.invoke(request, dry_run=dry_run)
                raw_record = RawExtractionRecord(
                    document_id=manifest.document_id,
                    scope_type=ScopeType.PAGE,
                    page_number=page.page_number,
                    provider_name=self.provider.provider_name,
                    model_name=request.model_name,
                    request_prompt_version=request.prompt_version,
                    request_metadata=request.request_metadata,
                    raw_response_text=invocation.raw_response_text,
                    raw_response_json=invocation.raw_response_json,
                    token_usage=invocation.token_usage,
                    latency_ms=invocation.latency_ms,
                    status=invocation.status,
                    error_details=invocation.error_details,
                    error_category=invocation.error_category,
                    http_status_code=invocation.http_status_code,
                    retry_after_seconds=invocation.retry_after_seconds,
                    attempt_count=invocation.attempt_count,
                    retry_count=invocation.retry_count,
                    waited_seconds_total=invocation.waited_seconds_total,
                    response_headers=invocation.response_headers,
                    provider_error_payload=invocation.provider_error_payload,
                )
                raw_response_path = self.raw_store.save_raw_response(raw_record)
                raw_by_page[page.page_number] = raw_record
                metrics.add_latency(invocation.latency_ms)
                metrics.add_retries(invocation.retry_count)

                if invocation.status == "failed":
                    status = self._page_status_from_invocation(invocation.error_category)
                    state = PageProcessingState(
                        page_number=page.page_number,
                        status=status,
                        image_path=page.image_path,
                        width=page.width,
                        height=page.height,
                        blank_detection=blank_payload,
                        error_summary=invocation.error_details or "Provider invocation failed",
                        error_category=invocation.error_category,
                        http_status_code=invocation.http_status_code,
                        attempt_count=invocation.attempt_count,
                        retry_after_seconds=invocation.retry_after_seconds,
                        raw_response_path=str(raw_response_path),
                        processed_at=processed_at,
                        updated_at=processed_at,
                    )
                    page_processing.append(state)
                    metrics.increment(f"{status.value}_pages")
                    self._emit_page_progress(progress_callback, manifest, state, page_index, total_pages)
                    continue

                try:
                    parsed_payload = self.provider.parse_response(
                        document_id=manifest.document_id,
                        page_number=page.page_number,
                        raw_response_text=invocation.raw_response_text,
                        raw_response_json=invocation.raw_response_json,
                    )
                except Exception as exc:  # noqa: BLE001
                    state = PageProcessingState(
                        page_number=page.page_number,
                        status=PageProcessingStatus.FAILED,
                        image_path=page.image_path,
                        width=page.width,
                        height=page.height,
                        blank_detection=blank_payload,
                        error_summary=str(exc),
                        error_category="parse_error",
                        http_status_code=invocation.http_status_code,
                        attempt_count=invocation.attempt_count,
                        retry_after_seconds=invocation.retry_after_seconds,
                        raw_response_path=str(raw_response_path),
                        processed_at=processed_at,
                        updated_at=processed_at,
                    )
                    page_processing.append(state)
                    metrics.increment("failed_pages")
                    self._emit_page_progress(progress_callback, manifest, state, page_index, total_pages)
                    continue

                try:
                    normalized = normalize_provider_payload(manifest.document_id, page.page_number, parsed_payload)
                except Exception as exc:  # noqa: BLE001
                    state = PageProcessingState(
                        page_number=page.page_number,
                        status=PageProcessingStatus.FAILED,
                        image_path=page.image_path,
                        width=page.width,
                        height=page.height,
                        blank_detection=blank_payload,
                        error_summary=str(exc),
                        error_category="normalization_error",
                        http_status_code=invocation.http_status_code,
                        attempt_count=invocation.attempt_count,
                        retry_after_seconds=invocation.retry_after_seconds,
                        raw_response_path=str(raw_response_path),
                        processed_at=processed_at,
                        updated_at=processed_at,
                    )
                    page_processing.append(state)
                    metrics.increment("failed_pages")
                    self._emit_page_progress(progress_callback, manifest, state, page_index, total_pages)
                    continue

                parsed_path = self.raw_store.save_parsed(normalized)
            except Exception as exc:  # noqa: BLE001
                status = PageProcessingStatus.NEEDS_RETRY if self._classify_failure(exc) == PipelineStatus.NEEDS_RETRY else PageProcessingStatus.FAILED
                state = PageProcessingState(
                    page_number=page.page_number,
                    status=status,
                    image_path=page.image_path,
                    width=page.width,
                    height=page.height,
                    blank_detection=blank_payload,
                    error_summary=str(exc),
                    error_category=self._category_for_exception(exc),
                    http_status_code=invocation.http_status_code if invocation is not None else None,
                    attempt_count=invocation.attempt_count if invocation is not None else None,
                    retry_after_seconds=invocation.retry_after_seconds if invocation is not None else None,
                    raw_response_path=str(raw_response_path) if raw_response_path is not None else None,
                    processed_at=processed_at,
                    updated_at=processed_at,
                )
                page_processing.append(state)
                metrics.increment(f"{status.value}_pages")
                self._emit_page_progress(progress_callback, manifest, state, page_index, total_pages)
                continue

            if self._page_is_parse_placeholder(normalized) and not dry_run:
                state = PageProcessingState(
                    page_number=page.page_number,
                    status=PageProcessingStatus.NEEDS_RETRY,
                    image_path=page.image_path,
                    width=page.width,
                    height=page.height,
                    blank_detection=blank_payload,
                    error_summary="Provider returned no parseable content for a non-empty page.",
                    error_category="empty_ocr_result",
                    raw_response_path=str(raw_response_path),
                    parsed_path=str(parsed_path),
                    processed_at=processed_at,
                    updated_at=processed_at,
                )
                page_processing.append(state)
                metrics.increment("needs_retry_pages")
                self._emit_page_progress(progress_callback, manifest, state, page_index, total_pages)
                continue
            normalized_by_page[page.page_number] = normalized
            reset_review_pages.add(page.page_number)
            state = PageProcessingState(
                page_number=page.page_number,
                status=PageProcessingStatus.OCR_SUCCEEDED,
                image_path=page.image_path,
                width=page.width,
                height=page.height,
                blank_detection=blank_payload,
                raw_response_path=str(raw_response_path),
                parsed_path=str(parsed_path),
                processed_at=processed_at,
                updated_at=processed_at,
            )
            page_processing.append(state)
            metrics.increment("ocr_succeeded_pages")
            self._emit_page_progress(progress_callback, manifest, state, page_index, total_pages)

        ordered_pages = [
            page
            for _, page in sorted(normalized_by_page.items())
            if any(
                state.page_number == (page.page_number or 0)
                and state.status == PageProcessingStatus.OCR_SUCCEEDED
                for state in page_processing
            )
        ]
        raw_records = [record for _, record in sorted(raw_by_page.items())]
        return raw_records, ordered_pages, page_processing, reset_review_pages

    def _process_document_scope(
        self,
        manifest: DocumentManifest,
        page_artifacts: list[PageArtifact],
        prompt_bundle: dict[str, str],
        dry_run: bool,
        metrics: PipelineMetrics,
    ) -> tuple[list[RawExtractionRecord], list[NormalizedExtraction]]:
        page_paths = [Path(page.image_path) for page in page_artifacts]
        media = self.provider.prepare_input(page_paths)
        request = self.provider.build_request(
            document_id=manifest.document_id,
            page_number=None,
            prompt_version=self.config.pipeline.prompt_version,
            prompt_bundle=prompt_bundle,
            media=media,
            mode=self.config.pipeline.process_mode,
        )
        self.raw_store.save_request(manifest.document_id, None, request.model_dump(mode="json"))
        invocation = self.provider.invoke(request, dry_run=dry_run)
        raw_record = RawExtractionRecord(
            document_id=manifest.document_id,
            scope_type=ScopeType.DOCUMENT,
            page_number=None,
            provider_name=self.provider.provider_name,
            model_name=request.model_name,
            request_prompt_version=request.prompt_version,
            request_metadata=request.request_metadata,
            raw_response_text=invocation.raw_response_text,
            raw_response_json=invocation.raw_response_json,
            token_usage=invocation.token_usage,
            latency_ms=invocation.latency_ms,
            status=invocation.status,
            error_details=invocation.error_details,
            error_category=invocation.error_category,
            http_status_code=invocation.http_status_code,
            retry_after_seconds=invocation.retry_after_seconds,
            attempt_count=invocation.attempt_count,
            retry_count=invocation.retry_count,
            waited_seconds_total=invocation.waited_seconds_total,
            response_headers=invocation.response_headers,
            provider_error_payload=invocation.provider_error_payload,
        )
        self.raw_store.save_raw_response(raw_record)
        metrics.add_latency(invocation.latency_ms)
        metrics.add_retries(invocation.retry_count)

        if invocation.status == "failed":
            raise DocumentProcessingError(
                final_status=self._status_from_invocation(invocation.error_category),
                failed_scope=ScopeType.DOCUMENT.value,
                page_number=None,
                error_summary=invocation.error_details or "Provider invocation failed",
                error_category=invocation.error_category,
                http_status_code=invocation.http_status_code,
                attempt_count=invocation.attempt_count,
                retry_after_seconds=invocation.retry_after_seconds,
            )

        try:
            parsed_payload = self.provider.parse_response(
                document_id=manifest.document_id,
                page_number=None,
                raw_response_text=invocation.raw_response_text,
                raw_response_json=invocation.raw_response_json,
            )
        except Exception as exc:  # noqa: BLE001
            raise DocumentProcessingError(
                final_status=PipelineStatus.FAILED,
                failed_scope=ScopeType.DOCUMENT.value,
                page_number=None,
                error_summary=str(exc),
                error_category="parse_error",
            ) from exc
        pages_payload = parsed_payload.get("pages")
        normalized_records: list[NormalizedExtraction] = []
        if isinstance(pages_payload, list) and pages_payload:
            for index, page_payload in enumerate(pages_payload, start=1):
                normalized = normalize_provider_payload(manifest.document_id, index, page_payload)
                self.raw_store.save_parsed(normalized)
                normalized_records.append(normalized)
        else:
            normalized = normalize_provider_payload(manifest.document_id, None, parsed_payload)
            self.raw_store.save_parsed(normalized)
            normalized_records.append(normalized)

        return [raw_record], normalized_records

    def _classify_failure(self, exc: Exception) -> PipelineStatus:
        if isinstance(exc, NasStorageError) and exc.category in NAS_ERROR_CATEGORIES:
            return PipelineStatus.NEEDS_RETRY
        if isinstance(exc, LockConflictError):
            return PipelineStatus.NEEDS_RETRY
        message = str(exc).lower()
        if "429" in message or "too many requests" in message or "connect" in message or "timeout" in message:
            return PipelineStatus.NEEDS_RETRY
        return PipelineStatus.FAILED

    def _status_from_invocation(self, error_category: str | None) -> PipelineStatus:
        if error_category in {"rate_limited", "network", "timeout", "server_error"}:
            return PipelineStatus.NEEDS_RETRY
        return PipelineStatus.FAILED
