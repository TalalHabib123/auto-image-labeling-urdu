from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from ncai_pipeline.config import AppConfig, EnvironmentSettings, PromptRegistry
from ncai_pipeline.ingestion.scanner import discover_documents
from ncai_pipeline.observability.metrics import PipelineMetrics
from ncai_pipeline.observability.tracing import create_run_context
from ncai_pipeline.parsing.normalizer import normalize_provider_payload
from ncai_pipeline.pipeline.stages import normalize_document
from ncai_pipeline.pipeline.state_manager import StateManager
from ncai_pipeline.providers.base import ExtractionProvider
from ncai_pipeline.review.artifact_builder import ReviewArtifactBuilder
from ncai_pipeline.schemas.common import PipelineStatus, ScopeType
from ncai_pipeline.schemas.document import DocumentManifest, PageArtifact
from ncai_pipeline.schemas.extraction import NormalizedExtraction, RawExtractionRecord
from ncai_pipeline.storage.manifests import ManifestStore
from ncai_pipeline.storage.paths import DataPaths
from ncai_pipeline.storage.raw_store import RawArtifactStore
from ncai_pipeline.storage.repository import FileRepository


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
        self.data_paths = data_paths
        self.provider = provider
        self.repository = repository or FileRepository()
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
        self.state_manager = StateManager()

    def run(
        self,
        input_dir: str | None = None,
        force: bool = False,
        dry_run: bool | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, object]:
        self.data_paths.ensure()
        run_context = create_run_context()
        metrics = PipelineMetrics()
        documents_needing_retry: list[dict[str, Any]] = []
        documents_failed: list[dict[str, Any]] = []
        effective_dry_run = self.config.pipeline.dry_run if dry_run is None else dry_run
        source_dir = Path(input_dir or self.config.pipeline.input_dir)
        prompt = self.prompts.get_prompt(self.config.pipeline.prompt_version)
        prompt_bundle = prompt.model_dump(mode="json")

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
                self._process_document(manifest, prompt_bundle, effective_dry_run, metrics)
                completed_documents += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "completed",
                            "document_id": manifest.document_id,
                            "source_path": manifest.source_path,
                            "completed_documents": completed_documents,
                            "total_documents": total_documents,
                            "status": PipelineStatus.REVIEW_PENDING.value,
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
        summary["documents_needing_retry"] = documents_needing_retry
        summary["documents_failed"] = documents_failed
        self.repository.write_json(self.data_paths.status_dir / run_context.run_id / "run_manifest.json", summary)
        self.repository.write_json(self.data_paths.status_dir / "latest_run.json", summary)
        return summary

    def _process_document(
        self,
        manifest: DocumentManifest,
        prompt_bundle: dict[str, str],
        dry_run: bool,
        metrics: PipelineMetrics,
    ) -> None:
        manifest = self.state_manager.update_status(manifest, PipelineStatus.DISCOVERED)
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
        else:
            raw_records, normalized_records = self._process_page_scope(
                manifest,
                page_artifacts,
                prompt_bundle,
                dry_run,
                metrics,
            )

        manifest = self.state_manager.update_status(manifest, PipelineStatus.EXTRACTED)
        self.manifests.save(manifest)
        review_record = self.review_builder.build(
            document_id=manifest.document_id,
            page_artifacts=page_artifacts,
            raw_records=raw_records,
            extractions=normalized_records,
            pipeline_version=manifest.pipeline_version,
            raw_output_path=self.data_paths.raw_responses_dir / manifest.document_id,
            is_dry_run=dry_run,
        )
        self.repository.write_json(self.data_paths.status_dir / manifest.document_id / "review_record.json", review_record)
        manifest = self.state_manager.update_status(manifest, PipelineStatus.REVIEW_PENDING)
        self.manifests.save(manifest)
        metrics.increment("processed_documents")
        metrics.increment("succeeded_documents")
        metrics.increment("pending_review_documents")

    def _process_page_scope(
        self,
        manifest: DocumentManifest,
        page_artifacts: list[PageArtifact],
        prompt_bundle: dict[str, str],
        dry_run: bool,
        metrics: PipelineMetrics,
    ) -> tuple[list[RawExtractionRecord], list[NormalizedExtraction]]:
        raw_records: list[RawExtractionRecord] = []
        normalized_records: list[NormalizedExtraction] = []
        for page in page_artifacts:
            page_path = Path(page.image_path)
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
            self.raw_store.save_raw_response(raw_record)
            raw_records.append(raw_record)
            metrics.add_latency(invocation.latency_ms)
            metrics.add_retries(invocation.retry_count)

            if invocation.status == "failed":
                raise DocumentProcessingError(
                    final_status=self._status_from_invocation(invocation.error_category),
                    failed_scope=ScopeType.PAGE.value,
                    page_number=page.page_number,
                    error_summary=invocation.error_details or "Provider invocation failed",
                    error_category=invocation.error_category,
                    http_status_code=invocation.http_status_code,
                    attempt_count=invocation.attempt_count,
                    retry_after_seconds=invocation.retry_after_seconds,
                )

            try:
                parsed_payload = self.provider.parse_response(
                    document_id=manifest.document_id,
                    page_number=page.page_number,
                    raw_response_text=invocation.raw_response_text,
                    raw_response_json=invocation.raw_response_json,
                )
            except Exception as exc:  # noqa: BLE001
                raise DocumentProcessingError(
                    final_status=PipelineStatus.FAILED,
                    failed_scope=ScopeType.PAGE.value,
                    page_number=page.page_number,
                    error_summary=str(exc),
                    error_category="parse_error",
                ) from exc
            normalized = normalize_provider_payload(manifest.document_id, page.page_number, parsed_payload)
            self.raw_store.save_parsed(normalized)
            normalized_records.append(normalized)
        return raw_records, normalized_records

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
        message = str(exc).lower()
        if "429" in message or "too many requests" in message or "connect" in message or "timeout" in message:
            return PipelineStatus.NEEDS_RETRY
        return PipelineStatus.FAILED

    def _status_from_invocation(self, error_category: str | None) -> PipelineStatus:
        if error_category in {"rate_limited", "network", "timeout", "server_error"}:
            return PipelineStatus.NEEDS_RETRY
        return PipelineStatus.FAILED
