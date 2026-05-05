from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from ncai_pipeline.config import NasInputReadinessSettings, NasSettings
from ncai_pipeline.storage.paths import DataPaths, NasDocumentVersionPaths, NasExportVersionPaths
from ncai_pipeline.storage.repository import FileRepository
from ncai_pipeline.utils.hashing import sha256_file


LOGGER = logging.getLogger(__name__)


NAS_ERROR_CATEGORIES = {
    "nas_unavailable",
    "nas_timeout",
    "nas_read_failed",
    "nas_write_failed",
    "nas_publish_failed",
    "nas_lock_conflict",
    "input_not_ready",
    "input_unstable",
    "input_marker_missing",
    "review_conflict",
    "scratch_cleanup_failed",
    "nas_permission_insufficient",
}

CREATE_ONLY_PUBLISH_STRATEGY = "local_scratch_create_only"


class NasStorageError(RuntimeError):
    def __init__(self, message: str, *, category: str) -> None:
        self.category = category
        super().__init__(message)


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    reason: str | None = None
    detail: str | None = None


def healthcheck_nas_paths(settings: NasSettings, repository: FileRepository) -> None:
    if not settings.healthcheck.enabled:
        return
    assert settings.input_root is not None
    assert settings.output_root is not None
    assert settings.scratch_root is not None

    input_root = Path(settings.input_root)
    output_root = Path(settings.output_root)
    scratch_root = Path(settings.scratch_root)
    if not input_root.exists() or not input_root.is_dir():
        raise NasStorageError(f"NAS input root is unavailable: {input_root}", category="nas_unavailable")
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        scratch_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise NasStorageError(f"Could not prepare NAS output or scratch root: {exc}", category="nas_unavailable") from exc
    if not output_root.is_dir():
        raise NasStorageError(f"NAS output root is not a directory: {output_root}", category="nas_unavailable")
    if not scratch_root.is_dir():
        raise NasStorageError(f"NAS scratch root is not a directory: {scratch_root}", category="nas_unavailable")
    if settings.publish_strategy == CREATE_ONLY_PUBLISH_STRATEGY:
        if settings.healthcheck.require_writable_output:
            _assert_create_only_writable(output_root, repository, "output")
        _assert_writable(scratch_root, repository, "scratch")
        return
    if settings.healthcheck.require_writable_output:
        _assert_nas_capabilities(output_root, repository, "output")
        _assert_writable(scratch_root, repository, "scratch")


def retry_operation(settings: NasSettings, operation: Callable[[], Any]) -> Any:
    last_error: OSError | NasStorageError | None = None
    for attempt in range(1, settings.retry.attempts + 1):
        try:
            return operation()
        except NasStorageError:
            raise
        except OSError as exc:
            last_error = exc
            if attempt >= settings.retry.attempts:
                break
            sleep_seconds = min(
                settings.retry.max_backoff_seconds,
                settings.retry.backoff_seconds * (2 ** (attempt - 1)),
            )
            time.sleep(sleep_seconds)
    raise NasStorageError(f"NAS filesystem operation failed after retries: {last_error}", category="nas_read_failed")


def make_input_readiness_checker(settings: NasInputReadinessSettings) -> Callable[[Path], ReadinessResult]:
    def checker(path: Path) -> ReadinessResult:
        return check_input_readiness(path, settings)

    return checker


def check_input_readiness(path: Path, settings: NasInputReadinessSettings) -> ReadinessResult:
    if not settings.enabled:
        return ReadinessResult(True)
    marker_path = Path(f"{path}{settings.ready_marker_suffix}")
    if settings.require_ready_marker and not marker_path.exists():
        return ReadinessResult(False, "input_marker_missing", f"Ready marker not found: {marker_path}")
    try:
        first = path.stat()
    except OSError as exc:
        return ReadinessResult(False, "input_not_ready", str(exc))
    if settings.stable_check_seconds > 0:
        time.sleep(settings.stable_check_seconds)
    try:
        second = path.stat()
    except OSError as exc:
        return ReadinessResult(False, "input_not_ready", str(exc))
    if first.st_size != second.st_size or first.st_mtime_ns != second.st_mtime_ns:
        return ReadinessResult(False, "input_unstable", f"File changed during readiness check: {path}")
    return ReadinessResult(True)


def ignored_source_path(path: Path, *, input_root: Path, extra_roots: Iterable[Path] = ()) -> bool:
    ignored_names = {
        ".staging",
        ".nas_pipeline_healthcheck",
        "System Volume Information",
        "$RECYCLE.BIN",
        "working",
        "review",
        "curated",
        "exports",
    }
    try:
        relative_parts = path.relative_to(input_root).parts
    except ValueError:
        relative_parts = path.parts
    if any(part.startswith(".") for part in relative_parts if part not in {".", ""}):
        return True
    if any(part in ignored_names for part in relative_parts):
        return True
    for root in extra_roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def copy_source_to_scratch(
    source_path: Path,
    scratch_paths: DataPaths,
    document_id: str,
    repository: FileRepository,
    *,
    verify_checksum: bool,
    expected_sha256: str | None = None,
) -> tuple[Path, str]:
    target = scratch_paths.sources_dir / document_id / source_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    current_sha256 = sha256_file(source_path)
    if verify_checksum and expected_sha256 is not None and current_sha256 != expected_sha256:
        raise NasStorageError(
            f"Source hash changed after discovery for {source_path}",
            category="input_unstable",
        )
    repository.copy_file(source_path, target)
    copied_sha256 = sha256_file(target)
    if verify_checksum and current_sha256 != copied_sha256:
        raise NasStorageError(
            f"Source hash changed while copying {source_path}",
            category="input_unstable",
        )
    return target, copied_sha256


def publish_document_artifacts(
    *,
    run_id: str,
    document_id: str,
    scratch_paths: DataPaths,
    final_paths: DataPaths,
    repository: FileRepository,
    required_artifacts: list[str],
    completion_payload: dict[str, Any],
    verify_checksums: bool,
) -> None:
    staging_dir = final_paths.staging_dir / run_id / document_id
    if staging_dir.exists():
        repository.safe_remove_tree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        _copy_document_artifacts_to_staging(document_id, scratch_paths, staging_dir, repository)
        _validate_required_artifacts(staging_dir, required_artifacts)
        if verify_checksums:
            _verify_staged_checksums(scratch_paths.root, staging_dir)
        _finalize_document_staging(document_id, staging_dir, final_paths, repository)
        repository.write_json(final_paths.completion_markers_dir / f"{document_id}.complete.json", completion_payload)
    except Exception as exc:  # noqa: BLE001
        raise NasStorageError(f"Could not publish NAS document artifacts for {document_id}: {exc}", category="nas_publish_failed") from exc
    finally:
        if staging_dir.exists():
            try:
                repository.safe_remove_tree(staging_dir)
            except OSError:
                LOGGER.warning("Could not remove NAS staging directory %s", staging_dir)


def publish_export_artifacts(
    *,
    run_id: str,
    scratch_paths: DataPaths,
    final_paths: DataPaths,
    repository: FileRepository,
    verify_checksums: bool,
) -> None:
    staging_dir = final_paths.staging_dir / run_id / "exports"
    if staging_dir.exists():
        repository.safe_remove_tree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        _copy_if_exists(scratch_paths.exports_dir, staging_dir / "exports", repository)
        _copy_if_exists(scratch_paths.approved_dir, staging_dir / "curated" / "approved", repository)
        _copy_if_exists(scratch_paths.dry_run_dir, staging_dir / "curated" / "dry_run", repository)
        manifest = staging_dir / "exports" / "export_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Export manifest not found in staging: {manifest}")
        if verify_checksums:
            _verify_staged_checksums(scratch_paths.root, staging_dir)
        _move_directory_children(staging_dir / "exports", final_paths.exports_dir, repository, skip_names={"export_manifest.json"})
        _move_directory_children(staging_dir / "curated" / "approved", final_paths.approved_dir, repository)
        _move_directory_children(staging_dir / "curated" / "dry_run", final_paths.dry_run_dir, repository)
        _replace_file(manifest, final_paths.exports_dir / "export_manifest.json", repository)
    except Exception as exc:  # noqa: BLE001
        raise NasStorageError(f"Could not publish NAS export artifacts: {exc}", category="nas_publish_failed") from exc
    finally:
        if staging_dir.exists():
            try:
                repository.safe_remove_tree(staging_dir)
            except OSError:
                LOGGER.warning("Could not remove NAS export staging directory %s", staging_dir)


def publish_document_artifacts_create_only(
    *,
    run_id: str,
    document_id: str,
    scratch_paths: DataPaths,
    final_paths: DataPaths,
    repository: FileRepository,
    verify_checksums: bool,
) -> dict[str, Any]:
    version_paths = NasDocumentVersionPaths(final_paths.root, document_id, run_id)
    copy_plan = _document_create_only_copy_plan(document_id, scratch_paths, version_paths)
    required_paths = [
        version_paths.manifest_path,
        version_paths.review_record_path,
        version_paths.correction_path,
        version_paths.report_path,
    ]
    missing_sources = [str(source) for source, _ in copy_plan if not source.exists()]
    if missing_sources:
        raise NasStorageError(
            f"Could not publish NAS document artifacts for {document_id}: missing source artifact(s): "
            f"{', '.join(missing_sources)}",
            category="nas_publish_failed",
        )
    existing_targets = [str(target) for _, target in copy_plan if target.exists()]
    if existing_targets:
        raise NasStorageError(
            f"Create-only NAS publish refused to overwrite existing artifact(s): {', '.join(existing_targets)}",
            category="nas_publish_failed",
        )

    checksums: dict[str, str] = {}
    try:
        for source, target in copy_plan:
            repository.copy_file_create_only(source, target)
            relative_path = _relative_to_root(target, final_paths.root)
            if verify_checksums:
                expected = sha256_file(source)
                if sha256_file(target) != expected:
                    raise NasStorageError(f"Checksum mismatch after create-only publish: {relative_path}", category="nas_publish_failed")
                checksums[relative_path] = expected

        required_relative = [_relative_to_root(path, final_paths.root) for path in required_paths]
        completion_payload = document_completion_payload_create_only(
            document_id=document_id,
            run_id=run_id,
            final_paths=final_paths,
            version_paths=version_paths,
            required_artifacts=required_relative,
            checksums_verified=verify_checksums,
            checksums=checksums,
        )
        repository.create_json(version_paths.completion_marker_path, completion_payload)
        return completion_payload
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, NasStorageError):
            raise
        raise NasStorageError(f"Could not publish NAS document artifacts for {document_id}: {exc}", category="nas_publish_failed") from exc


def publish_export_artifacts_create_only(
    *,
    run_id: str,
    scratch_paths: DataPaths,
    final_paths: DataPaths,
    repository: FileRepository,
    verify_checksums: bool,
) -> dict[str, Any]:
    version_paths = NasExportVersionPaths(final_paths.root, run_id)
    copy_plan = _export_create_only_copy_plan(scratch_paths, version_paths)
    manifest_source = scratch_paths.exports_dir / "export_manifest.json"
    if not manifest_source.exists():
        raise NasStorageError(f"Export manifest not found in scratch: {manifest_source}", category="nas_publish_failed")
    existing_targets = [str(target) for _, target in copy_plan if target.exists()]
    if existing_targets:
        raise NasStorageError(
            f"Create-only NAS export refused to overwrite existing artifact(s): {', '.join(existing_targets)}",
            category="nas_publish_failed",
        )

    checksums: dict[str, str] = {}
    try:
        for source, target in copy_plan:
            repository.copy_file_create_only(source, target)
            relative_path = _relative_to_root(target, final_paths.root)
            if verify_checksums:
                expected = sha256_file(source)
                if sha256_file(target) != expected:
                    raise NasStorageError(f"Checksum mismatch after create-only export publish: {relative_path}", category="nas_publish_failed")
                checksums[relative_path] = expected

        completion_payload = {
            "run_id": run_id,
            "status": "completed",
            "published_at": _utc_now_iso(),
            "publish_strategy": CREATE_ONLY_PUBLISH_STRATEGY,
            "export_manifest_path": str(version_paths.manifest_path),
            "required_artifacts": [_relative_to_root(version_paths.manifest_path, final_paths.root)],
            "checksums_verified": verify_checksums,
            "checksums": checksums,
        }
        repository.create_json(version_paths.completion_marker_path, completion_payload)
        return completion_payload
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, NasStorageError):
            raise
        raise NasStorageError(f"Could not publish NAS export artifacts: {exc}", category="nas_publish_failed") from exc


def document_completion_payload(
    *,
    document_id: str,
    run_id: str,
    final_paths: DataPaths,
    required_artifacts: list[str],
    checksums_verified: bool,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "run_id": run_id,
        "status": "completed",
        "published_at": _utc_now_iso(),
        "artifact_version": 1,
        "required_artifacts": required_artifacts,
        "checksums_verified": checksums_verified,
        "completion_marker_path": str(final_paths.completion_markers_dir / f"{document_id}.complete.json"),
    }


def document_completion_payload_create_only(
    *,
    document_id: str,
    run_id: str,
    final_paths: DataPaths,
    version_paths: NasDocumentVersionPaths,
    required_artifacts: list[str],
    checksums_verified: bool,
    checksums: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "run_id": run_id,
        "status": "completed",
        "published_at": _utc_now_iso(),
        "artifact_version": 1,
        "publish_strategy": CREATE_ONLY_PUBLISH_STRATEGY,
        "manifest_path": str(version_paths.manifest_path),
        "review_record_path": str(version_paths.review_record_path),
        "correction_path": str(version_paths.correction_path),
        "report_path": str(version_paths.report_path),
        "required_artifacts": required_artifacts,
        "checksums_verified": checksums_verified,
        "checksums": checksums or {},
        "completion_marker_path": str(version_paths.completion_marker_path),
        "completion_marker_relative_path": _relative_to_root(version_paths.completion_marker_path, final_paths.root),
    }


def nas_document_is_complete(
    final_paths: DataPaths,
    document_id: str,
    repository: FileRepository,
    *,
    required_artifacts: list[str] | None = None,
) -> bool:
    manifest_path = final_paths.manifests_dir / document_id / "manifest.json"
    marker_path = final_paths.completion_markers_dir / f"{document_id}.complete.json"
    if not manifest_path.exists() or not marker_path.exists():
        return False
    try:
        marker = repository.read_json(marker_path)
    except Exception:  # noqa: BLE001
        return False
    if marker.get("status") != "completed":
        return False
    required = required_artifacts or marker.get("required_artifacts") or []
    return all((final_paths.root / relative_path).exists() for relative_path in required)


def nas_document_is_complete_create_only(
    final_paths: DataPaths,
    document_id: str,
    repository: FileRepository,
    *,
    source_sha256: str | None = None,
) -> bool:
    marker = NasOutputIndex(final_paths, repository).latest_document_marker(document_id)
    if marker is None:
        return False
    if source_sha256 and marker.manifest.get("source_sha256") not in {None, source_sha256}:
        return False
    return True


@dataclass(frozen=True)
class NasDocumentMarker:
    document_id: str
    run_id: str
    marker_path: Path
    payload: dict[str, Any]
    manifest_path: Path
    manifest: dict[str, Any]
    review_record_path: Path
    correction_path: Path | None
    report_path: Path | None


class NasOutputIndex:
    def __init__(self, data_paths: DataPaths, repository: FileRepository) -> None:
        self.data_paths = data_paths
        self.repository = repository

    def latest_run_summary(self) -> dict[str, Any] | None:
        stable_path = self.data_paths.status_dir / "latest_run.json"
        if stable_path.exists():
            return self.repository.read_json(stable_path)
        run_paths = sorted(
            self.data_paths.run_manifests_dir.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for path in run_paths:
            try:
                payload = self.repository.read_json(path)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(payload, dict):
                return payload
        legacy_paths = sorted(
            self.data_paths.status_dir.glob("*/run_manifest.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for path in legacy_paths:
            try:
                payload = self.repository.read_json(path)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def latest_export_summary(self) -> dict[str, Any] | None:
        stable_path = self.data_paths.exports_dir / "export_manifest.json"
        if stable_path.exists():
            return self.repository.read_json(stable_path)
        legacy_path = self.data_paths.export_target_dir / "export_manifest.json"
        if legacy_path.exists():
            return self.repository.read_json(legacy_path)
        marker = self.latest_export_marker()
        if marker is None:
            return None
        manifest_path = Path(str(marker.get("export_manifest_path") or ""))
        if not manifest_path.exists():
            return None
        payload = self.repository.read_json(manifest_path)
        return payload if isinstance(payload, dict) else None

    def latest_export_marker(self) -> dict[str, Any] | None:
        marker_paths = sorted(
            (self.data_paths.exports_dir / "completed").glob("*.complete.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for marker_path in marker_paths:
            try:
                marker = self.repository.read_json(marker_path)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(marker, dict) or marker.get("status") != "completed":
                continue
            manifest_path = Path(str(marker.get("export_manifest_path") or ""))
            if manifest_path.exists():
                return marker
        return None

    def latest_document_marker(self, document_id: str) -> NasDocumentMarker | None:
        markers = self.document_markers(document_id)
        return markers[0] if markers else None

    def document_markers(self, document_id: str) -> list[NasDocumentMarker]:
        marker_dir = self.data_paths.completion_markers_dir / document_id
        marker_paths = list(marker_dir.glob("*.complete.json")) if marker_dir.exists() else []
        legacy_marker = self.data_paths.completion_markers_dir / f"{document_id}.complete.json"
        if legacy_marker.exists():
            marker_paths.append(legacy_marker)
        markers: list[NasDocumentMarker] = []
        for marker_path in marker_paths:
            marker = self._load_document_marker(marker_path, document_id)
            if marker is not None:
                markers.append(marker)
        return sorted(markers, key=lambda item: _marker_sort_key(item.payload, item.marker_path), reverse=True)

    def latest_document_markers(self) -> list[NasDocumentMarker]:
        by_document: dict[str, NasDocumentMarker] = {}
        candidate_paths = list(self.data_paths.completion_markers_dir.glob("*/*.complete.json"))
        candidate_paths.extend(self.data_paths.completion_markers_dir.glob("*.complete.json"))
        for marker_path in candidate_paths:
            marker = self._load_document_marker(marker_path, None)
            if marker is None:
                continue
            current = by_document.get(marker.document_id)
            if current is None or _marker_sort_key(marker.payload, marker.marker_path) > _marker_sort_key(current.payload, current.marker_path):
                by_document[marker.document_id] = marker
        return sorted(by_document.values(), key=lambda item: item.document_id)

    def latest_manifest(self, document_id: str) -> dict[str, Any] | None:
        marker = self.latest_document_marker(document_id)
        return marker.manifest if marker is not None else None

    def latest_review_record_path(self, document_id: str) -> Path | None:
        revision_marker = self.latest_review_revision_marker(document_id)
        if revision_marker is not None:
            path = Path(str(revision_marker.get("review_record_path") or ""))
            if path.exists():
                return path
        marker = self.latest_document_marker(document_id)
        return marker.review_record_path if marker is not None and marker.review_record_path.exists() else None

    def latest_review_record_paths(self) -> list[Path]:
        paths: list[Path] = []
        for marker in self.latest_document_markers():
            path = self.latest_review_record_path(marker.document_id)
            if path is not None:
                paths.append(path)
        return sorted(paths)

    def latest_review_revision_marker(self, document_id: str) -> dict[str, Any] | None:
        marker_dir = self.data_paths.status_dir / "review_revisions" / document_id
        marker_paths = sorted(
            marker_dir.glob("*.complete.json") if marker_dir.exists() else [],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for marker_path in marker_paths:
            try:
                marker = self.repository.read_json(marker_path)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(marker, dict) and marker.get("status") == "completed":
                return marker
        return None

    def latest_page_image_path(self, document_id: str, page_number: int) -> Path:
        marker = self.latest_document_marker(document_id)
        if marker is not None:
            run_path = self.data_paths.rendered_pages_dir / document_id / marker.run_id / f"page_{page_number:04d}.png"
            if run_path.exists():
                return run_path
        return self.data_paths.rendered_pages_dir / document_id / f"page_{page_number:04d}.png"

    def _load_document_marker(self, marker_path: Path, expected_document_id: str | None) -> NasDocumentMarker | None:
        try:
            payload = self.repository.read_json(marker_path)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict) or payload.get("status") != "completed":
            return None
        document_id = str(payload.get("document_id") or expected_document_id or "")
        run_id = str(payload.get("run_id") or marker_path.stem.replace(".complete", ""))
        if not document_id:
            return None
        if expected_document_id is not None and document_id != expected_document_id:
            return None
        manifest_path = Path(str(payload.get("manifest_path") or self.data_paths.manifests_dir / document_id / "manifest.json"))
        if not manifest_path.exists():
            return None
        try:
            manifest = self.repository.read_json(manifest_path)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(manifest, dict):
            return None
        required = payload.get("required_artifacts") or []
        if not _required_artifacts_exist(self.data_paths.root, required):
            return None
        review_record_path = Path(str(payload.get("review_record_path") or self.data_paths.status_dir / document_id / "review_record.json"))
        if not review_record_path.exists():
            return None
        correction_path = Path(str(payload["correction_path"])) if payload.get("correction_path") else None
        report_path = Path(str(payload["report_path"])) if payload.get("report_path") else None
        return NasDocumentMarker(
            document_id=document_id,
            run_id=run_id,
            marker_path=marker_path,
            payload=payload,
            manifest_path=manifest_path,
            manifest=manifest,
            review_record_path=review_record_path,
            correction_path=correction_path,
            report_path=report_path,
        )


def cleanup_successful_scratch(scratch_root: Path, run_id: str, settings: NasSettings, repository: FileRepository) -> None:
    active_run = scratch_root / "active" / run_id
    if not active_run.exists():
        return
    try:
        if settings.scratch.cleanup_on_success:
            repository.safe_remove_tree(active_run)
            return
        completed_run = scratch_root / "completed" / run_id
        if completed_run.exists():
            repository.safe_remove_tree(completed_run)
        completed_run.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(active_run), str(completed_run))
    except OSError as exc:
        raise NasStorageError(f"Could not clean NAS scratch run {run_id}: {exc}", category="scratch_cleanup_failed") from exc


def retain_failed_scratch(scratch_root: Path, run_id: str, repository: FileRepository) -> None:
    active_run = scratch_root / "active" / run_id
    if not active_run.exists():
        return
    failed_run = scratch_root / "failed" / run_id
    if failed_run.exists():
        repository.safe_remove_tree(failed_run)
    failed_run.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(active_run), str(failed_run))

def _assert_nas_capabilities(root: Path, repository: FileRepository, label: str) -> None:
    healthcheck_dir = root / ".nas_pipeline_healthcheck"
    pid = os.getpid()

    raw_probe = healthcheck_dir / f"raw_write_probe_{pid}.txt"
    rename_src = healthcheck_dir / f"rename_probe_{pid}.tmp"
    rename_dst = healthcheck_dir / f"rename_probe_{pid}.txt"
    delete_probe = healthcheck_dir / f"delete_probe_{pid}.txt"

    try:
        healthcheck_dir.mkdir(parents=True, exist_ok=True)

        # 1. Basic write/read
        raw_probe.write_text(_utc_now_iso(), encoding="utf-8")
        _ = raw_probe.read_text(encoding="utf-8")

        # 2. Rename/replace capability
        rename_src.write_text(_utc_now_iso(), encoding="utf-8")
        try:
            os.replace(rename_src, rename_dst)
        except PermissionError as exc:
            raise NasStorageError(
                f"NAS {label} root allows basic write but denies atomic rename/replace. "
                f"Your pipeline uses atomic writes, so this NAS user needs Modify/Delete/Rename permission. "
                f"root={root}",
                category="nas_permission_insufficient",
            ) from exc

        # 3. Delete capability
        delete_probe.write_text(_utc_now_iso(), encoding="utf-8")
        try:
            delete_probe.unlink()
        except PermissionError as exc:
            raise NasStorageError(
                f"NAS {label} root allows write but denies delete. "
                f"Pipeline cleanup, lock cleanup, and atomic writes may fail. "
                f"root={root}",
                category="nas_permission_insufficient",
            ) from exc

    except OSError as exc:
        raise NasStorageError(
            f"NAS {label} root failed storage capability check: {root}: {exc}",
            category="nas_unavailable",
        ) from exc

    finally:
        # Best-effort cleanup only. Never rely on this succeeding with restricted NAS permissions.
        for p in [raw_probe, rename_src, rename_dst, delete_probe]:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

        try:
            healthcheck_dir.rmdir()
        except OSError:
            pass


def _assert_create_only_writable(root: Path, repository: FileRepository, label: str) -> None:
    healthcheck_dir = root / ".nas_pipeline_healthcheck" / f"create_only_{os.getpid()}_{uuid.uuid4().hex}"
    probe = healthcheck_dir / "write_probe.json"
    try:
        healthcheck_dir.mkdir(parents=True, exist_ok=False)
        repository.create_json(
            probe,
            {
                "created_at": _utc_now_iso(),
                "purpose": "create-only NAS output healthcheck",
                "label": label,
            },
        )
        payload = repository.read_json(probe)
        if not isinstance(payload, dict) or payload.get("purpose") != "create-only NAS output healthcheck":
            raise NasStorageError(f"NAS {label} root create-only healthcheck readback mismatch: {root}", category="nas_write_failed")
    except NasStorageError:
        raise
    except OSError as exc:
        raise NasStorageError(
            f"NAS {label} root does not allow create/read output required for create-only mode: {root}: {exc}",
            category="nas_unavailable",
        ) from exc
    finally:
        # Best effort only: limited NAS accounts may not be allowed to delete healthcheck files.
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            healthcheck_dir.rmdir()
        except OSError:
            pass


def _assert_writable(root: Path, repository: FileRepository, label: str) -> None:
    probe = root / ".nas_pipeline_healthcheck" / f"write_probe_{os.getpid()}.txt"
    try:
        repository.write_text(probe, _utc_now_iso(), atomic=True)
        _ = repository.read_text(probe)
        probe.unlink(missing_ok=True)
        if probe.parent.exists():
            try:
                probe.parent.rmdir()
            except OSError:
                pass
    except OSError as exc:
        raise NasStorageError(f"NAS {label} root is not writable: {root}: {exc}", category="nas_unavailable") from exc


def _document_create_only_copy_plan(
    document_id: str,
    scratch_paths: DataPaths,
    version_paths: NasDocumentVersionPaths,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    pairs.extend(_tree_file_pairs(scratch_paths.rendered_pages_dir / document_id, version_paths.rendered_pages_dir))
    pairs.extend(_tree_file_pairs(scratch_paths.raw_requests_dir / document_id, version_paths.raw_requests_dir))
    pairs.extend(_tree_file_pairs(scratch_paths.raw_responses_dir / document_id, version_paths.raw_responses_dir))
    pairs.extend(_tree_file_pairs(scratch_paths.parsed_dir / document_id, version_paths.parsed_dir))
    pairs.append((scratch_paths.status_dir / document_id / "review_record.json", version_paths.review_record_path))
    pairs.append((scratch_paths.correction_files_dir / document_id / "corrected.yaml", version_paths.correction_path))
    pairs.append((scratch_paths.reports_dir / document_id / "report.html", version_paths.report_path))
    pairs.append((scratch_paths.manifests_dir / document_id / "manifest.json", version_paths.manifest_path))
    return pairs


def _export_create_only_copy_plan(
    scratch_paths: DataPaths,
    version_paths: NasExportVersionPaths,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    pairs.extend(_tree_file_pairs(scratch_paths.exports_dir, version_paths.export_root))
    return pairs


def _tree_file_pairs(source_dir: Path, target_dir: Path) -> list[tuple[Path, Path]]:
    if not source_dir.exists():
        return []
    if source_dir.is_file():
        return [(source_dir, target_dir)]
    return [
        (source, target_dir / source.relative_to(source_dir))
        for source in sorted(source_dir.rglob("*"))
        if source.is_file()
    ]


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _required_artifacts_exist(root: Path, required_artifacts: Any) -> bool:
    if not isinstance(required_artifacts, list):
        return False
    for item in required_artifacts:
        if not isinstance(item, str) or not item:
            return False
        candidate = Path(item)
        path = candidate if candidate.is_absolute() else root / candidate
        if not path.exists():
            return False
    return True


def _marker_sort_key(payload: dict[str, Any], marker_path: Path) -> tuple[str, float]:
    published_at = payload.get("published_at")
    timestamp = published_at if isinstance(published_at, str) else ""
    return (timestamp, marker_path.stat().st_mtime)


def _copy_document_artifacts_to_staging(document_id: str, scratch_paths: DataPaths, staging_dir: Path, repository: FileRepository) -> None:
    for relative_path in _document_artifact_paths(document_id):
        source = scratch_paths.root / relative_path
        if source.exists():
            target = staging_dir / relative_path
            if source.is_dir():
                repository.copy_tree(source, target, replace=True)
            else:
                repository.copy_file(source, target)


def _document_artifact_paths(document_id: str) -> list[Path]:
    return [
        Path("working") / "rendered_pages" / document_id,
        Path("working") / "raw_requests" / document_id,
        Path("working") / "raw_responses" / document_id,
        Path("working") / "parsed" / document_id,
        Path("working") / "status" / document_id,
        Path("review") / "reports" / document_id,
        Path("review") / "correction_files" / document_id,
        Path("working") / "manifests" / document_id / "manifest.json",
    ]


def required_document_artifacts(document_id: str) -> list[str]:
    return [
        str((Path("working") / "manifests" / document_id / "manifest.json").as_posix()),
        str((Path("review") / "correction_files" / document_id / "corrected.yaml").as_posix()),
        str((Path("working") / "status" / document_id / "review_record.json").as_posix()),
    ]


def _validate_required_artifacts(root: Path, required_artifacts: list[str]) -> None:
    missing = [relative_path for relative_path in required_artifacts if not (root / relative_path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required staged artifact(s): {', '.join(missing)}")


def _verify_staged_checksums(source_root: Path, staging_root: Path) -> None:
    for staged_file in staging_root.rglob("*"):
        if not staged_file.is_file():
            continue
        relative_path = staged_file.relative_to(staging_root)
        source_file = source_root / relative_path
        if source_file.exists() and sha256_file(source_file) != sha256_file(staged_file):
            raise NasStorageError(f"Checksum mismatch after staging: {relative_path}", category="nas_publish_failed")


def _finalize_document_staging(document_id: str, staging_dir: Path, final_paths: DataPaths, repository: FileRepository) -> None:
    manifest_relative = Path("working") / "manifests" / document_id / "manifest.json"
    manifest_source = staging_dir / manifest_relative
    for relative_path in _document_artifact_paths(document_id):
        if relative_path == manifest_relative:
            continue
        source = staging_dir / relative_path
        if not source.exists():
            continue
        target = final_paths.root / relative_path
        if source.is_dir():
            repository.atomic_publish_tree(source, target, replace=True)
        else:
            _replace_file(source, target, repository)
    _replace_file(manifest_source, final_paths.root / manifest_relative, repository)


def _copy_if_exists(source: Path, target: Path, repository: FileRepository) -> None:
    if not source.exists():
        return
    if source.is_dir():
        repository.copy_tree(source, target, replace=True)
    else:
        repository.copy_file(source, target)


def _move_directory_children(source: Path, target: Path, repository: FileRepository, *, skip_names: set[str] | None = None) -> None:
    if not source.exists():
        return
    skip_names = skip_names or set()
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in skip_names:
            continue
        destination = target / item.name
        if item.is_dir():
            repository.atomic_publish_tree(item, destination, replace=True)
        else:
            _replace_file(item, destination, repository)


def _replace_file(source: Path, target: Path, repository: FileRepository) -> None:
    repository.ensure_parent(target)
    if target.exists() and target.is_dir():
        repository.safe_remove_tree(target)
    os.replace(source, target)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
