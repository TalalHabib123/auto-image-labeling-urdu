from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from ncai_pipeline.config import NasInputReadinessSettings, NasSettings
from ncai_pipeline.storage.paths import DataPaths
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
}


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
    if settings.healthcheck.require_writable_output:
        _assert_writable(output_root, repository, "output")
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


def _assert_writable(root: Path, repository: FileRepository, label: str) -> None:
    probe = root / ".nas_pipeline_healthcheck" / f"write_probe_{os.getpid()}.txt"
    try:
        repository.write_text(probe, _utc_now_iso())
        _ = repository.read_text(probe)
        probe.unlink(missing_ok=True)
        if probe.parent.exists():
            try:
                probe.parent.rmdir()
            except OSError:
                pass
    except OSError as exc:
        raise NasStorageError(f"NAS {label} root is not writable: {root}: {exc}", category="nas_unavailable") from exc


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
