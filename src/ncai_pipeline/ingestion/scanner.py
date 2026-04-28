from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ncai_pipeline.config import PipelineSettings
from ncai_pipeline.ingestion.file_types import detect_file_type
from ncai_pipeline.schemas.common import PipelineStatus
from ncai_pipeline.schemas.document import DocumentManifest
from ncai_pipeline.storage.nas import ReadinessResult
from ncai_pipeline.utils.hashing import sha256_file, stable_document_id
from ncai_pipeline.utils.time import utc_now


LOGGER = logging.getLogger(__name__)


def discover_documents(
    input_dir: Path,
    settings: PipelineSettings,
    *,
    source_root: Path | None = None,
    ignored_path: Callable[[Path], bool] | None = None,
    readiness_checker: Callable[[Path], ReadinessResult] | None = None,
    storage_metadata: dict[str, object] | None = None,
) -> list[DocumentManifest]:
    manifests: list[DocumentManifest] = []
    seen_hashes: dict[str, str] = {}
    for path in sorted(p for p in input_dir.rglob("*") if p.is_file()):
        if ignored_path is not None and ignored_path(path):
            continue
        if path.suffix.lower() not in {ext.lower() for ext in settings.allowed_file_types}:
            continue
        file_type = detect_file_type(path)
        if file_type is None:
            continue
        if readiness_checker is not None:
            readiness = readiness_checker(path)
            if not readiness.ready:
                LOGGER.info(
                    "Skipping input that is not ready: %s | reason=%s | detail=%s",
                    path,
                    readiness.reason,
                    readiness.detail,
                )
                continue
        sha256 = sha256_file(path)
        stat = path.stat()
        document_id = stable_document_id(path.name, sha256)
        now = utc_now()
        duplicate_of = seen_hashes.get(sha256)
        seen_hashes.setdefault(sha256, document_id)
        root = source_root or input_dir
        try:
            source_relative_path = path.relative_to(root).as_posix()
        except ValueError:
            source_relative_path = path.name
        metadata = storage_metadata or {}
        manifests.append(
            DocumentManifest(
                document_id=document_id,
                source_path=str(path),
                file_name=path.name,
                file_type=file_type,
                sha256=sha256,
                status=PipelineStatus.DISCOVERED,
                created_at=now,
                updated_at=now,
                pipeline_version=settings.version,
                duplicate_of=duplicate_of,
                storage_mode=str(metadata.get("storage_mode")) if metadata.get("storage_mode") else None,
                source_root=str(root),
                source_relative_path=source_relative_path,
                source_size_bytes=stat.st_size,
                source_mtime_utc=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                source_sha256=sha256,
                nas_share_label=str(metadata.get("nas_share_label")) if metadata.get("nas_share_label") else None,
            )
        )
    return manifests
