from __future__ import annotations

from pathlib import Path

from ncai_pipeline.config import PipelineSettings
from ncai_pipeline.ingestion.file_types import detect_file_type
from ncai_pipeline.schemas.common import PipelineStatus
from ncai_pipeline.schemas.document import DocumentManifest
from ncai_pipeline.utils.hashing import sha256_file, stable_document_id
from ncai_pipeline.utils.time import utc_now


def discover_documents(input_dir: Path, settings: PipelineSettings) -> list[DocumentManifest]:
    manifests: list[DocumentManifest] = []
    seen_hashes: dict[str, str] = {}
    for path in sorted(p for p in input_dir.rglob("*") if p.is_file()):
        if path.suffix.lower() not in {ext.lower() for ext in settings.allowed_file_types}:
            continue
        file_type = detect_file_type(path)
        if file_type is None:
            continue
        sha256 = sha256_file(path)
        document_id = stable_document_id(path.name, sha256)
        now = utc_now()
        duplicate_of = seen_hashes.get(sha256)
        seen_hashes.setdefault(sha256, document_id)
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
            )
        )
    return manifests

