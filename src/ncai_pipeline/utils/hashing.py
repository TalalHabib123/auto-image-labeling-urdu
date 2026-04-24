from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_document_id(file_name: str, sha256: str) -> str:
    stem = Path(file_name).stem.lower().replace(" ", "_")
    safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch == "_").strip("_") or "document"
    return f"{safe_stem}_{sha256[:16]}"


def stable_split_score(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)

