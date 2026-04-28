from __future__ import annotations

import os
import shutil
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from ncai_pipeline.storage.repository import FileRepository


class LockConflictError(RuntimeError):
    def __init__(self, lock_path: Path, metadata: dict[str, Any] | None = None) -> None:
        self.lock_path = lock_path
        self.metadata = metadata or {}
        super().__init__(f"NAS lock conflict at {lock_path}")


@dataclass
class FilesystemLock:
    locks_dir: Path
    lock_name: str
    lock_type: str
    repository: FileRepository
    stale_after_minutes: int = 240
    run_id: str | None = None
    document_id: str | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        self.token = uuid.uuid4().hex
        self.lock_path = self.locks_dir / self.lock_name
        self.metadata_path = self.lock_path / "lock.json"

    def __enter__(self) -> "FilesystemLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def acquire(self) -> None:
        if not self.enabled:
            return
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.lock_path.mkdir()
        except FileExistsError as exc:
            metadata = self._read_existing_metadata()
            if not self._is_stale(metadata):
                raise LockConflictError(self.lock_path, metadata) from exc
            self._replace_stale_lock()
        self._write_metadata()

    def heartbeat(self) -> None:
        if not self.enabled or not self.metadata_path.exists():
            return
        metadata = self._read_existing_metadata()
        if metadata.get("token") != self.token:
            raise LockConflictError(self.lock_path, metadata)
        metadata["updated_at"] = _utc_now_iso()
        self.repository.write_json(self.metadata_path, metadata)

    def release(self) -> None:
        if not self.enabled or not self.lock_path.exists():
            return
        metadata = self._read_existing_metadata()
        if metadata.get("token") != self.token:
            return
        try:
            shutil.rmtree(self.lock_path)
        except OSError:
            return

    def _write_metadata(self) -> None:
        metadata = {
            "lock_type": self.lock_type,
            "lock_name": self.lock_name,
            "run_id": self.run_id,
            "document_id": self.document_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "token": self.token,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "stale_after_minutes": self.stale_after_minutes,
        }
        self.repository.write_json(self.metadata_path, metadata)

    def _read_existing_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        try:
            payload = self.repository.read_json(self.metadata_path)
            return payload if isinstance(payload, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def _is_stale(self, metadata: dict[str, Any]) -> bool:
        updated_at = metadata.get("updated_at") or metadata.get("created_at")
        if not isinstance(updated_at, str):
            return True
        try:
            timestamp = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except ValueError:
            return True
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - timestamp).total_seconds()
        return age_seconds > self.stale_after_minutes * 60

    def _replace_stale_lock(self) -> None:
        stale_path = self.locks_dir / f"{self.lock_name}.stale-{uuid.uuid4().hex}"
        try:
            self.lock_path.rename(stale_path)
            self.lock_path.mkdir()
        except OSError as exc:
            metadata = self._read_existing_metadata()
            raise LockConflictError(self.lock_path, metadata) from exc
        finally:
            if stale_path.exists():
                try:
                    shutil.rmtree(stale_path)
                except OSError:
                    pass


def document_lock(
    locks_dir: Path,
    document_id: str,
    repository: FileRepository,
    *,
    run_id: str | None = None,
    stale_after_minutes: int = 240,
    enabled: bool = True,
) -> FilesystemLock:
    return FilesystemLock(
        locks_dir=locks_dir,
        lock_name=f"document_{document_id}.lock",
        lock_type="document",
        repository=repository,
        stale_after_minutes=stale_after_minutes,
        run_id=run_id,
        document_id=document_id,
        enabled=enabled,
    )


def export_lock(
    locks_dir: Path,
    repository: FileRepository,
    *,
    run_id: str | None = None,
    stale_after_minutes: int = 240,
    enabled: bool = True,
) -> FilesystemLock:
    return FilesystemLock(
        locks_dir=locks_dir,
        lock_name="export.lock",
        lock_type="export",
        repository=repository,
        stale_after_minutes=stale_after_minutes,
        run_id=run_id,
        enabled=enabled,
    )


def status_lock(
    locks_dir: Path,
    repository: FileRepository,
    *,
    run_id: str | None = None,
    stale_after_minutes: int = 240,
    enabled: bool = True,
) -> FilesystemLock:
    return FilesystemLock(
        locks_dir=locks_dir,
        lock_name="status.lock",
        lock_type="status",
        repository=repository,
        stale_after_minutes=stale_after_minutes,
        run_id=run_id,
        enabled=enabled,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
