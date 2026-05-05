from __future__ import annotations

from contextlib import suppress
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel

from ncai_pipeline.utils.hashing import sha256_file


class FileRepository:
    def ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def write_json(self, path: Path, data: BaseModel | dict[str, Any] | list[Any]) -> None:
        payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
        self.atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    def create_json(self, path: Path, data: BaseModel | dict[str, Any] | list[Any]) -> None:
        payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
        self.create_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def write_yaml(self, path: Path, data: BaseModel | dict[str, Any]) -> None:
        payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
        self.atomic_write_text(path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))

    def create_yaml(self, path: Path, data: BaseModel | dict[str, Any]) -> None:
        payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
        self.create_text(path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))

    def read_yaml(self, path: Path) -> Any:
        return yaml.safe_load(path.read_text(encoding="utf-8-sig"))

    def write_text(self, path: Path, data: str, *, atomic: bool = True) -> None:
        if atomic:
            self.atomic_write_text(path, data)
        else:
            self.non_atomic_write_text(path, data)

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8-sig")

    def exists(self, path: Path) -> bool:
        return path.exists()

    def glob(self, path: Path, pattern: str) -> Iterable[Path]:
        return path.glob(pattern)

    def read_bytes(self, path: Path) -> bytes:
        return path.read_bytes()

    def write_bytes(self, path: Path, data: bytes) -> None:
        self.atomic_write_bytes(path, data)

    def create_bytes(self, path: Path, data: bytes) -> None:
        self.ensure_parent(path)
        try:
            with path.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise FileExistsError(f"Create-only write refused to overwrite existing file: {path}") from exc

    def copy_file(self, source: Path, target: Path) -> None:
        self.ensure_parent(target)
        shutil.copy2(source, target)

    def copy_file_create_only(self, source: Path, target: Path) -> None:
        if not source.exists():
            raise FileNotFoundError(f"Copy source not found: {source}")
        self.ensure_parent(target)
        try:
            with source.open("rb") as source_handle, target.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
                target_handle.flush()
                os.fsync(target_handle.fileno())
        except FileExistsError as exc:
            raise FileExistsError(f"Create-only copy refused to overwrite existing file: {target}") from exc

    def copy_tree(self, source: Path, target: Path, *, replace: bool = False) -> None:
        if replace and target.exists():
            self.safe_remove_tree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=replace)

    def copy_tree_create_only(self, source: Path, target: Path) -> None:
        if not source.exists():
            raise FileNotFoundError(f"Copy source tree not found: {source}")
        if not source.is_dir():
            raise NotADirectoryError(f"Copy source tree is not a directory: {source}")
        if target.exists():
            raise FileExistsError(f"Create-only tree copy refused to reuse existing directory: {target}")
        target.mkdir(parents=True, exist_ok=False)
        for item in sorted(source.rglob("*")):
            relative_path = item.relative_to(source)
            destination = target / relative_path
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif item.is_file():
                self.copy_file_create_only(item, destination)

    def atomic_write_json(self, path: Path, data: BaseModel | dict[str, Any] | list[Any]) -> None:
        self.write_json(path, data)

    def atomic_write_yaml(self, path: Path, data: BaseModel | dict[str, Any]) -> None:
        self.write_yaml(path, data)

    def non_atomic_write_text(self, path: Path, data: str) -> None:
        self.ensure_parent(path)
        path.write_text(data, encoding="utf-8")

    def create_text(self, path: Path, data: str) -> None:
        self.ensure_parent(path)
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise FileExistsError(f"Create-only write refused to overwrite existing file: {path}") from exc

    def atomic_write_text(self, path: Path, data: str) -> None:
        self.ensure_parent(path)

        temp_path = self._temp_path(path)

        try:
            temp_path.write_text(data, encoding="utf-8")
            os.replace(temp_path, path)

        except PermissionError as exc:
            raise PermissionError(
                f"Atomic write failed. Temp file was created but rename/replace was denied. "
                f"This usually means the NAS account has write permission but lacks modify/delete/rename permission. "
                f"temp_path={temp_path}, final_path={path}"
            ) from exc

        finally:
            # Do not let cleanup hide the real os.replace() error.
            with suppress(FileNotFoundError):
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except PermissionError:
                        # Leave temp file behind rather than masking the real failure.
                        pass

    def atomic_write_bytes(self, path: Path, data: bytes) -> None:
        self.ensure_parent(path)
        temp_path = self._temp_path(path)
        try:
            temp_path.write_bytes(data)
            os.replace(temp_path, path)
        except PermissionError as exc:
            raise PermissionError(
                f"Atomic byte write failed. Temp file was created but rename/replace was denied. "
                f"Use create-only publish for NAS accounts without modify/delete/rename permission. "
                f"temp_path={temp_path}, final_path={path}"
            ) from exc
        finally:
            with suppress(FileNotFoundError):
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except PermissionError:
                        pass

    def atomic_publish_tree(self, source: Path, target: Path, *, replace: bool = True) -> None:
        if not source.exists():
            raise FileNotFoundError(f"Publish source not found: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if replace and target.exists():
            self.safe_remove_tree(target)
        shutil.move(str(source), str(target))

    def verify_checksum(self, path: Path, expected_sha256: str) -> bool:
        return sha256_file(path) == expected_sha256

    def safe_remove_tree(self, path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path)
            return
        path.unlink()

    def _temp_path(self, path: Path) -> Path:
        suffix = f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
        return path.with_name(f"{path.name}{suffix}")
