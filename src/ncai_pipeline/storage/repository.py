from __future__ import annotations

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

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def write_yaml(self, path: Path, data: BaseModel | dict[str, Any]) -> None:
        payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
        self.atomic_write_text(path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))

    def read_yaml(self, path: Path) -> Any:
        return yaml.safe_load(path.read_text(encoding="utf-8-sig"))

    def write_text(self, path: Path, data: str) -> None:
        self.atomic_write_text(path, data)

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

    def copy_file(self, source: Path, target: Path) -> None:
        self.ensure_parent(target)
        shutil.copy2(source, target)

    def copy_tree(self, source: Path, target: Path, *, replace: bool = False) -> None:
        if replace and target.exists():
            self.safe_remove_tree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=replace)

    def atomic_write_json(self, path: Path, data: BaseModel | dict[str, Any] | list[Any]) -> None:
        self.write_json(path, data)

    def atomic_write_yaml(self, path: Path, data: BaseModel | dict[str, Any]) -> None:
        self.write_yaml(path, data)

    def atomic_write_text(self, path: Path, data: str) -> None:
        self.ensure_parent(path)
        temp_path = self._temp_path(path)
        try:
            temp_path.write_text(data, encoding="utf-8")
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def atomic_write_bytes(self, path: Path, data: bytes) -> None:
        self.ensure_parent(path)
        temp_path = self._temp_path(path)
        try:
            temp_path.write_bytes(data)
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

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
