from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class FileRepository:
    def ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def write_json(self, path: Path, data: BaseModel | dict[str, Any] | list[Any]) -> None:
        self.ensure_parent(path)
        payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def write_yaml(self, path: Path, data: BaseModel | dict[str, Any]) -> None:
        self.ensure_parent(path)
        payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def read_yaml(self, path: Path) -> Any:
        return yaml.safe_load(path.read_text(encoding="utf-8-sig"))

    def write_text(self, path: Path, data: str) -> None:
        self.ensure_parent(path)
        path.write_text(data, encoding="utf-8")

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8-sig")
