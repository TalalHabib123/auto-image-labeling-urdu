from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ncai_pipeline.config import AppConfig
from ncai_pipeline.storage.paths import DataPaths


@dataclass(frozen=True)
class StorageRuntime:
    storage_mode: str
    input_root: Path
    output_root: Path
    final_paths: DataPaths
    scratch_root: Path | None = None
    scratch_paths: DataPaths | None = None
    metadata: dict[str, Any] | None = None

    @property
    def is_nas(self) -> bool:
        return self.storage_mode == "nas"

    def for_run(self, run_id: str) -> "StorageRuntime":
        if not self.is_nas:
            return self
        assert self.scratch_root is not None
        run_scratch_root = self.scratch_root / "active" / run_id
        metadata = dict(self.metadata or {})
        metadata["run_scratch_root"] = str(run_scratch_root)
        return StorageRuntime(
            storage_mode=self.storage_mode,
            input_root=self.input_root,
            output_root=self.output_root,
            final_paths=self.final_paths,
            scratch_root=self.scratch_root,
            scratch_paths=DataPaths(run_scratch_root),
            metadata=metadata,
        )


def resolve_storage_runtime(config: AppConfig) -> StorageRuntime:
    if config.pipeline.storage_mode == "local":
        output_root = Path(config.pipeline.output_root)
        return StorageRuntime(
            storage_mode="local",
            input_root=Path(config.pipeline.input_dir),
            output_root=output_root,
            final_paths=DataPaths(output_root),
            scratch_root=None,
            scratch_paths=None,
            metadata={
                "storage_mode": "local",
                "input_root": str(Path(config.pipeline.input_dir)),
                "output_root": str(output_root),
            },
        )

    assert config.nas.input_root is not None
    assert config.nas.output_root is not None
    assert config.nas.scratch_root is not None
    input_root = Path(config.nas.input_root)
    output_root = Path(config.nas.output_root)
    scratch_root = Path(config.nas.scratch_root)
    return StorageRuntime(
        storage_mode="nas",
        input_root=input_root,
        output_root=output_root,
        final_paths=DataPaths(output_root),
        scratch_root=scratch_root,
        scratch_paths=DataPaths(scratch_root),
        metadata={
            "storage_mode": "nas",
            "source_root": str(input_root),
            "output_root": str(output_root),
            "scratch_root": str(scratch_root),
            "nas_share_label": config.nas.share_label,
            "publish_strategy": config.nas.publish_strategy,
        },
    )
