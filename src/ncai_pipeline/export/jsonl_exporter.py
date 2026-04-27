from __future__ import annotations

from pathlib import Path

from ncai_pipeline.schemas.export import CuratedTrainingSample
from ncai_pipeline.export.writers import write_jsonl as write_jsonl_records


def write_jsonl(path: Path, samples: list[CuratedTrainingSample]) -> None:
    write_jsonl_records(path, [sample.model_dump(mode="json") for sample in samples])
