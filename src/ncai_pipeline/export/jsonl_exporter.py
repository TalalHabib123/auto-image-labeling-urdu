from __future__ import annotations

import json
from pathlib import Path

from ncai_pipeline.schemas.export import CuratedTrainingSample


def write_jsonl(path: Path, samples: list[CuratedTrainingSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(sample.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) for sample in samples]
    path.write_text("\n".join(lines), encoding="utf-8")

