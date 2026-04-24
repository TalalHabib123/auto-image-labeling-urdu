from __future__ import annotations

from ncai_pipeline.config import ExportRatios
from ncai_pipeline.utils.hashing import stable_split_score


def assign_split(document_id: str, ratios: ExportRatios) -> str:
    score = stable_split_score(document_id)
    if score < ratios.train:
        return "train"
    if score < ratios.train + ratios.val:
        return "val"
    return "test"

