from __future__ import annotations


def normalize_concurrency(requested: int) -> int:
    return max(1, requested)

