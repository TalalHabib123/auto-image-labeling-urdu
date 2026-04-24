from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import uuid4


@dataclass
class RunContext:
    run_id: str
    started_at: float


def create_run_context() -> RunContext:
    return RunContext(run_id=f"run_{uuid4().hex[:12]}", started_at=time.time())


@contextmanager
def timed() -> float:
    start = time.perf_counter()
    try:
        yield start
    finally:
        _ = time.perf_counter() - start

