from __future__ import annotations

import time
from typing import Callable, TypeVar


T = TypeVar("T")


def with_retries(
    operation: Callable[[], T],
    retry_limit: int,
    backoff_seconds: float,
    is_retryable: Callable[[Exception], bool] | None = None,
) -> tuple[T, int]:
    attempts = 0
    while True:
        try:
            attempts += 1
            return operation(), attempts - 1
        except Exception as exc:  # noqa: BLE001
            should_retry = attempts <= retry_limit and (is_retryable(exc) if is_retryable else True)
            if not should_retry:
                raise
            time.sleep(backoff_seconds * attempts)

