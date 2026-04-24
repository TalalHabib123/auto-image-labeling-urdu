from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class PipelineMetrics:
    counters: Counter = field(default_factory=Counter)
    latencies_ms: list[int] = field(default_factory=list)
    retry_counts: list[int] = field(default_factory=list)

    def increment(self, key: str, amount: int = 1) -> None:
        self.counters[key] += amount

    def add_latency(self, latency_ms: int | None) -> None:
        if latency_ms is not None:
            self.latencies_ms.append(latency_ms)

    def add_retries(self, retry_count: int) -> None:
        self.retry_counts.append(retry_count)

    def summary(self) -> dict[str, object]:
        average_latency = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0
        total_retries = sum(self.retry_counts)
        return {
            **dict(self.counters),
            "average_latency_ms": round(average_latency, 2),
            "retry_count": total_retries,
        }

