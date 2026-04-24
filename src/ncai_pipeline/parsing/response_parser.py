from __future__ import annotations

from typing import Any


def ensure_response_dict(response_payload: dict[str, Any] | None) -> dict[str, Any]:
    if response_payload is None:
        return {}
    return response_payload

