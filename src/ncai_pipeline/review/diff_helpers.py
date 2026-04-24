from __future__ import annotations

import difflib


def unified_text_diff(original: str, corrected: str) -> str:
    diff = difflib.unified_diff(
        original.splitlines(),
        corrected.splitlines(),
        fromfile="extracted",
        tofile="corrected",
        lineterm="",
    )
    return "\n".join(diff)

