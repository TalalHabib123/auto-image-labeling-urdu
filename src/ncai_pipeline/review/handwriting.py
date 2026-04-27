from __future__ import annotations

from typing import Any, Iterable

from ncai_pipeline.schemas.extraction import ExtractedBlock


DEFAULT_HANDWRITING_KEYWORDS = [
    "handwritten",
    "handwriting",
    "hand-written",
    "written by hand",
    "manuscript",
    "دستی",
    "دستخط",
    "تحریر",
]


def text_mentions_handwriting(value: str | None, keywords: Iterable[str] | None = None) -> bool:
    if not value:
        return False
    normalized = value.lower()
    return any(keyword.strip().lower() in normalized for keyword in (keywords or DEFAULT_HANDWRITING_KEYWORDS) if keyword.strip())


def block_auto_handwritten(block: ExtractedBlock, keywords: Iterable[str] | None = None) -> bool:
    if block.handwritten_auto is not None:
        return bool(block.handwritten_auto)
    if text_mentions_handwriting(block.confidence_notes, keywords):
        return True

    extra_values: list[Any] = []
    for field_name in ("warnings", "warning", "handwriting_warnings", "notes"):
        value = getattr(block, field_name, None)
        if value is not None:
            extra_values.append(value)

    for value in extra_values:
        if isinstance(value, str) and text_mentions_handwriting(value, keywords):
            return True
        if isinstance(value, list) and any(text_mentions_handwriting(str(item), keywords) for item in value):
            return True
    return False


def block_handwriting_status(
    block: ExtractedBlock,
    *,
    keywords: Iterable[str] | None = None,
    auto_detect: bool = True,
) -> dict[str, bool | str]:
    manual = block.handwritten_manual
    auto = block_auto_handwritten(block, keywords) if auto_detect else False
    if manual is not None:
        return {
            "handwritten_auto": auto,
            "handwritten_manual": bool(manual),
            "handwritten_effective": bool(manual),
            "handwritten_source": "manual",
        }
    return {
        "handwritten_auto": auto,
        "handwritten_manual": False,
        "handwritten_effective": auto,
        "handwritten_source": "auto" if auto else "none",
    }
