from __future__ import annotations

from typing import Any

from ncai_pipeline.schemas.common import BoundingBox
from ncai_pipeline.schemas.extraction import ExtractedBlock, ExtractedLine, ExtractedTable, NormalizedExtraction


PARSER_VERSION = "1.0.0"


def _to_bbox(value: Any) -> BoundingBox | None:
    if not value:
        return None
    if isinstance(value, dict):
        try:
            return BoundingBox.model_validate(value)
        except Exception:  # noqa: BLE001
            return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            coordinates = [float(item) for item in value]
        except (TypeError, ValueError):
            return None

        # Gemini responses in this dataset emit bbox arrays as
        # [top, left, bottom, right]. Fall back to left/top/right/bottom
        # if that interpretation is obviously invalid.
        top, left, bottom, right = coordinates
        if right < left or bottom < top:
            left, top, right, bottom = coordinates
            if right < left or bottom < top:
                return None
        return BoundingBox(left=left, top=top, right=right, bottom=bottom)
    return None


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _dict_items(value: Any) -> list[dict[str, Any]]:
    return [item for item in _as_sequence(value) if isinstance(item, dict)]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if item is not None and str(item).strip()]
        return ", ".join(parts) or None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_sequence(value)]


def _language(payload: dict[str, Any], structured_content: dict[str, Any]) -> str | None:
    return _optional_string(payload.get("language") or structured_content.get("language"))


def _table_rows(value: Any) -> list[list[str]]:
    rows = []
    for row in _as_sequence(value):
        if isinstance(row, (list, tuple, set)):
            rows.append([str(item) for item in row])
        elif row is not None:
            rows.append([str(row)])
    return rows


def normalize_provider_payload(document_id: str, page_number: int | None, payload: dict[str, Any]) -> NormalizedExtraction:
    payload = _as_mapping(payload)
    structured_content = _as_mapping(payload.get("structured_content"))
    blocks = [
        ExtractedBlock(
            block_id=str(item.get("block_id", f"block_{index + 1}")),
            text=str(item.get("text", "")),
            bbox=_to_bbox(item.get("bbox")),
            confidence_notes=_optional_string(item.get("confidence_notes")),
            handwritten_auto=item.get("handwritten_auto", item.get("auto_handwritten", item.get("is_handwritten"))),
            handwritten_manual=item.get("handwritten_manual"),
            lines=[
                ExtractedLine(
                    line_id=str(line.get("line_id", f"line_{line_index + 1}")),
                    text=str(line.get("text", "")),
                    bbox=_to_bbox(line.get("bbox")),
                    confidence_notes=_optional_string(line.get("confidence_notes")),
                )
                for line_index, line in enumerate(_dict_items(item.get("lines")))
            ],
        )
        for index, item in enumerate(_dict_items(payload.get("blocks")))
    ]

    lines = [
        ExtractedLine(
            line_id=str(item.get("line_id", f"line_{index + 1}")),
            text=str(item.get("text", "")),
            bbox=_to_bbox(item.get("bbox")),
            confidence_notes=_optional_string(item.get("confidence_notes")),
        )
        for index, item in enumerate(_dict_items(payload.get("lines")))
    ]

    tables = [
        ExtractedTable(
            table_id=str(item.get("table_id", f"table_{index + 1}")),
            rows=_table_rows(item.get("rows")),
            confidence_notes=_optional_string(item.get("confidence_notes")),
        )
        for index, item in enumerate(_dict_items(payload.get("tables")))
    ]

    return NormalizedExtraction(
        document_id=document_id,
        page_number=page_number,
        full_text=str(payload.get("full_text", "")),
        structured_content=structured_content,
        blocks=blocks,
        lines=lines,
        tables=tables,
        warnings=_string_list(payload.get("warnings")),
        confidence_notes=_string_list(payload.get("confidence_notes")),
        parser_version=PARSER_VERSION,
        language=_language(payload, structured_content),
    )
