from __future__ import annotations

from typing import Any

from ncai_pipeline.schemas.common import BoundingBox
from ncai_pipeline.schemas.extraction import ExtractedBlock, ExtractedLine, ExtractedTable, NormalizedExtraction


PARSER_VERSION = "1.0.0"


def _to_bbox(value: dict[str, Any] | None) -> BoundingBox | None:
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


def normalize_provider_payload(document_id: str, page_number: int | None, payload: dict[str, Any]) -> NormalizedExtraction:
    blocks = [
        ExtractedBlock(
            block_id=str(item.get("block_id", f"block_{index + 1}")),
            text=str(item.get("text", "")),
            bbox=_to_bbox(item.get("bbox")),
            confidence_notes=item.get("confidence_notes"),
            handwritten_auto=item.get("handwritten_auto", item.get("auto_handwritten", item.get("is_handwritten"))),
            handwritten_manual=item.get("handwritten_manual"),
            lines=[
                ExtractedLine(
                    line_id=str(line.get("line_id", f"line_{line_index + 1}")),
                    text=str(line.get("text", "")),
                    bbox=_to_bbox(line.get("bbox")),
                    confidence_notes=line.get("confidence_notes"),
                )
                for line_index, line in enumerate(item.get("lines", []))
            ],
        )
        for index, item in enumerate(payload.get("blocks", []))
    ]

    lines = [
        ExtractedLine(
            line_id=str(item.get("line_id", f"line_{index + 1}")),
            text=str(item.get("text", "")),
            bbox=_to_bbox(item.get("bbox")),
            confidence_notes=item.get("confidence_notes"),
        )
        for index, item in enumerate(payload.get("lines", []))
    ]

    tables = [
        ExtractedTable(
            table_id=str(item.get("table_id", f"table_{index + 1}")),
            rows=[list(map(str, row)) for row in item.get("rows", [])],
            confidence_notes=item.get("confidence_notes"),
        )
        for index, item in enumerate(payload.get("tables", []))
    ]

    return NormalizedExtraction(
        document_id=document_id,
        page_number=page_number,
        full_text=str(payload.get("full_text", "")),
        structured_content=dict(payload.get("structured_content", {})),
        blocks=blocks,
        lines=lines,
        tables=tables,
        warnings=[str(item) for item in payload.get("warnings", [])],
        confidence_notes=[str(item) for item in payload.get("confidence_notes", [])],
        parser_version=PARSER_VERSION,
        language=payload.get("language") or payload.get("structured_content", {}).get("language"),
    )
