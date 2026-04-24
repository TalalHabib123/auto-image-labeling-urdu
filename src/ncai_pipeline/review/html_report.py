from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template

from ncai_pipeline.review.diff_helpers import unified_text_diff
from ncai_pipeline.schemas.extraction import NormalizedExtraction, RawExtractionRecord
from ncai_pipeline.schemas.review import CorrectedDocument


REPORT_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{{ title }} - {{ document_id }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f5f5f5; color: #111; }
    h1, h2 { margin-bottom: 8px; }
    .page { background: #fff; border: 1px solid #ddd; margin-bottom: 24px; padding: 16px; }
    .page img { max-width: 100%; border: 1px solid #ccc; margin-bottom: 16px; }
    pre { white-space: pre-wrap; word-wrap: break-word; background: #fafafa; padding: 12px; border: 1px solid #ddd; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <p><strong>Document ID:</strong> {{ document_id }}</p>
  <p><strong>Correction File:</strong> {{ correction_path }}</p>
  {% for page in pages %}
  <div class="page">
    <h2>Page {{ page.page_number }}</h2>
    <img src="{{ page.image_path }}" alt="Source page {{ page.page_number }}" />
    <div class="grid">
      <div>
        <h3>Normalized Extraction</h3>
        <pre>{{ page.normalized_json }}</pre>
      </div>
      <div>
        <h3>Raw Provider Response</h3>
        <pre>{{ page.raw_response }}</pre>
      </div>
    </div>
  </div>
  {% endfor %}
  <div class="page">
    <h2>Correction Snapshot</h2>
    <pre>{{ corrected_json }}</pre>
  </div>
  <div class="page">
    <h2>Text Diff</h2>
    <pre>{{ diff_text }}</pre>
  </div>
</body>
</html>
"""
)


def render_review_report(
    title: str,
    document_id: str,
    page_image_paths: list[Path],
    raw_records: list[RawExtractionRecord],
    extractions: list[NormalizedExtraction],
    corrected: CorrectedDocument,
    correction_path: Path,
) -> str:
    pages: list[dict[str, str | int]] = []
    raw_by_page = {record.page_number: record for record in raw_records}
    for extraction in extractions:
        raw_record = raw_by_page.get(extraction.page_number)
        page_image = page_image_paths[(extraction.page_number or 1) - 1]
        pages.append(
            {
                "page_number": extraction.page_number or 0,
                "image_path": Path(page_image).resolve().as_posix(),
                "normalized_json": json.dumps(extraction.model_dump(mode="json"), ensure_ascii=False, indent=2),
                "raw_response": raw_record.raw_response_text if raw_record and raw_record.raw_response_text else json.dumps(raw_record.model_dump(mode="json"), ensure_ascii=False, indent=2) if raw_record else "",
            }
        )

    diff_text = unified_text_diff(
        "\n\n".join(item.full_text for item in extractions),
        corrected.consolidated_text,
    )
    return REPORT_TEMPLATE.render(
        title=title,
        document_id=document_id,
        pages=pages,
        corrected_json=json.dumps(corrected.model_dump(mode="json"), ensure_ascii=False, indent=2),
        correction_path=str(correction_path.resolve()),
        diff_text=diff_text or "No textual changes yet.",
    )

