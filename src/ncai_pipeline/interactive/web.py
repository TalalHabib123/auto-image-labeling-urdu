from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from jinja2 import BaseLoader, Environment, select_autoescape

from ncai_pipeline.interactive.service import (
    InteractiveReviewService,
    serialize_summary,
    serialize_target_output,
    serialize_timestamp,
)
from ncai_pipeline.schemas.common import ReviewStatus
from ncai_pipeline.schemas.review import PageVerificationStatus


TEMPLATE_ENV = Environment(loader=BaseLoader(), autoescape=select_autoescape(("html", "xml")))


def _serialize_script_data(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _bbox_payload_from_fragment(fragment: Any) -> dict[str, float] | None:
    if not getattr(fragment, "has_bbox", False) or getattr(fragment, "bbox", None) is None:
        return None
    bbox = fragment.bbox
    return {
        "left": float(bbox.left or 0),
        "top": float(bbox.top or 0),
        "right": float(bbox.right or 0),
        "bottom": float(bbox.bottom or 0),
    }


def _page_nav_items(
    page_summaries: list[Any],
    selected_page_number: int,
    *,
    edge_count: int = 2,
    radius: int = 4,
) -> list[dict[str, Any]]:
    if len(page_summaries) <= (edge_count * 2) + (radius * 2) + 1:
        return [{"kind": "page", "summary": item} for item in page_summaries]

    last_page = page_summaries[-1].page_number if page_summaries else selected_page_number
    included_pages = {
        item.page_number
        for item in page_summaries
        if item.page_number <= edge_count
        or item.page_number > last_page - edge_count
        or abs(item.page_number - selected_page_number) <= radius
    }

    items: list[dict[str, Any]] = []
    previous_page: int | None = None
    for summary in page_summaries:
        page_number = summary.page_number
        if page_number not in included_pages:
            continue
        if previous_page is not None and page_number - previous_page > 1:
            items.append({"kind": "gap", "start": previous_page + 1, "end": page_number - 1})
        items.append({"kind": "page", "summary": summary})
        previous_page = page_number
    return items


DASHBOARD_TEMPLATE = TEMPLATE_ENV.from_string(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>NCAI Interactive Review</title>
  <style>
    :root {
      --bg: #f4efe5;
      --panel: #fffaf0;
      --ink: #1f1a17;
      --muted: #6a5b52;
      --accent: #9c4f2f;
      --accent-soft: #e9cfbf;
      --line: #d9c8ba;
      --ok: #2f6f4f;
      --warn: #9a6b1d;
      --bad: #8a2f2f;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: radial-gradient(circle at top, #fbf6ef 0%, var(--bg) 58%, #efe4d2 100%); color: var(--ink); font-family: "Segoe UI Variable", "Noto Naskh Arabic", Georgia, serif; }
    .shell { max-width: 1400px; margin: 0 auto; padding: 28px; }
    .hero { display: grid; grid-template-columns: 1.4fr 0.8fr; gap: 20px; margin-bottom: 24px; }
    .card { background: rgba(255,250,240,0.92); border: 1px solid var(--line); border-radius: 18px; padding: 20px; box-shadow: 0 20px 40px rgba(88, 58, 34, 0.08); }
    h1, h2, h3 { margin: 0 0 10px; }
    h1 { font-size: 2rem; letter-spacing: 0.02em; }
    p, li, td, th, label { color: var(--muted); }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
    .stat { background: rgba(255,255,255,0.55); border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; }
    .stat strong { display: block; color: var(--ink); font-size: 1.1rem; margin-top: 4px; }
    form.filters { display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)) auto; gap: 12px; align-items: end; margin-bottom: 18px; }
    select, button, input[type="text"], textarea { width: 100%; border-radius: 12px; border: 1px solid var(--line); padding: 10px 12px; font: inherit; background: #fffdfa; color: var(--ink); }
    button { background: var(--accent); color: white; cursor: pointer; }
    button.secondary, a.secondary { background: transparent; color: var(--accent); border: 1px solid var(--accent); text-decoration: none; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 12px 10px; border-bottom: 1px solid rgba(217, 200, 186, 0.7); vertical-align: top; }
    tbody tr:hover { background: rgba(156, 79, 47, 0.05); }
    .pill { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.85rem; }
    .pipeline-approved, .review-approved, .page-verified { background: rgba(47, 111, 79, 0.16); color: var(--ok); }
    .pipeline-in_review, .review-in_review { background: rgba(156, 79, 47, 0.14); color: var(--accent); }
    .pipeline-needs_retry, .review-needs_retry, .page-needs_fix { background: rgba(154, 107, 29, 0.16); color: var(--warn); }
    .pipeline-failed, .review-rejected { background: rgba(138, 47, 47, 0.16); color: var(--bad); }
    .pipeline-normalized, .pipeline-discovered, .pipeline-extracted, .pipeline-review_pending, .review-pending, .page-pending { background: rgba(106, 91, 82, 0.14); color: var(--muted); }
    .commands pre, pre.json { white-space: pre-wrap; word-break: break-word; background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 14px; color: var(--ink); }
    .top-actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
    .meta { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="card">
        <h1>NCAI Interactive Review</h1>
        <p>Review OCR output against page images, verify pages, and track pipeline/export state from one local dashboard.</p>
        <div class="top-actions">
          <a class="secondary" href="/">Refresh Dashboard</a>
          <a class="secondary" href="/status/run">Latest Run JSON</a>
          <a class="secondary" href="/status/export">Latest Export JSON</a>
        </div>
      </div>
      <div class="card commands">
        <h2>Command Hints</h2>
        <pre>{% for label, command in command_hints.items() %}{{ label }}: {{ command }}
{% endfor %}</pre>
      </div>
    </div>

    <div class="card" style="margin-bottom: 20px;">
      <h2>Pipeline Status</h2>
      <div class="stats">
        {% for key, value in pipeline_counts.items() %}
        <div class="stat"><span>{{ key }}</span><strong>{{ value }}</strong></div>
        {% endfor %}
      </div>
    </div>

    <div class="card" style="margin-bottom: 20px;">
      <h2>Review Status</h2>
      <div class="stats">
        {% for key, value in review_counts.items() %}
        <div class="stat"><span>{{ key }}</span><strong>{{ value }}</strong></div>
        {% endfor %}
      </div>
    </div>

    <div class="hero" style="grid-template-columns: 1fr 1fr;">
      <div class="card">
        <h2>Latest Run Summary</h2>
        <pre class="json">{{ latest_run }}</pre>
      </div>
      <div class="card">
        <h2>Latest Export Summary</h2>
        <pre class="json">{{ latest_export }}</pre>
      </div>
    </div>

    <div class="card">
      <h2>Document Queue</h2>
      <form class="filters" method="get" action="/">
        <label>
          Pipeline Status
          <select name="pipeline_status">
            {% for option in pipeline_options %}
            <option value="{{ option }}" {% if option == selected_pipeline_status %}selected{% endif %}>{{ option }}</option>
            {% endfor %}
          </select>
        </label>
        <label>
          Review Status
          <select name="review_status">
            {% for option in review_options %}
            <option value="{{ option }}" {% if option == selected_review_status %}selected{% endif %}>{{ option }}</option>
            {% endfor %}
          </select>
        </label>
        <label>
          Dry Run
          <select name="include_dry_run">
            <option value="true" {% if include_dry_run %}selected{% endif %}>include</option>
            <option value="false" {% if not include_dry_run %}selected{% endif %}>exclude</option>
          </select>
        </label>
        <button type="submit">Apply Filters</button>
      </form>
      <table>
        <thead>
          <tr>
            <th>Document</th>
            <th>Pipeline</th>
            <th>Review</th>
            <th>Pages</th>
            <th>Source</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {% for item in documents %}
          <tr>
            <td>
              <strong>{{ item.document_id }}</strong>
              <div class="meta">
                {% if item.is_dry_run %}<span class="pill review-pending">dry-run</span>{% endif %}
                {% if item.error_details %}<span class="pill pipeline-failed">error logged</span>{% endif %}
              </div>
            </td>
            <td><span class="pill pipeline-{{ item.pipeline_status }}">{{ item.pipeline_status }}</span></td>
            <td><span class="pill review-{{ item.review_status }}">{{ item.review_status }}</span></td>
            <td>
              {{ item.page_count }}
              {% if item.page_status_counts %}
              <div class="meta">
                <span class="pill page-verified">{{ item.page_status_counts.get("reviewable", 0) }} reviewable</span>
                {% if item.page_status_counts.get("skipped_empty", 0) %}<span class="pill review-pending">{{ item.page_status_counts.get("skipped_empty", 0) }} empty</span>{% endif %}
                {% if item.page_status_counts.get("needs_retry", 0) %}<span class="pill review-needs_retry">{{ item.page_status_counts.get("needs_retry", 0) }} retry</span>{% endif %}
                {% if item.page_status_counts.get("failed", 0) %}<span class="pill pipeline-failed">{{ item.page_status_counts.get("failed", 0) }} failed</span>{% endif %}
              </div>
              {% endif %}
            </td>
            <td>{{ item.source_path }}</td>
            <td>
              {% if item.has_review_bundle %}
              <a class="secondary" href="/documents/{{ item.document_id }}">Review</a>
              {% endif %}
              {% if item.report_path %}
              <a class="secondary" href="/reports/{{ item.document_id }}">Static Report</a>
              {% endif %}
            </td>
          </tr>
          {% else %}
          <tr><td colspan="6">No documents match the selected filters.</td></tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
)


DOCUMENT_TEMPLATE = TEMPLATE_ENV.from_string(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Review {{ context.queue_entry.document_id }}</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --paper: rgba(255, 255, 255, 0.96);
      --paper-strong: #ffffff;
      --line: #d8dee8;
      --line-strong: #a9b5c6;
      --ink: #17202a;
      --muted: #5e6a78;
      --accent: #245b6b;
      --accent-soft: rgba(36, 91, 107, 0.1);
      --ok: #267553;
      --warn: #9a6417;
      --bad: #a43b3b;
      --shadow: 0 14px 32px rgba(26, 39, 55, 0.08);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background: linear-gradient(180deg, #ffffff 0%, var(--bg) 54%, #edf1f6 100%);
      font-family: "Segoe UI Variable", "Segoe UI", "Noto Naskh Arabic", Arial, sans-serif;
      overflow-x: hidden;
    }
    body.modal-open {
      overflow: hidden;
    }
    a, button, input, select, textarea, summary { touch-action: manipulation; }
    a { color: inherit; }
    button, input, select, textarea {
      font: inherit;
      color: var(--ink);
      background: var(--paper-strong);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    button, .ghost-link, .toggle-link {
      transition: background-color 160ms ease, color 160ms ease, border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
    }
    button:focus-visible,
    a:focus-visible,
    summary:focus-visible,
    input:focus-visible,
    select:focus-visible,
    textarea:focus-visible {
      outline: 3px solid rgba(141, 74, 43, 0.28);
      outline-offset: 2px;
    }
    input, select, textarea {
      width: 100%;
      padding: 10px 12px;
    }
    textarea {
      resize: vertical;
      min-height: 120px;
    }
    summary {
      cursor: pointer;
      list-style: none;
    }
    summary::-webkit-details-marker { display: none; }
    .shell {
      width: min(1760px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 14px 0 150px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
      padding: 0 10px 14px;
    }
    .topbar h1 {
      margin: 0;
      font-size: clamp(1.45rem, 1.1rem + 0.9vw, 2.2rem);
      letter-spacing: 0;
      text-wrap: balance;
    }
    .topbar p {
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 60ch;
      word-break: break-word;
    }
    .top-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: flex-end;
    }
    .ghost-link, button, .toggle-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-height: 40px;
      padding: 9px 14px;
      border-radius: 8px;
      border: 1px solid var(--line-strong);
      text-decoration: none;
      background: transparent;
      color: var(--accent);
      font-weight: 650;
      font-size: 0.92rem;
      cursor: pointer;
    }
    .ghost-link:hover, .toggle-link:hover {
      background: var(--accent-soft);
      transform: translateY(-1px);
    }
    button {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
      box-shadow: 0 10px 22px rgba(36, 91, 107, 0.18);
    }
    button:hover {
      background: #1d4c5a;
      border-color: #1d4c5a;
      transform: translateY(-1px);
    }
    .state-rail {
      position: sticky;
      top: 8px;
      z-index: 20;
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 0;
      margin: 0 10px 18px;
      padding: 10px;
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid rgba(169, 181, 198, 0.6);
      border-radius: 8px;
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow);
    }
    .rail-item {
      min-width: 0;
      padding: 6px 10px;
      border-left: 1px solid rgba(169, 181, 198, 0.38);
    }
    .rail-item:first-child { border-left: 0; }
    .rail-label {
      display: block;
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 0.72rem;
      letter-spacing: 0;
      text-transform: uppercase;
    }
    .rail-value {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      font-size: 0.95rem;
      font-weight: 700;
      word-break: break-word;
    }
    .workspace-summary + .rail-value,
    .system-alert-summary + .rail-value,
    .page-dimensions + .panel-subtle,
    .fragment-summary + p {
      display: none;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 700;
      white-space: nowrap;
    }
    .pipeline-approved, .page-verified, .review-approved { background: rgba(47,111,79,0.16); color: var(--ok); }
    .pipeline-review_pending, .pipeline-discovered, .pipeline-normalized, .pipeline-extracted, .page-pending, .review-pending { background: rgba(105,93,85,0.14); color: var(--muted); }
    .pipeline-in_review, .review-in_review { background: rgba(141, 74, 43, 0.14); color: var(--accent); }
    .pipeline-needs_retry, .page-needs_fix, .review-needs_retry { background: rgba(154,107,29,0.16); color: var(--warn); }
    .pipeline-failed, .pipeline-rejected, .review-rejected { background: rgba(138,47,47,0.16); color: var(--bad); }
    .notices {
      display: grid;
      gap: 10px;
      margin: 0 10px 18px;
    }
    .notice {
      padding: 12px 14px;
      border-radius: 8px;
      border: 1px solid rgba(169, 181, 198, 0.6);
      background: rgba(255, 255, 255, 0.96);
      color: var(--ink);
      box-shadow: 0 12px 26px rgba(26, 39, 55, 0.06);
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(226px, 260px) minmax(360px, 0.92fr) minmax(420px, 1.08fr);
      gap: 14px;
      align-items: start;
      padding: 0 10px;
    }
    .panel {
      min-width: 0;
      padding: 16px;
      border-radius: 8px;
      border: 1px solid rgba(169, 181, 198, 0.62);
      background: var(--paper);
      box-shadow: var(--shadow);
    }
    .panel h2, .panel h3 {
      margin: 0;
      text-wrap: balance;
    }
    .panel-heading {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }
    .panel-subtle {
      color: var(--muted);
      font-size: 0.9rem;
    }
    .page-panel {
      position: sticky;
      top: 118px;
      display: grid;
      grid-template-rows: auto auto auto minmax(0, 1fr);
      max-height: calc(100vh - 136px);
      overflow: hidden;
    }
    .page-jump {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      margin-bottom: 10px;
    }
    .page-jump input {
      min-height: 38px;
    }
    .page-jump button {
      min-width: 54px;
      box-shadow: none;
    }
    .page-stepper {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    .page-stepper .ghost-link,
    .page-stepper .page-step-disabled {
      min-height: 36px;
      padding: 8px 10px;
      font-size: 0.86rem;
    }
    .page-step-disabled {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: #8a96a5;
      background: #f3f5f8;
      font-weight: 650;
    }
    .page-nav {
      display: grid;
      align-content: start;
      gap: 8px;
      min-height: 0;
      overflow: auto;
      padding-right: 4px;
      scrollbar-width: thin;
    }
    .page-link {
      display: grid;
      gap: 7px;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid rgba(169, 181, 198, 0.62);
      background: #ffffff;
      text-decoration: none;
      min-width: 0;
      transition: border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease, background-color 160ms ease;
    }
    .page-link:hover {
      transform: translateY(-1px);
      border-color: var(--accent);
      box-shadow: 0 10px 20px rgba(26, 39, 55, 0.08);
    }
    .page-link.active {
      background: rgba(36, 91, 107, 0.08);
      border-color: var(--accent);
    }
    .page-link-header,
    .page-link-meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 5px 8px;
      align-items: center;
      min-width: 0;
    }
    .page-link strong {
      min-width: 0;
    }
    .meta-text {
      color: var(--muted);
      font-size: 0.82rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .page-gap {
      padding: 5px 8px;
      color: var(--muted);
      font-size: 0.8rem;
      text-align: center;
    }
    .preview-toggle {
      display: inline-flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .toggle-link.active {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      box-shadow: 0 10px 22px rgba(36, 91, 107, 0.18);
    }
    .image-frame,
    .preview-frame {
      border-radius: 8px;
      border: 1px solid rgba(169, 181, 198, 0.62);
      overflow: hidden;
      background: white;
    }
    .media-frame {
      position: relative;
      min-height: 320px;
      aspect-ratio: var(--media-aspect, auto);
      isolation: isolate;
    }
    .media-buffer {
      position: absolute;
      inset: 0;
      z-index: 1;
      display: grid;
      place-items: center;
      gap: 10px;
      align-content: center;
      color: var(--muted);
      background:
        linear-gradient(90deg, rgba(245,247,250,0.85), rgba(255,255,255,0.96), rgba(245,247,250,0.85));
      background-size: 220% 100%;
      animation: media-buffer-sheen 1.4s ease-in-out infinite;
      transition: opacity 160ms ease, visibility 160ms ease;
    }
    .media-buffer span:last-child {
      font-size: 0.9rem;
      font-weight: 650;
    }
    .loader {
      width: 28px;
      height: 28px;
      border-radius: 999px;
      border: 3px solid rgba(36, 91, 107, 0.16);
      border-top-color: var(--accent);
      animation: media-loader-spin 800ms linear infinite;
    }
    .media-frame.is-loaded .media-buffer {
      opacity: 0;
      visibility: hidden;
    }
    .image-frame img,
    .preview-frame img,
    .preview-frame object {
      display: block;
      width: 100%;
      height: auto;
      background: white;
    }
    .media-frame > img,
    .media-frame > object {
      opacity: 0;
      transition: opacity 180ms ease;
    }
    .media-frame.is-loaded > img,
    .media-frame.is-loaded > object {
      opacity: 1;
    }
    .preview-frame object {
      min-height: 760px;
      border: 0;
    }
    @keyframes media-loader-spin {
      to { transform: rotate(360deg); }
    }
    @keyframes media-buffer-sheen {
      0% { background-position: 140% 0; }
      100% { background-position: -80% 0; }
    }
    .bbox-trigger-group {
      display: grid;
      justify-items: end;
      gap: 8px;
      text-align: right;
    }
    .bbox-trigger-group button {
      width: auto;
    }
    .bbox-modal[hidden] {
      display: none;
    }
    .bbox-modal {
      position: fixed;
      inset: 0;
      z-index: 40;
    }
    .bbox-modal-backdrop {
      position: absolute;
      inset: 0;
      background: rgba(20, 14, 10, 0.74);
      backdrop-filter: blur(6px);
    }
    .bbox-modal-panel {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-rows: auto 1fr;
      width: min(1700px, calc(100vw - 32px));
      height: min(960px, calc(100vh - 32px));
      margin: 16px auto;
      border-radius: 28px;
      border: 1px solid rgba(185, 147, 115, 0.34);
      background: rgba(255, 251, 246, 0.98);
      box-shadow: 0 36px 80px rgba(38, 24, 14, 0.28);
      overflow: hidden;
    }
    .bbox-modal-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      padding: 20px 24px 16px;
      border-bottom: 1px solid rgba(185, 147, 115, 0.26);
      background: linear-gradient(180deg, rgba(255, 250, 243, 0.98), rgba(252, 244, 234, 0.9));
    }
    .bbox-modal-header h2 {
      margin: 0;
    }
    .bbox-modal-header p {
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 72ch;
    }
    .bbox-modal-body {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(340px, 0.7fr);
      gap: 0;
      min-height: 0;
    }
    .bbox-stage {
      display: grid;
      gap: 12px;
      padding: 22px 24px 24px;
      min-width: 0;
      min-height: 0;
      background:
        radial-gradient(circle at top, rgba(255,255,255,0.65), transparent 32%),
        linear-gradient(180deg, #f9f2e8 0%, #f5eadb 100%);
    }
    .bbox-stage-copy {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    .bbox-stage-copy p {
      margin: 0;
      color: var(--muted);
    }
    .bbox-canvas-wrap {
      min-height: 0;
      overflow: auto;
      border-radius: 22px;
      border: 1px solid rgba(185, 147, 115, 0.24);
      background:
        linear-gradient(45deg, rgba(185, 147, 115, 0.06) 25%, transparent 25%, transparent 75%, rgba(185, 147, 115, 0.06) 75%),
        linear-gradient(45deg, rgba(185, 147, 115, 0.06) 25%, transparent 25%, transparent 75%, rgba(185, 147, 115, 0.06) 75%),
        #fffaf4;
      background-position: 0 0, 14px 14px, 0 0;
      background-size: 28px 28px;
      padding: 18px;
    }
    .bbox-canvas {
      position: relative;
      display: inline-block;
      line-height: 0;
      max-width: 100%;
    }
    .bbox-canvas img {
      display: block;
      max-width: min(72vw, 1100px);
      width: 100%;
      height: auto;
      border-radius: 18px;
      box-shadow: 0 22px 50px rgba(45, 30, 18, 0.18);
      user-select: none;
      -webkit-user-drag: none;
    }
    .bbox-overlay {
      position: absolute;
      inset: 0;
      cursor: crosshair;
    }
    .bbox-box {
      position: absolute;
      border-radius: 12px;
      border: 2px solid rgba(141, 74, 43, 0.92);
      background: rgba(141, 74, 43, 0.16);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.42);
      cursor: move;
    }
    .bbox-box.is-selected {
      border-color: #1f6ad4;
      background: rgba(31, 106, 212, 0.18);
      box-shadow: 0 0 0 2px rgba(31, 106, 212, 0.16), inset 0 0 0 1px rgba(255,255,255,0.52);
    }
    .bbox-box-label {
      position: absolute;
      top: -28px;
      left: 0;
      display: inline-flex;
      align-items: center;
      max-width: min(280px, 100%);
      padding: 5px 10px;
      border-radius: 999px;
      background: rgba(24, 18, 14, 0.86);
      color: #fff7ed;
      font-size: 0.82rem;
      font-weight: 700;
      line-height: 1.2;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      pointer-events: none;
    }
    .bbox-handle {
      position: absolute;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      background: white;
      border: 2px solid currentColor;
      color: inherit;
      box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    }
    .bbox-handle[data-handle="nw"] { top: -7px; left: -7px; cursor: nwse-resize; }
    .bbox-handle[data-handle="ne"] { top: -7px; right: -7px; cursor: nesw-resize; }
    .bbox-handle[data-handle="sw"] { bottom: -7px; left: -7px; cursor: nesw-resize; }
    .bbox-handle[data-handle="se"] { bottom: -7px; right: -7px; cursor: nwse-resize; }
    .bbox-sidebar {
      display: grid;
      align-content: start;
      gap: 14px;
      padding: 22px 24px 24px;
      border-left: 1px solid rgba(185, 147, 115, 0.24);
      background: rgba(255, 253, 249, 0.98);
      min-height: 0;
      overflow: auto;
    }
    .bbox-sidebar label {
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .bbox-fragment-preview {
      min-height: 110px;
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(185, 147, 115, 0.26);
      background: #fffdf8;
      color: var(--ink);
      white-space: pre-wrap;
      overflow: auto;
      line-height: 1.55;
    }
    .bbox-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .bbox-actions {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .bbox-save-button {
      grid-column: 1 / -1;
    }
    .bbox-note-surface {
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(185, 147, 115, 0.26);
      background: rgba(255, 249, 242, 0.92);
      color: var(--muted);
      line-height: 1.55;
    }
    .bbox-note-surface strong {
      display: block;
      margin-bottom: 8px;
      color: var(--ink);
    }
    .image-stack {
      display: grid;
      gap: 16px;
      position: sticky;
      top: 118px;
    }
    .crop-empty {
      padding: 20px;
      border-radius: 18px;
      border: 1px dashed rgba(185, 147, 115, 0.4);
      color: var(--muted);
      background: rgba(255, 252, 247, 0.75);
    }
    .editor-stage {
      grid-column: 1 / -1;
      display: grid;
      gap: 18px;
    }
    .editor-form {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    .editor-form > .data-drawer {
      grid-column: 1 / -1;
    }
    .fragment-editor {
      display: grid;
      gap: 14px;
      padding: 14px;
      border-radius: 8px;
      border: 1px solid rgba(169, 181, 198, 0.56);
      background: rgba(255, 255, 255, 0.94);
      scroll-margin-top: 118px;
    }
    .fragment-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 10px;
    }
    .fragment-head p,
    .fragment-note,
    .empty-copy,
    .drawer-copy,
    .save-copy p {
      margin: 6px 0 0;
      color: var(--muted);
    }
    .jump-link {
      font-size: 0.92rem;
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
    }
    .fragment-editor label,
    .settings-grid label,
    .document-form label {
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .fragment-editor textarea {
      min-height: 140px;
      background: white;
      font-size: 1.25rem;
      direction: rtl;
      text-align: right;
      unicode-bidi: plaintext;
    }
    .fragment-crop-frame {
      overflow: hidden;
      min-height: 132px;
      border-radius: 8px;
      border: 1px solid rgba(169, 181, 198, 0.56);
      background: white;
    }
    .fragment-crop-frame img {
      display: block;
      width: 100%;
      height: auto;
      max-height: 320px;
      object-fit: contain;
      background: white;
    }
    .fragment-crop-frame img:not([src]) {
      visibility: hidden;
      min-height: 132px;
    }
    .fragment-crop-meta {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .fragment-flags {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin: 12px 0;
    }
    .fragment-flag {
      width: auto;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid rgba(169, 181, 198, 0.62);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.94);
      color: var(--ink);
      font-weight: 700;
    }
    .fragment-flag input {
      width: auto;
      min-width: 18px;
      min-height: 18px;
      padding: 0;
    }
    .handwriting-auto { background: rgba(154,107,29,0.16); color: var(--warn); }
    .handwriting-manual { background: rgba(47,111,79,0.16); color: var(--ok); }
    .fragment-empty {
      padding: 18px;
      border-radius: 8px;
      border: 1px dashed rgba(169, 181, 198, 0.7);
      background: rgba(246, 248, 251, 0.9);
      color: var(--muted);
    }
    .data-drawer,
    .settings-drawer {
      border-top: 1px solid rgba(185, 147, 115, 0.3);
      padding-top: 12px;
    }
    .data-drawer summary,
    .settings-drawer summary {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 0;
      font-weight: 700;
      color: var(--ink);
    }
    .data-drawer summary::after,
    .settings-drawer summary::after {
      content: "+";
      color: var(--accent);
      font-size: 1.2rem;
      line-height: 1;
    }
    .data-drawer[open] summary::after,
    .settings-drawer[open] summary::after {
      content: "−";
    }
    .data-drawer[open] summary::after,
    .settings-drawer[open] summary::after {
      content: "-";
    }
    pre {
      margin: 12px 0 0;
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      border-radius: 8px;
      border: 1px solid rgba(169, 181, 198, 0.56);
      background: #ffffff;
      padding: 14px;
      color: var(--ink);
    }
    .empty-copy {
      padding: 18px;
      border-radius: 8px;
      border: 1px dashed rgba(169, 181, 198, 0.7);
      background: rgba(246, 248, 251, 0.9);
    }
    .save-bar {
      position: sticky;
      bottom: 0;
      z-index: 18;
      display: grid;
      gap: 14px;
      margin: 18px 10px 0;
      padding: 14px;
      border-radius: 8px 8px 0 0;
      border: 1px solid rgba(169, 181, 198, 0.62);
      border-bottom: 0;
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(16px);
      box-shadow: 0 -18px 42px rgba(26, 39, 55, 0.1);
      transition: padding 180ms ease, box-shadow 180ms ease, transform 180ms ease;
    }
    .save-bar-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    .save-bar-heading {
      display: grid;
      gap: 8px;
      min-width: 0;
    }
    .save-bar-heading h2 {
      margin: 0;
    }
    .save-bar-heading p {
      margin: 0;
      color: var(--muted);
    }
    .save-bar-header-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .save-bar-toggle {
      min-width: 188px;
      box-shadow: none;
    }
    .save-bar-body {
      display: grid;
      gap: 14px;
      max-height: 1800px;
      opacity: 1;
      overflow: hidden;
      transform: translateY(0);
      transition: max-height 240ms ease, opacity 180ms ease, transform 240ms ease;
    }
    .save-bar.is-collapsed {
      padding-block: 12px;
      box-shadow: 0 -10px 26px rgba(26, 39, 55, 0.08);
    }
    .save-bar.is-collapsed .save-bar-body {
      max-height: 0;
      opacity: 0;
      pointer-events: none;
      transform: translateY(10px);
    }
    .save-primary {
      display: grid;
      grid-template-columns: 1.15fr 1.35fr auto;
      gap: 16px;
      align-items: start;
    }
    .save-copy h2,
    .document-form h3 {
      margin: 0;
    }
    .review-readiness {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-top: 14px;
    }
    .readiness-note {
      color: var(--muted);
      font-size: 0.95rem;
    }
    .settings-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      min-width: 0;
    }
    .settings-grid textarea {
      min-height: 88px;
    }
    .save-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: flex-end;
    }
    .button-secondary,
    .button-warn,
    .button-danger {
      box-shadow: none;
    }
    .button-secondary {
      background: transparent;
      color: var(--accent);
      border-color: var(--line-strong);
    }
    .button-secondary:hover {
      background: var(--accent-soft);
      border-color: var(--accent);
    }
    .button-warn {
      background: #a36b1f;
      border-color: #a36b1f;
    }
    .button-warn:hover {
      background: #8a5916;
      border-color: #8a5916;
    }
    .button-danger {
      background: var(--bad);
      border-color: var(--bad);
    }
    .button-danger:hover {
      background: #712525;
      border-color: #712525;
    }
    button[disabled] {
      background: #d8cfc5;
      border-color: #d8cfc5;
      color: #6c6057;
      box-shadow: none;
      cursor: not-allowed;
      transform: none;
    }
    button[disabled]:hover {
      background: #d8cfc5;
      border-color: #d8cfc5;
      transform: none;
    }
    .review-panel {
      display: grid;
      gap: 14px;
      padding: 16px;
      border-radius: 8px;
      border: 1px solid rgba(169, 181, 198, 0.62);
      background: rgba(255, 255, 255, 0.94);
    }
    .review-panel-header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
    }
    .review-panel-header p {
      margin: 8px 0 0;
      color: var(--muted);
    }
    .decision-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .decision-grid button {
      width: 100%;
    }
    .document-drawers {
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.9fr);
      gap: 14px;
    }
    .document-form {
      display: grid;
      gap: 12px;
      padding-top: 8px;
    }
    .document-form textarea {
      min-height: 160px;
    }
    .document-form .compact {
      min-height: 96px;
    }
    .save-bar-sentinel {
      height: 1px;
      margin-top: 18px;
    }
    .page-transition[hidden] {
      display: none;
    }
    .page-transition {
      position: fixed;
      inset: 0;
      z-index: 60;
      display: grid;
      place-items: center;
      background: rgba(245, 247, 250, 0.74);
      backdrop-filter: blur(6px);
    }
    .page-transition-card {
      display: grid;
      gap: 12px;
      justify-items: center;
      min-width: 240px;
      padding: 22px 24px;
      border-radius: 8px;
      border: 1px solid rgba(169, 181, 198, 0.7);
      background: #ffffff;
      box-shadow: 0 18px 42px rgba(26, 39, 55, 0.14);
      color: var(--muted);
      font-weight: 650;
    }
    @media (max-width: 1400px) {
      .state-rail { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .save-primary { grid-template-columns: 1fr; }
      .save-actions { justify-content: flex-start; }
      .decision-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 1180px) {
      .workspace { grid-template-columns: 1fr; }
      .page-panel,
      .image-stack { position: static; max-height: none; }
      .page-nav { max-height: 320px; }
      .editor-form { grid-template-columns: 1fr; }
      .document-drawers,
      .settings-grid,
      .state-rail { grid-template-columns: 1fr 1fr; }
      .bbox-modal-panel {
        width: calc(100vw - 18px);
        height: calc(100vh - 18px);
        margin: 9px;
      }
      .bbox-modal-body {
        grid-template-columns: 1fr;
      }
      .bbox-canvas img {
        max-width: 100%;
      }
    }
    @media (max-width: 760px) {
      .shell { width: min(100vw - 18px, 100%); padding-top: 12px; }
      .topbar { flex-direction: column; padding-inline: 4px; }
      .top-actions { justify-content: flex-start; }
      .save-bar-header { flex-direction: column; align-items: flex-start; }
      .save-bar-toggle { width: 100%; min-width: 0; }
      .save-bar.is-collapsed .save-bar-heading,
      .save-bar.is-collapsed .save-bar-header-meta .pill {
        display: none;
      }
      .save-bar.is-collapsed .save-bar-header-meta {
        width: 100%;
      }
      .workspace,
      .document-drawers,
      .settings-grid,
      .state-rail { grid-template-columns: 1fr; }
      .panel, .save-bar { border-radius: 8px; }
      .save-bar { border-bottom: 1px solid rgba(169, 181, 198, 0.62); }
      .bbox-actions,
      .bbox-grid { grid-template-columns: 1fr; }
      .bbox-modal-header,
      .bbox-stage,
      .bbox-sidebar { padding-inline: 16px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div>
        <h1>{{ context.queue_entry.document_id }}</h1>
        <p>{{ context.queue_entry.source_path }}</p>
      </div>
      <div class="top-actions">
        <a class="ghost-link" href="/">Back To Dashboard</a>
        <a class="ghost-link" href="/reports/{{ context.queue_entry.document_id }}">Static Report</a>
      </div>
    </header>

    <section class="state-rail" aria-label="Document state">
      <div class="rail-item">
        <span class="rail-label">Pipeline</span>
        <span class="rail-value"><span class="pill pipeline-{{ context.queue_entry.pipeline_status }}">{{ context.queue_entry.pipeline_status }}</span></span>
      </div>
      <div class="rail-item">
        <span class="rail-label">Review</span>
        <span class="rail-value"><span class="pill review-{{ context.queue_entry.review_status }}">{{ context.queue_entry.review_status }}</span></span>
      </div>
      <div class="rail-item">
        <span class="rail-label">Page</span>
        <span class="rail-value">{{ context.selected_page_number }} / {{ context.queue_entry.page_count }}</span>
      </div>
      <div class="rail-item">
        <span class="rail-label">Page States</span>
        <span class="rail-value">{{ page_status_counts.get("reviewable", 0) }} reviewable · {{ page_status_counts.get("skipped_empty", 0) }} empty · {{ page_status_counts.get("needs_retry", 0) }} retry · {{ page_status_counts.get("failed", 0) }} failed</span>
      </div>
      <div class="rail-item">
        <span class="rail-label">Workspace</span>
        <span class="rail-value workspace-summary">{{ page_workspace.mode }} - {{ page_workspace.fragments|length }} fields - {{ page_workspace.handwritten_count }} handwritten</span>
        <span class="rail-value">{{ page_workspace.mode }} · {{ page_workspace.fragments|length }} fields</span>
      </div>
      <div class="rail-item">
        <span class="rail-label">System Alerts</span>
        <span class="rail-value system-alert-summary">{% if display_warning_count or display_raw_warning_count %}{{ display_warning_count }} current / {{ display_raw_warning_count }} raw{% else %}clear{% endif %}</span>
        <span class="rail-value">{{ context.selected_page.warnings|length }} current · {{ context.selected_raw_page.warnings|length }} raw</span>
      </div>
      <div class="rail-item">
        <span class="rail-label">Preview</span>
        <span class="rail-value">{{ preview_mode }}</span>
      </div>
    </section>

    {% if notices %}
    <section class="notices" aria-live="polite">
      {% for notice in notices %}
      <div class="notice">{{ notice }}</div>
      {% endfor %}
    </section>
    {% endif %}

    <main class="workspace">
      <aside class="panel page-panel">
        <div class="panel-heading">
          <h2>Pages</h2>
          <span class="panel-subtle">{{ context.queue_entry.page_count }} total</span>
        </div>
        <form class="page-jump" method="get" action="/documents/{{ context.queue_entry.document_id }}" data-page-loading>
          <input
            type="number"
            name="page"
            min="1"
            max="{{ context.queue_entry.page_count }}"
            value="{{ context.selected_page_number }}"
            aria-label="Jump to page"
          />
          <input type="hidden" name="preview" value="{{ preview_mode }}" />
          <button type="submit" class="button-secondary">Go</button>
        </form>
        <div class="page-stepper" aria-label="Adjacent pages">
          {% if previous_page_number %}
          <a class="ghost-link" data-page-loading href="/documents/{{ context.queue_entry.document_id }}?page={{ previous_page_number }}&preview={{ preview_mode }}">Previous</a>
          {% else %}
          <span class="page-step-disabled">Previous</span>
          {% endif %}
          {% if next_page_number %}
          <a class="ghost-link" data-page-loading href="/documents/{{ context.queue_entry.document_id }}?page={{ next_page_number }}&preview={{ preview_mode }}">Next</a>
          {% else %}
          <span class="page-step-disabled">Next</span>
          {% endif %}
        </div>
        <nav class="page-nav" aria-label="Page navigation">
          {% for nav_item in page_nav_items %}
          {% if nav_item.kind == "gap" %}
          <div class="page-gap">Pages {{ nav_item.start }}-{{ nav_item.end }}</div>
          {% else %}
          {% set item = nav_item.summary %}
          <a class="page-link {% if item.page_number == context.selected_page_number %}active{% endif %}" data-page-loading href="/documents/{{ context.queue_entry.document_id }}?page={{ item.page_number }}&preview={{ preview_mode }}">
            <div class="page-link-header">
              <strong>Page {{ item.page_number }}</strong>
              <span class="pill page-{{ item.review_status }}">{{ item.review_status }}</span>
            </div>
            <div class="page-link-meta">
              <span class="meta-text">{{ item.fragment_count }} {{ item.fragment_mode }}</span>
              <span class="meta-text">{{ item.processing_status }}</span>
              <span class="meta-text">{{ item.handwritten_count }} handwritten</span>
              <span class="meta-text">{% if item.warning_count %}{{ item.warning_count }} system alerts{% else %}clear{% endif %}</span>
            </div>
          </a>
          {% endif %}
          {% endfor %}
        </nav>
      </aside>

      <section class="panel">
        <div class="image-stack">
          <div>
            <div class="panel-heading">
              <h2>Original Page</h2>
              <span class="panel-subtle page-dimensions">{{ context.page_image_width }} x {{ context.page_image_height }}</span>
              <span class="panel-subtle">{{ context.page_image_width }} × {{ context.page_image_height }}</span>
            </div>
            <div class="image-frame media-frame is-loading" data-media-frame style="--media-aspect: {{ context.page_image_width }} / {{ context.page_image_height }};">
              <div class="media-buffer" aria-hidden="true">
                <span class="loader"></span>
                <span>Loading page image</span>
              </div>
              <img
                data-media-image
                src="/documents/{{ context.queue_entry.document_id }}/assets/page/{{ context.selected_page_number }}"
                alt="Original document page {{ context.selected_page_number }}"
                width="{{ context.page_image_width }}"
                height="{{ context.page_image_height }}"
                fetchpriority="high"
                decoding="async"
              />
            </div>
          </div>

        </div>
      </section>

      <section class="panel">
        <div class="panel-heading">
          <div>
            <h2>Mapped Preview</h2>
            <p class="panel-subtle">Switch between the raw parsed mapping and the current corrected mapping.</p>
          </div>
          <div class="preview-toggle" aria-label="Preview mode">
            <a class="toggle-link {% if preview_mode == 'raw' %}active{% endif %}" data-page-loading href="/documents/{{ context.queue_entry.document_id }}?page={{ context.selected_page_number }}&preview=raw">Raw Mapping</a>
            <a class="toggle-link {% if preview_mode == 'corrected' %}active{% endif %}" data-page-loading href="/documents/{{ context.queue_entry.document_id }}?page={{ context.selected_page_number }}&preview=corrected">Corrected Mapping</a>
          </div>
        </div>
        <div class="preview-frame media-frame is-loading" data-media-frame style="--media-aspect: {{ context.page_image_width }} / {{ context.page_image_height }}; margin-bottom: 16px;">
          <div class="media-buffer" aria-hidden="true">
            <span class="loader"></span>
            <span>Building mapped preview</span>
          </div>
          <img
            data-media-image
            src="/documents/{{ context.queue_entry.document_id }}/pages/{{ context.selected_page_number }}/preview?mode={{ preview_mode }}"
            alt="{{ preview_mode }} mapped preview for page {{ context.selected_page_number }}"
            width="{{ context.page_image_width }}"
            height="{{ context.page_image_height }}"
            fetchpriority="high"
            decoding="async"
          />
        </div>
      </section>

      <section class="panel editor-stage">
        <div class="panel-heading">
          <div>
            <h2>Blocks Workspace</h2>
            <p class="panel-subtle">Review the extracted crop and correct its text in one place, without bouncing back to the main page image.</p>
          </div>
          <div class="bbox-trigger-group">
            <span class="panel-subtle">{{ page_workspace.fragments|length }} fields / {{ page_workspace.crop_count }} mapped crops</span>
            {% if bbox_editor_enabled %}
            <button type="button" class="button-secondary" id="bbox-editor-open">Adjust Bounding Boxes</button>
            {% else %}
            <button type="button" class="button-secondary" id="bbox-editor-open" disabled>Adjust Bounding Boxes</button>
            <span class="panel-subtle">No bbox-editable fragments exist for this page.</span>
            {% endif %}
          </div>
        </div>
        <form
          id="page-commit-form"
          class="editor-form"
          method="post"
          action="/documents/{{ context.queue_entry.document_id }}/pages/{{ context.selected_page_number }}/commit"
          data-dirty-track
        >
          <input type="hidden" id="bbox_payload_json" name="bbox_payload_json" value="{}" />
          {% for fragment in page_workspace.fragments %}
          <section class="fragment-editor" id="{{ fragment.anchor_id }}">
            <div class="fragment-head">
              <div>
                <h3>{{ fragment.label }}</h3>
                <p class="fragment-summary">{{ fragment.source_kind|replace('_', ' ') }}{% if fragment.confidence_notes %} - {{ fragment.confidence_notes }}{% endif %}</p>
                <p>{{ fragment.source_kind|replace('_', ' ') }}{% if fragment.confidence_notes %} · {{ fragment.confidence_notes }}{% endif %}</p>
              </div>
              <span class="panel-subtle">{{ fragment.source_kind|replace('_', ' ') }}</span>
            </div>
            {% if fragment.source_kind == "block" %}
            <div class="fragment-flags">
              <label class="fragment-flag" for="handwritten_{{ loop.index0 }}">
                <input
                  id="handwritten_{{ loop.index0 }}"
                  type="checkbox"
                  name="handwritten_{{ loop.index0 }}"
                  value="true"
                  {% if fragment.handwritten_effective %}checked{% endif %}
                />
                Handwritten
              </label>
              {% if fragment.handwritten_auto %}
              <span class="pill handwriting-auto">auto detected</span>
              {% endif %}
              {% if fragment.handwritten_source == "manual" %}
              <span class="pill handwriting-manual">manual</span>
              {% endif %}
            </div>
            {% endif %}
            {% if fragment.has_bbox %}
            <div class="fragment-crop-frame media-frame is-loading" data-media-frame>
              <div class="media-buffer" aria-hidden="true">
                <span class="loader"></span>
                <span>Loading crop</span>
              </div>
              <img
                data-media-image
                data-lazy-src="/documents/{{ context.queue_entry.document_id }}/pages/{{ context.selected_page_number }}/crops/{{ fragment.fragment_id }}"
                alt="{{ fragment.label }} crop from page {{ context.selected_page_number }}"
                loading="lazy"
                decoding="async"
              />
              <noscript>
                <img
                  src="/documents/{{ context.queue_entry.document_id }}/pages/{{ context.selected_page_number }}/crops/{{ fragment.fragment_id }}"
                  alt="{{ fragment.label }} crop from page {{ context.selected_page_number }}"
                />
              </noscript>
            </div>
            <div class="fragment-crop-meta">
              <span>Reference crop</span>
              {% if fragment.confidence_notes %}
              <span>{{ fragment.confidence_notes }}</span>
              {% endif %}
            </div>
            {% else %}
            <div class="fragment-empty">No crop is available for this fragment. Edit the extracted text directly from the normalized output below.</div>
            {% endif %}
            <label for="{{ fragment.field_name }}">
              {{ fragment.label }} Text
              <textarea id="{{ fragment.field_name }}" data-bidi-lines oninput="adjustDirection(this)" lang="ur" dir="auto" name="{{ fragment.field_name }}" rows="{% if fragment.source_kind == 'full_text' %}12{% else %}5{% endif %}">{{ fragment.text }}</textarea>
            </label>
          </section>
          {% endfor %}

          <details class="data-drawer">
            <summary>Structured Content</summary>
            <p class="drawer-copy">Edit the page-level structured JSON without leaving the workspace.</p>
            <label for="structured_content_json">
              Structured Content JSON
              <textarea id="structured_content_json" name="structured_content_json" rows="14">{{ structured_content_json }}</textarea>
            </label>
          </details>

          <details class="data-drawer">
            <summary>Normalized JSON</summary>
            <p class="drawer-copy">Current corrected page JSON.</p>
            <pre>{{ current_normalized_json }}</pre>
            <p class="drawer-copy">Recovered raw normalized page JSON.</p>
            <pre>{{ raw_normalized_json }}</pre>
          </details>

          <details class="data-drawer">
            <summary>Raw Response</summary>
            <pre>{{ raw_response }}</pre>
          </details>

          <details class="data-drawer">
            <summary>Lines, Tables & System Alerts</summary>
            <pre>{{ diagnostics_json }}</pre>
          </details>
        </form>
      </section>
    </main>

    <script>
      (function () {
        const LRM = "\\u200e";
        const RLM = "\\u200f";
        const lineMarkerPattern = /(^|\\n)[\\u200e\\u200f]/g;
        const urduPattern = /[\\u0600-\\u06ff\\u0750-\\u077f\\u08a0-\\u08ff]/;

        function stripLineDirectionMarkers(value) {
          return value.replace(lineMarkerPattern, "$1");
        }

        function decorateBidiLines(value) {
          return stripLineDirectionMarkers(value)
            .split("\\n")
            .map(function (line) {
              if (line.length === 0) {
                return line;
              }
              return (urduPattern.test(line) ? RLM : LRM) + line;
            })
            .join("\\n");
        }

        function plainIndexFromDecorated(value, decoratedIndex) {
          return stripLineDirectionMarkers(value.slice(0, decoratedIndex)).length;
        }

        function decoratedIndexFromPlain(value, plainIndex) {
          let plainSeen = 0;
          for (let index = 0; index < value.length; index += 1) {
            const character = value.charAt(index);
            const isLineStart = index === 0 || value.charAt(index - 1) === "\\n";
            const isMarker = character === LRM || character === RLM;
            if (isLineStart && isMarker) {
              continue;
            }
            if (plainSeen >= plainIndex) {
              return index;
            }
            plainSeen += 1;
          }
          return value.length;
        }

        window.adjustDirection = function (el) {
          const selectionStart = plainIndexFromDecorated(el.value, el.selectionStart || 0);
          const selectionEnd = plainIndexFromDecorated(el.value, el.selectionEnd || 0);
          const decoratedValue = decorateBidiLines(el.value);

          if (el.value !== decoratedValue) {
            el.value = decoratedValue;
          }

          el.setSelectionRange(
            decoratedIndexFromPlain(decoratedValue, selectionStart),
            decoratedIndexFromPlain(decoratedValue, selectionEnd)
          );
        };

        document.querySelectorAll("textarea[data-bidi-lines]").forEach(function (textarea) {
          window.adjustDirection(textarea);
        });

        document.addEventListener("submit", function () {
          document.querySelectorAll("textarea[data-bidi-lines]").forEach(function (textarea) {
            textarea.value = stripLineDirectionMarkers(textarea.value);
          });
        });
      }());
    </script>

    <script id="bbox-editor-data" type="application/json">{{ bbox_editor_json|safe }}</script>
    <div class="bbox-modal" id="bbox-editor-modal" hidden>
      <div class="bbox-modal-backdrop" data-bbox-close></div>
      <div class="bbox-modal-panel" role="dialog" aria-modal="true" aria-labelledby="bbox-editor-title">
        <div class="bbox-modal-header">
          <div>
            <h2 id="bbox-editor-title">Adjust Bounding Boxes</h2>
            <p>Overlay the real page image with editable boxes, move or resize them until they match the source, then apply the changes to the page save form.</p>
          </div>
          <button type="button" class="button-secondary" id="bbox-editor-close">Close</button>
        </div>
        <div class="bbox-modal-body">
          <section class="bbox-stage">
            <div class="bbox-stage-copy">
              <p>Drag on the image to draw a new box. Drag an existing box to move it, or use the corner handles to resize.</p>
              <span class="pill review-{{ context.corrected.approval_metadata.review_status.value }}">image coordinates</span>
            </div>
            <div class="bbox-canvas-wrap">
              <div class="bbox-canvas" id="bbox-canvas">
                <img
                  id="bbox-editor-image"
                  src="/documents/{{ context.queue_entry.document_id }}/assets/page/{{ context.selected_page_number }}"
                  alt="Bounding box editor image for page {{ context.selected_page_number }}"
                  width="{{ context.page_image_width }}"
                  height="{{ context.page_image_height }}"
                  draggable="false"
                />
                <div class="bbox-overlay" id="bbox-overlay" aria-hidden="true"></div>
              </div>
            </div>
          </section>
          <aside class="bbox-sidebar">
            <label for="bbox-fragment-select">
              Fragment
              <select id="bbox-fragment-select">
                {% for fragment in page_workspace.fragments %}
                <option value="{{ fragment.fragment_id }}">{{ fragment.label }} - {{ fragment.text|replace('\n', ' ')|truncate(64, True) }}</option>
                {% endfor %}
              </select>
            </label>
            <label for="bbox-fragment-preview">
              Text Preview
              <div class="bbox-fragment-preview" id="bbox-fragment-preview"></div>
            </label>
            <div class="bbox-grid">
              <label for="bbox-left">
                Left
                <input type="number" id="bbox-left" inputmode="decimal" step="0.1" />
              </label>
              <label for="bbox-top">
                Top
                <input type="number" id="bbox-top" inputmode="decimal" step="0.1" />
              </label>
              <label for="bbox-right">
                Right
                <input type="number" id="bbox-right" inputmode="decimal" step="0.1" />
              </label>
              <label for="bbox-bottom">
                Bottom
                <input type="number" id="bbox-bottom" inputmode="decimal" step="0.1" />
              </label>
            </div>
            <div class="bbox-actions">
              <button type="button" id="bbox-apply">Apply</button>
              <button type="button" id="bbox-reset-selected" class="button-secondary">Reset Selected</button>
              <button type="button" id="bbox-save" class="bbox-save-button">Save Bounding Boxes</button>
              <button type="button" id="bbox-clear-selected" class="button-danger">Clear Box</button>
            </div>
            <div class="bbox-note-surface" id="bbox-editor-status">
              <strong>Ready</strong>
              Select a fragment, adjust its box, then press Apply or Save Bounding Boxes to persist those geometry edits.
            </div>
          </aside>
        </div>
      </div>
    </div>
    <form id="bbox-save-form" method="post" action="/documents/{{ context.queue_entry.document_id }}/pages/{{ context.selected_page_number }}/bbox-save">
      <input type="hidden" id="bbox_save_payload_json" name="bbox_payload_json" value="{}" />
      <input type="hidden" id="bbox_save_reviewer" name="reviewer" value="" />
    </form>

    <div class="save-bar-sentinel" id="save-bar-sentinel" aria-hidden="true"></div>
    <footer class="save-bar is-collapsed" id="save-bar">
      <div class="save-bar-header">
        <div class="save-bar-heading">
          <h2>Save Verification</h2>
          <p>Collapsed while you review the page, then opens automatically as you reach the save area.</p>
        </div>
        <div class="save-bar-header-meta">
          <span class="pill page-{{ context.page_review.status.value }}">{{ "success" if context.page_review.status.value == "verified" else context.page_review.status.value|replace("_", " ") }}</span>
          <span class="pill review-{{ context.corrected.approval_metadata.review_status.value }}">{{ context.corrected.approval_metadata.review_status.value }}</span>
          <button type="button" class="save-bar-toggle button-secondary" id="save-bar-toggle" aria-controls="save-bar-body" aria-expanded="false">Open Save Verification</button>
        </div>
      </div>

      <div class="save-bar-body" id="save-bar-body">
      <section class="save-primary">
        <div class="save-copy">
          <h2>Save Verification</h2>
          <p>Commit the edited page fields, keep the structured content in sync, and switch the preview to the corrected mapping.</p>
          <div class="review-readiness">
            <span class="pill page-{{ context.page_review.status.value }}">{{ "success" if context.page_review.status.value == "verified" else context.page_review.status.value|replace("_", " ") }}</span>
            <span class="pill review-{{ context.corrected.approval_metadata.review_status.value }}">{{ context.corrected.approval_metadata.review_status.value }}</span>
            <span class="readiness-note">{% if all_pages_verified %}All reviewable pages are verified and no OCR pages are blocked.{% else %}Approve unlocks when every reviewable page is success and retry/failed pages are cleared by a rerun.{% endif %}</span>
          </div>
        </div>
        <div class="settings-grid">
          <label>
            Reviewer
            <input
              type="text"
              name="reviewer"
              value="{{ context.page_review.reviewer or context.corrected.approval_metadata.reviewer or context.corrected.approval_metadata.corrected_by or '' }}"
              form="page-commit-form"
              autocomplete="off"
            />
          </label>
          <label>
            Page Status
            <select name="status" form="page-commit-form">
              {% for option in verification_options %}
              <option value="{{ option }}" {% if option == context.page_review.status.value %}selected{% endif %}>{{ "success" if option == "verified" else option|replace("_", " ") }}</option>
              {% endfor %}
            </select>
          </label>
          <label>
            Page Notes
            <textarea name="comments" class="compact" form="page-commit-form">{{ context.page_review.comments or '' }}</textarea>
          </label>
        </div>
        <div class="save-actions">
          <button type="submit" form="page-commit-form">Save Page Review</button>
          <a class="ghost-link" href="/documents/{{ context.queue_entry.document_id }}?page={{ context.selected_page_number }}&preview=raw">Open Raw Preview</a>
        </div>
      </section>

      <section class="review-panel">
        <form
          id="document-review-form"
          class="document-form"
          method="post"
          action="/documents/{{ context.queue_entry.document_id }}/approve"
          data-dirty-track
        >
          <input type="hidden" name="redirect_page" value="{{ context.selected_page_number }}" />
          <div class="review-panel-header">
            <div>
              <h3>Document Review</h3>
              <p>Use these actions after the page save above. Pipeline status now syncs automatically with the document review decision.</p>
            </div>
            <span class="pill review-{{ context.corrected.approval_metadata.review_status.value }}">{{ context.corrected.approval_metadata.review_status.value }}</span>
          </div>
          <div class="settings-grid">
            <label>
              Reviewer
              <input
                type="text"
                name="reviewer"
                value="{{ context.corrected.approval_metadata.reviewer or context.corrected.approval_metadata.corrected_by or '' }}"
                autocomplete="off"
              />
            </label>
            <label style="grid-column: span 2;">
              Comments
              <textarea name="comments" class="compact">{{ context.corrected.approval_metadata.comments or '' }}</textarea>
            </label>
          </div>
          <div class="decision-grid">
            <button type="submit" name="status" value="approved" {% if not all_pages_verified %}disabled{% endif %}>Approve Document</button>
            <button type="submit" name="status" value="in_review" class="button-secondary">Keep In Review</button>
            <button type="submit" name="status" value="needs_retry" class="button-warn">Needs Retry</button>
            <button type="submit" name="status" value="rejected" class="button-danger">Reject</button>
          </div>
          {% if not all_pages_verified %}
          <p class="drawer-copy">Approval stays locked until every reviewable page is success and retry/failed pages are cleared by a rerun.</p>
          {% endif %}
        </form>
      </section>

      <section class="document-drawers">
        <details class="settings-drawer">
          <summary>Document Text & Target Output</summary>
          <form
            id="document-save-form"
            class="document-form"
            method="post"
            action="/documents/{{ context.queue_entry.document_id }}/document"
            data-dirty-track
          >
            <input type="hidden" name="redirect_page" value="{{ context.selected_page_number }}" />
            <h3>Document Save</h3>
            <label>
              Reviewer
              <input
                type="text"
                name="reviewer"
                value="{{ context.corrected.approval_metadata.corrected_by or context.corrected.approval_metadata.reviewer or '' }}"
                autocomplete="off"
              />
            </label>
            <label>
              Consolidated Text
              <textarea name="consolidated_text">{{ context.corrected.consolidated_text }}</textarea>
            </label>
            <label>
              Target Output JSON
              <textarea name="target_output_json">{{ target_output_json }}</textarea>
            </label>
            <label>
              Notes
              <textarea name="comments" class="compact">{{ context.corrected.approval_metadata.comments or '' }}</textarea>
            </label>
            <button type="submit">Save Document Edits</button>
          </form>
        </details>

        <details class="settings-drawer">
          <summary>Workflow Reference</summary>
          <p class="drawer-copy">Current save timeline and command shortcuts.</p>
          <pre>{{ review_timeline_json }}</pre>
          <pre>{% for label, command in context.command_hints.items() %}{{ label }}: {{ command }}
{% endfor %}</pre>
        </details>
      </section>
      </div>
    </footer>
  </div>

  <div class="page-transition" id="page-transition" hidden aria-live="polite" aria-label="Loading page">
    <div class="page-transition-card">
      <span class="loader"></span>
      <span>Loading selected page</span>
    </div>
  </div>

  <script>
    (function () {
      let dirty = false;
      const saveBar = document.getElementById("save-bar");
      const saveBarBody = document.getElementById("save-bar-body");
      const saveBarToggle = document.getElementById("save-bar-toggle");
      const saveBarSentinel = document.getElementById("save-bar-sentinel");
      const pageTransition = document.getElementById("page-transition");
      const bboxPayloadInput = document.getElementById("bbox_payload_json");
      const bboxEditorDataNode = document.getElementById("bbox-editor-data");
      const bboxEditorModal = document.getElementById("bbox-editor-modal");
      const bboxEditorOpen = document.getElementById("bbox-editor-open");
      const bboxEditorClose = document.getElementById("bbox-editor-close");
      const bboxOverlay = document.getElementById("bbox-overlay");
      const bboxImage = document.getElementById("bbox-editor-image");
      const bboxFragmentSelect = document.getElementById("bbox-fragment-select");
      const bboxFragmentPreview = document.getElementById("bbox-fragment-preview");
      const bboxStatus = document.getElementById("bbox-editor-status");
      const bboxApply = document.getElementById("bbox-apply");
      const bboxSave = document.getElementById("bbox-save");
      const bboxResetSelected = document.getElementById("bbox-reset-selected");
      const bboxClearSelected = document.getElementById("bbox-clear-selected");
      const bboxSaveForm = document.getElementById("bbox-save-form");
      const bboxSavePayloadInput = document.getElementById("bbox_save_payload_json");
      const bboxSaveReviewerInput = document.getElementById("bbox_save_reviewer");
      const pageCommitReviewerInput = document.querySelector('input[name="reviewer"][form="page-commit-form"]');
      const bboxInputs = {
        left: document.getElementById("bbox-left"),
        top: document.getElementById("bbox-top"),
        right: document.getElementById("bbox-right"),
        bottom: document.getElementById("bbox-bottom"),
      };
      let bboxState = null;

      function showPageTransition() {
        if (pageTransition) {
          pageTransition.hidden = false;
        }
      }

      function markMediaLoaded(image) {
        const frame = image.closest("[data-media-frame]");
        if (!frame) {
          return;
        }
        frame.classList.remove("is-loading");
        frame.classList.add("is-loaded");
      }

      function prepareMediaImage(image) {
        if (!image) {
          return;
        }
        if (image.complete && (image.currentSrc || image.getAttribute("src"))) {
          markMediaLoaded(image);
          return;
        }
        image.addEventListener("load", function () {
          markMediaLoaded(image);
        }, { once: true });
        image.addEventListener("error", function () {
          markMediaLoaded(image);
        }, { once: true });
      }

      function activateLazyImage(image) {
        const lazySrc = image.dataset.lazySrc;
        if (!lazySrc) {
          return;
        }
        prepareMediaImage(image);
        image.src = lazySrc;
        image.removeAttribute("data-lazy-src");
      }

      document.querySelectorAll("img[data-media-image]:not([data-lazy-src])").forEach(prepareMediaImage);
      const lazyImages = Array.from(document.querySelectorAll("img[data-lazy-src]"));
      if ("IntersectionObserver" in window) {
        const mediaObserver = new IntersectionObserver(
          function (entries, observer) {
            entries.forEach(function (entry) {
              if (!entry.isIntersecting) {
                return;
              }
              activateLazyImage(entry.target);
              observer.unobserve(entry.target);
            });
          },
          { rootMargin: "360px 0px" }
        );
        lazyImages.forEach(function (image) {
          mediaObserver.observe(image);
        });
      } else {
        lazyImages.forEach(activateLazyImage);
      }

      document.addEventListener("click", function (event) {
        const link = event.target.closest("a[data-page-loading]");
        if (!link || event.defaultPrevented || dirty || link.href === window.location.href) {
          return;
        }
        showPageTransition();
      });

      document.addEventListener("submit", function (event) {
        if (event.target && event.target.matches("form[data-page-loading]")) {
          showPageTransition();
        }
      });

      const pageNav = document.querySelector(".page-nav");
      const activePageLink = pageNav ? pageNav.querySelector(".page-link.active") : null;
      if (pageNav && activePageLink) {
        const navRect = pageNav.getBoundingClientRect();
        const activeRect = activePageLink.getBoundingClientRect();
        const activeTop = (activeRect.top - navRect.top) + pageNav.scrollTop;
        const compactNav = window.matchMedia && window.matchMedia("(max-width: 760px)").matches;
        pageNav.scrollTop = Math.max(
          compactNav ? activeTop - 8 : activeTop - (pageNav.clientHeight / 2) + (activeRect.height / 2),
          0
        );
      }

      function setSaveBarExpanded(expanded) {
        if (!saveBar || !saveBarToggle) {
          return;
        }
        saveBar.classList.toggle("is-collapsed", !expanded);
        saveBarToggle.setAttribute("aria-expanded", expanded ? "true" : "false");
        saveBarToggle.textContent = expanded ? "Collapse Save Verification" : "Open Save Verification";
      }

      function markDirty() {
        dirty = true;
      }

      if (saveBar && saveBarBody && saveBarToggle && saveBarSentinel) {
        setSaveBarExpanded(false);

        const observer = new IntersectionObserver(
          function (entries) {
            const entry = entries[0];
            setSaveBarExpanded(Boolean(entry && entry.isIntersecting));
          },
          {
            rootMargin: "0px 0px -18% 0px",
            threshold: 0,
          }
        );
        observer.observe(saveBarSentinel);

        saveBarToggle.addEventListener("click", function () {
          setSaveBarExpanded(saveBar.classList.contains("is-collapsed"));
        });

        saveBar.addEventListener("focusin", function () {
          setSaveBarExpanded(true);
        });

        document.addEventListener("focusin", function (event) {
          if (!saveBar.contains(event.target)) {
            const sentinelVisible = saveBarSentinel.getBoundingClientRect().top <= window.innerHeight * 0.82;
            if (!sentinelVisible) {
              setSaveBarExpanded(false);
            }
          }
        });
      }

      document.addEventListener("input", function (event) {
        if (event.target.form && event.target.form.matches("[data-dirty-track]")) {
          markDirty();
        }
      });
      document.addEventListener("change", function (event) {
        if (event.target.form && event.target.form.matches("[data-dirty-track]")) {
          markDirty();
        }
      });
      document.querySelectorAll("form[data-dirty-track]").forEach(function (form) {
        form.addEventListener("submit", function () {
          dirty = false;
        });
      });
      window.addEventListener("beforeunload", function (event) {
        if (!dirty) {
          return;
        }
        event.preventDefault();
        event.returnValue = "";
      });

      function cloneBBox(bbox) {
        if (!bbox) {
          return null;
        }
        return {
          left: Number(bbox.left),
          top: Number(bbox.top),
          right: Number(bbox.right),
          bottom: Number(bbox.bottom),
        };
      }

      function bboxEqual(first, second) {
        if (!first || !second) {
          return first === second;
        }
        return first.left === second.left
          && first.top === second.top
          && first.right === second.right
          && first.bottom === second.bottom;
      }

      function clamp(value, minimum, maximum) {
        return Math.max(minimum, Math.min(maximum, value));
      }

      function normalizeBBox(bbox) {
        if (!bbox || !bboxState) {
          return null;
        }
        const left = clamp(Number(bbox.left), 0, bboxState.imageWidth);
        const right = clamp(Number(bbox.right), 0, bboxState.imageWidth);
        const top = clamp(Number(bbox.top), 0, bboxState.imageHeight);
        const bottom = clamp(Number(bbox.bottom), 0, bboxState.imageHeight);
        return {
          left: Math.min(left, right),
          top: Math.min(top, bottom),
          right: Math.max(left, right),
          bottom: Math.max(top, bottom),
        };
      }

      function roundedBBox(bbox) {
        if (!bbox) {
          return null;
        }
        return {
          left: Math.round(bbox.left * 100) / 100,
          top: Math.round(bbox.top * 100) / 100,
          right: Math.round(bbox.right * 100) / 100,
          bottom: Math.round(bbox.bottom * 100) / 100,
        };
      }

      function getSelectedFragment() {
        if (!bboxState || !bboxState.selectedId) {
          return null;
        }
        return bboxState.fragments.find(function (fragment) {
          return fragment.fragmentId === bboxState.selectedId;
        }) || null;
      }

      function getDraftBBox(fragmentId) {
        return bboxState ? bboxState.draft[fragmentId] || null : null;
      }

      function setDraftBBox(fragmentId, bbox) {
        if (!bboxState) {
          return;
        }
        bboxState.draft[fragmentId] = bbox ? normalizeBBox(bbox) : null;
      }

      function hasPendingBBoxChanges() {
        if (!bboxState) {
          return false;
        }
        return bboxState.fragments.some(function (fragment) {
          return !bboxEqual(getDraftBBox(fragment.fragmentId), bboxState.applied[fragment.fragmentId] || null);
        });
      }

      function pointToNatural(clientX, clientY) {
        if (!bboxState || !bboxImage) {
          return { x: 0, y: 0 };
        }
        const rect = bboxImage.getBoundingClientRect();
        return {
          x: clamp(((clientX - rect.left) / Math.max(rect.width, 1)) * bboxState.imageWidth, 0, bboxState.imageWidth),
          y: clamp(((clientY - rect.top) / Math.max(rect.height, 1)) * bboxState.imageHeight, 0, bboxState.imageHeight),
        };
      }

      function setBBoxStatus(title, body) {
        if (!bboxStatus) {
          return;
        }
        bboxStatus.innerHTML = "<strong>" + title + "</strong>" + body;
      }

      function syncHiddenBBoxPayload() {
        if (!bboxState || !bboxPayloadInput) {
          return;
        }
        const payload = {};
        bboxState.fragments.forEach(function (fragment) {
          const original = bboxState.original[fragment.fragmentId] || null;
          const applied = bboxState.applied[fragment.fragmentId] || null;
          if (!bboxEqual(original, applied)) {
            payload[fragment.fragmentId] = applied ? roundedBBox(applied) : null;
          }
        });
        bboxPayloadInput.value = JSON.stringify(payload);
        bboxPayloadInput.dispatchEvent(new Event("input", { bubbles: true }));
        markDirty();
      }

      function syncSelectedInputs() {
        const fragment = getSelectedFragment();
        const bbox = fragment ? getDraftBBox(fragment.fragmentId) : null;
        ["left", "top", "right", "bottom"].forEach(function (field) {
          if (!bboxInputs[field]) {
            return;
          }
          bboxInputs[field].value = bbox ? String(Math.round(bbox[field] * 100) / 100) : "";
        });
      }

      function syncSelectedPreview() {
        const fragment = getSelectedFragment();
        if (!fragment || !bboxFragmentPreview) {
          return;
        }
        bboxFragmentPreview.textContent = fragment.text || "No extracted text available for this fragment.";
      }

      function updateBBoxStatus() {
        const fragment = getSelectedFragment();
        if (!fragment) {
          setBBoxStatus("Ready", "Select a fragment to begin.");
          return;
        }
        if (hasPendingBBoxChanges()) {
          setBBoxStatus("Pending Modal Changes", "Press Apply to attach the current geometry edits to the page save form.");
          return;
        }
        const appliedBBox = bboxState.applied[fragment.fragmentId] || null;
        if (appliedBBox) {
          setBBoxStatus("Applied", "Bounding box changes are attached to the page save form. Finish with Save Page Review to persist them.");
          return;
        }
        setBBoxStatus("Ready", "Drag on the image to draw a new box, or enter exact coordinates for the selected fragment.");
      }

      function renderOverlay() {
        if (!bboxState || !bboxOverlay || !bboxImage) {
          return;
        }
        const rect = bboxImage.getBoundingClientRect();
        bboxOverlay.innerHTML = "";
        bboxOverlay.style.width = rect.width + "px";
        bboxOverlay.style.height = rect.height + "px";
        bboxState.fragments.forEach(function (fragment) {
          const bbox = getDraftBBox(fragment.fragmentId);
          if (!bbox) {
            return;
          }
          const box = document.createElement("div");
          box.className = "bbox-box" + (fragment.fragmentId === bboxState.selectedId ? " is-selected" : "");
          box.dataset.fragmentId = fragment.fragmentId;
          box.style.left = (bbox.left / Math.max(bboxState.imageWidth, 1)) * rect.width + "px";
          box.style.top = (bbox.top / Math.max(bboxState.imageHeight, 1)) * rect.height + "px";
          box.style.width = Math.max(((bbox.right - bbox.left) / Math.max(bboxState.imageWidth, 1)) * rect.width, 1) + "px";
          box.style.height = Math.max(((bbox.bottom - bbox.top) / Math.max(bboxState.imageHeight, 1)) * rect.height, 1) + "px";

          const label = document.createElement("div");
          label.className = "bbox-box-label";
          label.textContent = fragment.label;
          box.appendChild(label);

          ["nw", "ne", "sw", "se"].forEach(function (handleName) {
            const handle = document.createElement("div");
            handle.className = "bbox-handle";
            handle.dataset.handle = handleName;
            handle.dataset.fragmentId = fragment.fragmentId;
            box.appendChild(handle);
          });

          bboxOverlay.appendChild(box);
        });
        syncSelectedInputs();
        syncSelectedPreview();
        updateBBoxStatus();
      }

      function openBBoxEditor() {
        if (!bboxState || !bboxState.enabled || !bboxEditorModal) {
          return;
        }
        bboxEditorModal.hidden = false;
        document.body.classList.add("modal-open");
        renderOverlay();
      }

      function closeBBoxEditor() {
        if (!bboxEditorModal) {
          return;
        }
        bboxEditorModal.hidden = true;
        document.body.classList.remove("modal-open");
      }

      function applyBBoxDraft() {
        if (!bboxState) {
          return;
        }
        bboxState.fragments.forEach(function (fragment) {
          bboxState.applied[fragment.fragmentId] = cloneBBox(getDraftBBox(fragment.fragmentId));
        });
        syncHiddenBBoxPayload();
        updateBBoxStatus();
      }

      function persistBBoxEdits() {
        if (!bboxSaveForm || !bboxSavePayloadInput || !bboxSaveReviewerInput) {
          return;
        }
        applyBBoxDraft();
        bboxSavePayloadInput.value = bboxPayloadInput ? bboxPayloadInput.value : "{}";
        bboxSaveReviewerInput.value = pageCommitReviewerInput ? pageCommitReviewerInput.value : "";
        dirty = false;
        bboxSaveForm.submit();
      }

      function resetSelectedBBox() {
        const fragment = getSelectedFragment();
        if (!fragment) {
          return;
        }
        setDraftBBox(fragment.fragmentId, cloneBBox(bboxState.original[fragment.fragmentId]));
        renderOverlay();
      }

      function clearSelectedBBox() {
        const fragment = getSelectedFragment();
        if (!fragment) {
          return;
        }
        setDraftBBox(fragment.fragmentId, null);
        renderOverlay();
      }

      function updateSelectedBBoxFromInputs() {
        const fragment = getSelectedFragment();
        if (!fragment) {
          return;
        }
        const rawValues = {
          left: bboxInputs.left ? bboxInputs.left.value.trim() : "",
          top: bboxInputs.top ? bboxInputs.top.value.trim() : "",
          right: bboxInputs.right ? bboxInputs.right.value.trim() : "",
          bottom: bboxInputs.bottom ? bboxInputs.bottom.value.trim() : "",
        };
        if (!rawValues.left && !rawValues.top && !rawValues.right && !rawValues.bottom) {
          setDraftBBox(fragment.fragmentId, null);
          renderOverlay();
          return;
        }
        if (!rawValues.left || !rawValues.top || !rawValues.right || !rawValues.bottom) {
          return;
        }
        setDraftBBox(fragment.fragmentId, {
          left: Number(rawValues.left),
          top: Number(rawValues.top),
          right: Number(rawValues.right),
          bottom: Number(rawValues.bottom),
        });
        renderOverlay();
      }

      function beginBBoxInteraction(event) {
        if (!bboxState || !bboxState.enabled || !bboxOverlay) {
          return;
        }
        const point = pointToNatural(event.clientX, event.clientY);
        const handle = event.target.closest("[data-handle]");
        const box = event.target.closest("[data-fragment-id]");
        if (handle && box) {
          bboxState.selectedId = box.dataset.fragmentId;
          bboxFragmentSelect.value = bboxState.selectedId;
          bboxState.pointer = {
            type: "resize",
            fragmentId: bboxState.selectedId,
            handle: handle.dataset.handle,
            startPoint: point,
            startBBox: cloneBBox(getDraftBBox(bboxState.selectedId)),
          };
          bboxOverlay.setPointerCapture(event.pointerId);
          renderOverlay();
          event.preventDefault();
          return;
        }
        if (box) {
          bboxState.selectedId = box.dataset.fragmentId;
          bboxFragmentSelect.value = bboxState.selectedId;
          bboxState.pointer = {
            type: "move",
            fragmentId: bboxState.selectedId,
            startPoint: point,
            startBBox: cloneBBox(getDraftBBox(bboxState.selectedId)),
          };
          bboxOverlay.setPointerCapture(event.pointerId);
          renderOverlay();
          event.preventDefault();
          return;
        }
        if (!bboxState.selectedId) {
          return;
        }
        bboxState.pointer = {
          type: "draw",
          fragmentId: bboxState.selectedId,
          startPoint: point,
          startBBox: cloneBBox(getDraftBBox(bboxState.selectedId)),
        };
        setDraftBBox(bboxState.selectedId, {
          left: point.x,
          top: point.y,
          right: point.x,
          bottom: point.y,
        });
        bboxOverlay.setPointerCapture(event.pointerId);
        renderOverlay();
        event.preventDefault();
      }

      function updateBBoxInteraction(event) {
        if (!bboxState || !bboxState.pointer) {
          return;
        }
        const pointer = bboxState.pointer;
        const point = pointToNatural(event.clientX, event.clientY);
        const startBBox = pointer.startBBox || {
          left: pointer.startPoint.x,
          top: pointer.startPoint.y,
          right: pointer.startPoint.x,
          bottom: pointer.startPoint.y,
        };
        if (pointer.type === "draw") {
          setDraftBBox(pointer.fragmentId, {
            left: pointer.startPoint.x,
            top: pointer.startPoint.y,
            right: point.x,
            bottom: point.y,
          });
        } else if (pointer.type === "move") {
          const width = startBBox.right - startBBox.left;
          const height = startBBox.bottom - startBBox.top;
          const nextLeft = clamp(startBBox.left + (point.x - pointer.startPoint.x), 0, bboxState.imageWidth - width);
          const nextTop = clamp(startBBox.top + (point.y - pointer.startPoint.y), 0, bboxState.imageHeight - height);
          setDraftBBox(pointer.fragmentId, {
            left: nextLeft,
            top: nextTop,
            right: nextLeft + width,
            bottom: nextTop + height,
          });
        } else if (pointer.type === "resize") {
          const nextBBox = cloneBBox(startBBox);
          if (pointer.handle.indexOf("n") !== -1) {
            nextBBox.top = point.y;
          }
          if (pointer.handle.indexOf("s") !== -1) {
            nextBBox.bottom = point.y;
          }
          if (pointer.handle.indexOf("w") !== -1) {
            nextBBox.left = point.x;
          }
          if (pointer.handle.indexOf("e") !== -1) {
            nextBBox.right = point.x;
          }
          setDraftBBox(pointer.fragmentId, nextBBox);
        }
        renderOverlay();
        event.preventDefault();
      }

      function endBBoxInteraction(event) {
        if (!bboxState || !bboxState.pointer || !bboxOverlay) {
          return;
        }
        if (bboxOverlay.hasPointerCapture(event.pointerId)) {
          bboxOverlay.releasePointerCapture(event.pointerId);
        }
        bboxState.pointer = null;
        renderOverlay();
      }

      if (bboxEditorDataNode && bboxPayloadInput && bboxEditorOpen && bboxEditorModal && bboxOverlay && bboxImage && bboxFragmentSelect) {
        try {
          const editorData = JSON.parse(bboxEditorDataNode.textContent || "{}");
          if (editorData && Array.isArray(editorData.fragments)) {
            bboxState = {
              enabled: Boolean(editorData.enabled),
              imageWidth: Number(editorData.width || 0),
              imageHeight: Number(editorData.height || 0),
              fragments: editorData.fragments,
              selectedId: editorData.fragments.length ? editorData.fragments[0].fragmentId : "",
              original: {},
              applied: {},
              draft: {},
              pointer: null,
            };
            editorData.fragments.forEach(function (fragment) {
              bboxState.original[fragment.fragmentId] = cloneBBox(fragment.bbox);
              bboxState.applied[fragment.fragmentId] = cloneBBox(fragment.bbox);
              bboxState.draft[fragment.fragmentId] = cloneBBox(fragment.bbox);
            });
            bboxFragmentSelect.value = bboxState.selectedId;
            renderOverlay();

            bboxEditorOpen.addEventListener("click", openBBoxEditor);
            bboxEditorClose.addEventListener("click", closeBBoxEditor);
            bboxApply.addEventListener("click", applyBBoxDraft);
            if (bboxSave) {
              bboxSave.addEventListener("click", persistBBoxEdits);
            }
            bboxResetSelected.addEventListener("click", resetSelectedBBox);
            bboxClearSelected.addEventListener("click", clearSelectedBBox);
            bboxFragmentSelect.addEventListener("change", function () {
              bboxState.selectedId = bboxFragmentSelect.value;
              renderOverlay();
            });
            Object.keys(bboxInputs).forEach(function (field) {
              if (!bboxInputs[field]) {
                return;
              }
              bboxInputs[field].addEventListener("change", updateSelectedBBoxFromInputs);
            });
            bboxOverlay.addEventListener("pointerdown", beginBBoxInteraction);
            bboxOverlay.addEventListener("pointermove", updateBBoxInteraction);
            bboxOverlay.addEventListener("pointerup", endBBoxInteraction);
            bboxOverlay.addEventListener("pointercancel", endBBoxInteraction);
            window.addEventListener("resize", renderOverlay);
            if (bboxImage.complete) {
              renderOverlay();
            } else {
              bboxImage.addEventListener("load", renderOverlay);
            }
            bboxEditorModal.addEventListener("click", function (event) {
              if (event.target && event.target.hasAttribute("data-bbox-close")) {
                closeBBoxEditor();
              }
            });
            document.addEventListener("keydown", function (event) {
              if (event.key === "Escape" && !bboxEditorModal.hidden) {
                closeBBoxEditor();
              }
            });
          }
        } catch (error) {
          setBBoxStatus("Unavailable", "Bounding box editor data could not be loaded on this page.");
        }
      }
    }());
  </script>
</body>
</html>
"""
)


@dataclass
class WebResponse:
    status_code: int
    body: bytes
    content_type: str = "text/html; charset=utf-8"
    headers: dict[str, str] | None = None


class InteractiveWebController:
    def __init__(self, service: InteractiveReviewService, host: str, port: int) -> None:
        self.service = service
        self.host = host
        self.port = port

    def handle_request(self, method: str, raw_path: str, body: bytes | None = None) -> WebResponse:
        parsed = urlparse(raw_path)
        path = parsed.path
        query = parse_qs(parsed.query)
        form = self._parse_form(body) if method == "POST" and body is not None else {}

        try:
            if method == "GET" and path == "/":
                return self._render_dashboard(query)
            if method == "GET" and path.startswith("/documents/") and "/assets/page/" in path:
                return self._serve_page_asset(path)
            if method == "GET" and path.startswith("/documents/") and path.endswith("/preview"):
                return self._serve_page_preview(path, query)
            if method == "GET" and path.startswith("/documents/") and "/crops/" in path:
                return self._serve_fragment_crop(path)
            if method == "GET" and path.startswith("/documents/"):
                return self._render_document(path, query)
            if method == "GET" and path.startswith("/reports/"):
                return self._serve_report(path)
            if method == "GET" and path == "/status/run":
                return self._json_response(self.service.load_latest_run_summary() or {})
            if method == "GET" and path == "/status/export":
                return self._json_response(self.service.load_latest_export_summary() or {})
            if method == "POST" and path.startswith("/documents/") and path.endswith("/bbox-save"):
                return self._handle_bbox_save(path, form)
            if method == "POST" and path.startswith("/documents/") and path.endswith("/commit"):
                return self._handle_page_commit(path, form)
            if method == "POST" and path.startswith("/documents/") and path.endswith("/verify"):
                return self._handle_verify(path, form)
            if method == "POST" and path.startswith("/documents/") and path.endswith("/document"):
                return self._handle_document_save(path, form)
            if method == "POST" and path.startswith("/documents/") and path.endswith("/approve"):
                return self._handle_approve(path, form)
            if method == "POST" and path.startswith("/documents/"):
                return self._handle_page_save(path, form)
        except FileNotFoundError as exc:
            return self._message_response(str(exc), status_code=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            return self._message_response(str(exc), status_code=HTTPStatus.BAD_REQUEST)

        return self._message_response("Not found.", status_code=HTTPStatus.NOT_FOUND)

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def document_url(self, document_id: str | None = None) -> str:
        if document_id:
            return f"{self.base_url()}/documents/{quote(document_id)}"
        return self.base_url()

    def _render_dashboard(self, query: dict[str, list[str]]) -> WebResponse:
        pipeline_status = query.get("pipeline_status", ["all"])[0]
        review_status = query.get("review_status", ["all"])[0]
        include_dry_run = query.get("include_dry_run", ["true"])[0].lower() != "false"
        state = self.service.get_dashboard_state(
            pipeline_status=pipeline_status,
            review_status=review_status,
            include_dry_run=include_dry_run,
        )
        html = DASHBOARD_TEMPLATE.render(
            pipeline_counts=state["pipeline_counts"],
            review_counts=state["review_counts"],
            latest_run=serialize_summary(state["latest_run"]),
            latest_export=serialize_summary(state["latest_export"]),
            documents=state["documents"],
            command_hints=state["command_hints"],
            selected_pipeline_status=pipeline_status,
            selected_review_status=review_status,
            include_dry_run=include_dry_run,
            pipeline_options=[
                "all",
                "discovered",
                "normalized",
                "extracted",
                "review_pending",
                "in_review",
                "needs_retry",
                "failed",
                "approved",
                "rejected",
                "skipped",
            ],
            review_options=["all", "pending", "in_review", "approved", "rejected", "needs_retry"],
        )
        return WebResponse(status_code=HTTPStatus.OK, body=html.encode("utf-8"))

    def _render_document(self, path: str, query: dict[str, list[str]]) -> WebResponse:
        document_id = unquote(path.split("/")[2])
        page_number = int(query.get("page", ["1"])[0])
        preview_mode = query.get("preview", ["corrected"])[0].lower()
        if preview_mode not in {"raw", "corrected"}:
            preview_mode = "corrected"
        message = query.get("message", [""])[0]
        context = self.service.get_document_context(document_id, page_number=page_number)
        notices = [item for item in [message, context.auto_heal_note] if item]
        display_warnings = self.service.filter_display_warnings(context.selected_page.warnings)
        display_raw_warnings = self.service.filter_display_warnings(context.selected_raw_page.warnings)
        all_pages_verified = self.service.all_pages_verified(context.corrected)
        raw_response = ""
        if context.selected_raw_record is not None:
            raw_response = (
                context.selected_raw_record.raw_response_text
                if context.selected_raw_record.raw_response_text
                else json.dumps(context.selected_raw_record.model_dump(mode="json"), ensure_ascii=False, indent=2)
            )

        diagnostics = {
            "warnings": display_warnings,
            "confidence_notes": context.selected_page.confidence_notes,
            "lines": [item.model_dump(mode="json") for item in context.selected_page.lines],
            "tables": [item.model_dump(mode="json") for item in context.selected_page.tables],
            "raw_preview_warnings": display_raw_warnings,
            "handwritten_count": context.selected_page_workspace.handwritten_count,
        }
        bbox_editor_enabled = context.selected_page_workspace.mode != "full_text"
        bbox_editor_fragments = [
            {
                "fragmentId": fragment.fragment_id,
                "label": fragment.label,
                "text": fragment.text,
                "sourceKind": fragment.source_kind,
                "confidenceNotes": fragment.confidence_notes or "",
                "handwrittenAuto": fragment.handwritten_auto,
                "handwrittenManual": fragment.handwritten_manual,
                "handwrittenEffective": fragment.handwritten_effective,
                "handwrittenSource": fragment.handwritten_source,
                "bbox": _bbox_payload_from_fragment(fragment),
            }
            for fragment in context.selected_page_workspace.fragments
        ]
        bbox_editor_data = {
            "documentId": context.queue_entry.document_id,
            "pageNumber": context.selected_page_number,
            "imageUrl": f"/documents/{quote(context.queue_entry.document_id)}/assets/page/{context.selected_page_number}",
            "width": context.page_image_width,
            "height": context.page_image_height,
            "mode": context.selected_page_workspace.mode,
            "enabled": bbox_editor_enabled,
            "fragments": bbox_editor_fragments,
        }
        review_timeline = {
            "pipeline_status": context.queue_entry.pipeline_status,
            "document_review_status": context.corrected.approval_metadata.review_status.value,
            "page_review_status": context.page_review.status.value,
            "page_reviewer": context.page_review.reviewer,
            "page_reviewed_at": serialize_timestamp(context.page_review.reviewed_at),
            "document_reviewer": context.corrected.approval_metadata.reviewer,
            "corrected_by": context.corrected.approval_metadata.corrected_by,
            "corrected_at": serialize_timestamp(context.corrected.approval_metadata.corrected_at),
            "document_reviewed_at": serialize_timestamp(context.corrected.approval_metadata.reviewed_at),
        }
        previous_page_number = (
            context.page_summaries[context.selected_page_index - 1].page_number
            if context.selected_page_index > 0
            else None
        )
        next_page_number = (
            context.page_summaries[context.selected_page_index + 1].page_number
            if context.selected_page_index + 1 < len(context.page_summaries)
            else None
        )
        html = DOCUMENT_TEMPLATE.render(
            context=context,
            page_summaries=context.page_summaries,
            page_nav_items=_page_nav_items(context.page_summaries, context.selected_page_number),
            previous_page_number=previous_page_number,
            next_page_number=next_page_number,
            page_status_counts=context.page_status_counts,
            page_workspace=context.selected_page_workspace,
            preview_mode=preview_mode,
            notices=notices,
            all_pages_verified=all_pages_verified,
            display_warning_count=len(display_warnings),
            display_raw_warning_count=len(display_raw_warnings),
            structured_content_json=json.dumps(context.selected_page.structured_content, ensure_ascii=False, indent=2, sort_keys=True),
            current_normalized_json=json.dumps(context.selected_page.model_dump(mode="json"), ensure_ascii=False, indent=2),
            raw_normalized_json=json.dumps(context.selected_raw_page.model_dump(mode="json"), ensure_ascii=False, indent=2),
            diagnostics_json=json.dumps(diagnostics, ensure_ascii=False, indent=2),
            bbox_editor_enabled=bbox_editor_enabled,
            bbox_editor_json=_serialize_script_data(bbox_editor_data),
            review_timeline_json=json.dumps(review_timeline, ensure_ascii=False, indent=2),
            target_output_json=serialize_target_output(context.corrected),
            corrected_at=serialize_timestamp(context.corrected.approval_metadata.corrected_at),
            reviewed_at=serialize_timestamp(context.corrected.approval_metadata.reviewed_at),
            raw_response=raw_response or json.dumps({"message": "No raw provider response stored for this page."}, ensure_ascii=False, indent=2),
            verification_options=[item.value for item in PageVerificationStatus],
            review_options=[
                ReviewStatus.APPROVED.value,
                ReviewStatus.REJECTED.value,
                ReviewStatus.IN_REVIEW.value,
                ReviewStatus.PENDING.value,
                ReviewStatus.NEEDS_RETRY.value,
            ],
        )
        return WebResponse(status_code=HTTPStatus.OK, body=html.encode("utf-8"))

    def _serve_page_asset(self, path: str) -> WebResponse:
        parts = path.strip("/").split("/")
        document_id = unquote(parts[1])
        page_number = int(parts[-1])
        image_path = Path(self.service.page_image_path(document_id, page_number))
        if not image_path.exists():
            raise FileNotFoundError(f"Rendered page image not found: {image_path}")
        mime_type, _ = mimetypes.guess_type(str(image_path))
        return WebResponse(
            status_code=HTTPStatus.OK,
            body=image_path.read_bytes(),
            content_type=mime_type or "application/octet-stream",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    def _serve_page_preview(self, path: str, query: dict[str, list[str]]) -> WebResponse:
        parts = path.strip("/").split("/")
        document_id = unquote(parts[1])
        page_number = int(parts[3])
        mode = query.get("mode", ["corrected"])[0].lower()
        svg = self.service.build_page_preview_svg(document_id, page_number, mode=mode)
        return WebResponse(
            status_code=HTTPStatus.OK,
            body=svg.encode("utf-8"),
            content_type="image/svg+xml; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    def _serve_fragment_crop(self, path: str) -> WebResponse:
        parts = path.strip("/").split("/")
        document_id = unquote(parts[1])
        page_number = int(parts[3])
        fragment_id = unquote(parts[-1])
        crop = self.service.load_fragment_crop(document_id, page_number, fragment_id)
        return WebResponse(status_code=HTTPStatus.OK, body=crop, content_type="image/png", headers={"Cache-Control": "no-store"})

    def _serve_report(self, path: str) -> WebResponse:
        document_id = unquote(path.split("/")[2])
        report_path = self.service.data_paths.reports_dir / document_id / "report.html"
        if not report_path.exists():
            raise FileNotFoundError(f"Static report not found for document: {document_id}")
        return WebResponse(status_code=HTTPStatus.OK, body=report_path.read_bytes())

    def _handle_page_save(self, path: str, form: dict[str, str]) -> WebResponse:
        parts = path.strip("/").split("/")
        document_id = unquote(parts[1])
        page_number = int(parts[3])
        self.service.save_page_edits(
            document_id,
            page_number,
            full_text=form.get("full_text", ""),
            structured_content_text=form.get("structured_content_json", "{}"),
            reviewer=form.get("reviewer", ""),
            bbox_payload_text=form.get("bbox_payload_json", ""),
        )
        return self._redirect(
            f"/documents/{quote(document_id)}?page={page_number}&preview=corrected&message={quote('Page edits saved.')}"
        )

    def _handle_bbox_save(self, path: str, form: dict[str, str]) -> WebResponse:
        parts = path.strip("/").split("/")
        document_id = unquote(parts[1])
        page_number = int(parts[3])
        self.service.save_page_bboxes(
            document_id,
            page_number,
            bbox_payload_text=form.get("bbox_payload_json", ""),
            reviewer=form.get("reviewer", ""),
        )
        return self._redirect(
            f"/documents/{quote(document_id)}?page={page_number}&preview=corrected&message={quote('Bounding boxes saved.')}"
        )

    def _handle_page_commit(self, path: str, form: dict[str, str]) -> WebResponse:
        parts = path.strip("/").split("/")
        document_id = unquote(parts[1])
        page_number = int(parts[3])
        fragment_pairs = sorted(
            (
                int(key.split("_")[1]),
                value,
            )
            for key, value in form.items()
            if key.startswith("fragment_") and key.split("_")[1].isdigit()
        )
        fragment_texts = [value for _, value in fragment_pairs]
        handwritten_flags = [form.get(f"handwritten_{index}") == "true" for index, _ in fragment_pairs]
        self.service.commit_page_review(
            document_id,
            page_number,
            fragment_texts=fragment_texts,
            fragment_handwritten_flags=handwritten_flags,
            structured_content_text=form.get("structured_content_json", "{}"),
            reviewer=form.get("reviewer", ""),
            status=PageVerificationStatus(form.get("status", PageVerificationStatus.PENDING.value)),
            comments=form.get("comments", ""),
            bbox_payload_text=form.get("bbox_payload_json", ""),
        )
        return self._redirect(
            f"/documents/{quote(document_id)}?page={page_number}&preview=corrected&message={quote('Page verification saved.')}"
        )

    def _handle_verify(self, path: str, form: dict[str, str]) -> WebResponse:
        parts = path.strip("/").split("/")
        document_id = unquote(parts[1])
        page_number = int(parts[3])
        self.service.set_page_verification(
            document_id,
            page_number,
            status=PageVerificationStatus(form.get("status", PageVerificationStatus.PENDING.value)),
            reviewer=form.get("reviewer", ""),
            comments=form.get("comments", ""),
        )
        return self._redirect(f"/documents/{quote(document_id)}?page={page_number}&message={quote('Page verification updated.')}")

    def _handle_document_save(self, path: str, form: dict[str, str]) -> WebResponse:
        document_id = unquote(path.strip("/").split("/")[1])
        redirect_page = form.get("redirect_page", "1")
        self.service.save_document_edits(
            document_id,
            consolidated_text=form.get("consolidated_text", ""),
            target_output_text=form.get("target_output_json", "{}"),
            reviewer=form.get("reviewer", ""),
            comments=form.get("comments", ""),
        )
        return self._redirect(
            f"/documents/{quote(document_id)}?page={quote(redirect_page)}&preview=corrected&message={quote('Document edits saved.')}"
        )

    def _handle_approve(self, path: str, form: dict[str, str]) -> WebResponse:
        document_id = unquote(path.strip("/").split("/")[1])
        redirect_page = form.get("redirect_page", "1")
        status = ReviewStatus(form.get("status", ReviewStatus.IN_REVIEW.value))
        self.service.set_document_review_status(
            document_id,
            status=status,
            reviewer=form.get("reviewer", ""),
            comments=form.get("comments", ""),
            require_verified_pages=True,
        )
        status_message = {
            ReviewStatus.APPROVED: "Document approved. Pipeline status synced automatically.",
            ReviewStatus.IN_REVIEW: "Document kept in review. Pipeline status synced automatically.",
            ReviewStatus.NEEDS_RETRY: "Document marked for retry. Pipeline status synced automatically.",
            ReviewStatus.REJECTED: "Document rejected. Pipeline status synced automatically.",
            ReviewStatus.PENDING: "Document returned to pending. Pipeline status synced automatically.",
        }[status]
        return self._redirect(
            f"/documents/{quote(document_id)}?page={quote(redirect_page)}&preview=corrected&message={quote(status_message)}"
        )

    def _message_response(self, message: str, status_code: HTTPStatus) -> WebResponse:
        body = f"<html><body><h1>{status_code.value}</h1><p>{message}</p></body></html>".encode("utf-8")
        return WebResponse(status_code=status_code, body=body)

    def _redirect(self, location: str) -> WebResponse:
        return WebResponse(status_code=HTTPStatus.SEE_OTHER, body=b"", headers={"Location": location})

    def _json_response(self, payload: dict[str, Any]) -> WebResponse:
        return WebResponse(
            status_code=HTTPStatus.OK,
            body=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )

    def _parse_form(self, body: bytes) -> dict[str, str]:
        payload = parse_qs(body.decode("utf-8"), keep_blank_values=True)
        return {key: values[0] for key, values in payload.items()}


class InteractiveRequestHandler(BaseHTTPRequestHandler):
    controller: InteractiveWebController

    def do_GET(self) -> None:  # noqa: N802
        self._send(self.controller.handle_request("GET", self.path))

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self._send(self.controller.handle_request("POST", self.path, body=body))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send(self, response: WebResponse) -> None:
        self.send_response(response.status_code)
        self.send_header("Content-Type", response.content_type)
        for key, value in (response.headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if response.body:
            self.wfile.write(response.body)


def create_http_server(controller: InteractiveWebController, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(InteractiveRequestHandler):
        pass

    Handler.controller = controller
    return ThreadingHTTPServer((host, port), Handler)


def start_background_server(
    service: InteractiveReviewService,
    host: str,
    port: int,
) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    controller = InteractiveWebController(service, host, port)
    server = create_http_server(controller, host, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, controller.base_url()


def serve_web_ui(
    service: InteractiveReviewService,
    host: str,
    port: int,
    *,
    open_browser: bool = True,
    document_id: str | None = None,
) -> str:
    controller = InteractiveWebController(service, host, port)
    server = create_http_server(controller, host, port)
    url = controller.document_url(document_id)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return url
