from __future__ import annotations

import base64
import json
import logging
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from ncai_pipeline.config import EnvironmentSettings, ProviderSettings
from ncai_pipeline.ingestion.file_types import guess_mime_type
from ncai_pipeline.providers.base import (
    ExtractionProvider,
    MediaPart,
    ProviderHealth,
    ProviderInvocationResult,
    ProviderRequestEnvelope,
)
from ncai_pipeline.schemas.common import TokenUsage


LOGGER = logging.getLogger(__name__)


class GeminiProvider(ExtractionProvider):
    provider_name = "gemini"

    def __init__(self, settings: ProviderSettings, environment: EnvironmentSettings) -> None:
        self.settings = settings
        self.environment = environment
        self._last_call_started_at = 0.0
        self._cooldown_until = 0.0

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(
            provider_name=self.provider_name,
            healthy=bool(self.environment.resolved_gemini_api_key),
            details={"model_name": self.settings.model_name},
        )

    def prepare_input(self, page_paths: list[Path]) -> list[MediaPart]:
        prepared: list[MediaPart] = []
        for index, page_path in enumerate(page_paths, start=1):
            mime_type = guess_mime_type(page_path)
            data_base64 = base64.b64encode(page_path.read_bytes()).decode("utf-8")
            prepared.append(
                MediaPart(
                    page_number=index,
                    path=str(page_path),
                    mime_type=mime_type,
                    data_base64=data_base64,
                )
            )
        return prepared

    def build_request(
        self,
        document_id: str,
        page_number: int | None,
        prompt_version: str,
        prompt_bundle: dict[str, str],
        media: list[MediaPart],
        mode: str,
    ) -> ProviderRequestEnvelope:
        parts: list[dict[str, object]] = [{"text": prompt_bundle["extraction_prompt"]}]
        for media_part in media:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": media_part.mime_type,
                        "data": media_part.data_base64,
                    }
                }
            )

        payload = {
            "system_instruction": {"parts": [{"text": prompt_bundle["system_instruction"]}]},
            "contents": [{"role": "user", "parts": parts}],
            "generation_config": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        }
        return ProviderRequestEnvelope(
            document_id=document_id,
            page_number=page_number,
            prompt_version=prompt_version,
            model_name=self.settings.model_name,
            request_payload=payload,
            request_metadata={"mode": mode, "media_count": len(media)},
        )

    def invoke(self, request: ProviderRequestEnvelope, dry_run: bool = False) -> ProviderInvocationResult:
        if dry_run:
            placeholder = {
                "document_id": request.document_id,
                "page_number": request.page_number,
                "language": "ur",
                "full_text": "",
                "structured_content": {"mode": "dry_run", "notes": ["Dry-run placeholder. No provider call was made."]},
                "blocks": [],
                "lines": [],
                "tables": [],
                "warnings": ["Dry-run mode enabled."],
                "confidence_notes": ["No model confidence available in dry-run."],
            }
            return ProviderInvocationResult(
                status="dry_run",
                raw_response_text=json.dumps(placeholder, ensure_ascii=False, indent=2),
                raw_response_json={"candidates": [{"content": {"parts": [{"text": json.dumps(placeholder, ensure_ascii=False)}]}}]},
                token_usage=TokenUsage(prompt_tokens=0, output_tokens=0, total_tokens=0, raw={"dry_run": True}),
                latency_ms=0,
                retry_count=0,
                attempt_count=1,
            )

        api_key = self.environment.resolved_gemini_api_key
        if not api_key:
            raise RuntimeError("Gemini API key is not configured. Set GEMINI_PROJECT_API or GEMINI_API_KEY.")

        attempts = 0
        waited_seconds_total = 0.0
        last_result: ProviderInvocationResult | None = None
        url = f"{self.settings.api_base_url}/models/{request.model_name}:generateContent"

        while True:
            attempts += 1
            self._respect_rate_limit()
            self._respect_cooldown()
            start = time.perf_counter()
            try:
                response = httpx.post(
                    url,
                    params={"key": api_key},
                    json=request.request_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.settings.timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                latency_ms = int((time.perf_counter() - start) * 1000)
                last_result = self._build_exception_result(
                    exc=exc,
                    latency_ms=latency_ms,
                    attempts=attempts,
                    waited_seconds_total=waited_seconds_total,
                    error_category="timeout",
                )
                if attempts <= self.settings.retry_limit:
                    waited_seconds_total += self._wait_for_retry(
                        request,
                        attempts,
                        self._compute_backoff_seconds(attempts),
                        "timeout",
                    )
                    continue
                LOGGER.error(
                    "Gemini invocation failed for %s page %s: %s",
                    request.document_id,
                    request.page_number,
                    last_result.error_details,
                )
                return last_result
            except httpx.NetworkError as exc:
                latency_ms = int((time.perf_counter() - start) * 1000)
                last_result = self._build_exception_result(
                    exc=exc,
                    latency_ms=latency_ms,
                    attempts=attempts,
                    waited_seconds_total=waited_seconds_total,
                    error_category="network",
                )
                if attempts <= self.settings.retry_limit:
                    waited_seconds_total += self._wait_for_retry(
                        request,
                        attempts,
                        self._compute_backoff_seconds(attempts),
                        "network",
                    )
                    continue
                LOGGER.error(
                    "Gemini invocation failed for %s page %s: %s",
                    request.document_id,
                    request.page_number,
                    last_result.error_details,
                )
                return last_result

            latency_ms = int((time.perf_counter() - start) * 1000)
            safe_headers = self._safe_response_headers(response)
            response_payload = self._parse_response_payload(response)
            response_text = self._safe_response_text(response)

            if response.is_success:
                raw_json = response_payload if isinstance(response_payload, dict) else {}
                return ProviderInvocationResult(
                    status="succeeded",
                    raw_response_text=self._extract_text(raw_json),
                    raw_response_json=raw_json,
                    token_usage=self._extract_usage(raw_json),
                    latency_ms=latency_ms,
                    retry_count=attempts - 1,
                    attempt_count=attempts,
                    waited_seconds_total=round(waited_seconds_total, 2),
                    response_headers=safe_headers or None,
                )

            retry_after_seconds = self._parse_retry_after_header(response.headers.get("retry-after"))
            error_payload = response_payload if isinstance(response_payload, dict) else None
            error_category = self._categorize_http_status(response.status_code)
            error_summary = self._summarize_http_error(response.status_code, error_payload, response_text)
            last_result = ProviderInvocationResult(
                status="failed",
                raw_response_text=response_text,
                latency_ms=latency_ms,
                error_details=error_summary,
                error_category=error_category,
                http_status_code=response.status_code,
                retry_after_seconds=retry_after_seconds,
                attempt_count=attempts,
                retry_count=attempts - 1,
                waited_seconds_total=round(waited_seconds_total, 2),
                response_headers=safe_headers or None,
                provider_error_payload=error_payload,
            )

            if response.status_code == 429:
                wait_seconds = self._compute_rate_limit_wait_seconds(attempts, retry_after_seconds, waited_seconds_total)
                if wait_seconds is not None:
                    waited_seconds_total += self._wait_for_retry(
                        request,
                        attempts,
                        wait_seconds,
                        "rate_limit",
                        response.status_code,
                    )
                    continue
                LOGGER.error(
                    "Gemini invocation failed for %s page %s: %s",
                    request.document_id,
                    request.page_number,
                    error_summary,
                )
                last_result.waited_seconds_total = round(waited_seconds_total, 2)
                return last_result

            if response.status_code >= 500 and attempts <= self.settings.retry_limit:
                waited_seconds_total += self._wait_for_retry(
                    request,
                    attempts,
                    self._compute_backoff_seconds(attempts),
                    "server_error",
                    response.status_code,
                )
                continue

            LOGGER.error(
                "Gemini invocation failed for %s page %s: %s",
                request.document_id,
                request.page_number,
                error_summary,
            )
            last_result.waited_seconds_total = round(waited_seconds_total, 2)
            return last_result

    def parse_response(
        self,
        document_id: str,
        page_number: int | None,
        raw_response_text: str | None,
        raw_response_json: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if raw_response_text:
            try:
                payload = json.loads(raw_response_text)
                normalized_payload = self._coerce_payload(document_id, page_number, payload)
                if normalized_payload is not None:
                    return normalized_payload
            except json.JSONDecodeError:
                pass

        if raw_response_json:
            text = self._extract_text(raw_response_json)
            if text:
                try:
                    payload = json.loads(text)
                    normalized_payload = self._coerce_payload(document_id, page_number, payload)
                    if normalized_payload is not None:
                        return normalized_payload
                except json.JSONDecodeError:
                    return {
                        "document_id": document_id,
                        "page_number": page_number,
                        "full_text": text,
                        "structured_content": {"raw_text_only": True},
                        "blocks": [],
                        "lines": [],
                        "tables": [],
                        "warnings": ["Provider response was not valid JSON; raw text preserved."],
                        "confidence_notes": [],
                    }

        return {
            "document_id": document_id,
            "page_number": page_number,
            "full_text": "",
            "structured_content": {},
            "blocks": [],
            "lines": [],
            "tables": [],
            "warnings": ["No parseable content returned by provider."],
            "confidence_notes": [],
        }

    def _coerce_payload(
        self,
        document_id: str,
        page_number: int | None,
        payload: dict[str, Any] | list[Any],
    ) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            pages = payload.get("pages")
            if isinstance(pages, list):
                selected_page = self._select_page_payload(pages, page_number)
                if page_number is not None and selected_page is not None:
                    return selected_page
                if page_number is None:
                    page_payloads = [item for item in pages if isinstance(item, dict)]
                    if page_payloads:
                        return {"document_id": document_id, "pages": page_payloads}
            return payload

        if isinstance(payload, list):
            page_payloads = [item for item in payload if isinstance(item, dict)]
            if not page_payloads:
                return None
            if page_number is None:
                return {"document_id": document_id, "pages": page_payloads}
            selected_page = self._select_page_payload(page_payloads, page_number)
            return selected_page or page_payloads[0]

        return None

    def _select_page_payload(self, pages: list[Any], page_number: int | None) -> dict[str, Any] | None:
        page_payloads = [item for item in pages if isinstance(item, dict)]
        if not page_payloads:
            return None
        if page_number is not None:
            for item in page_payloads:
                if item.get("page_number") == page_number:
                    return item
        if len(page_payloads) == 1:
            return page_payloads[0]
        return page_payloads[0]

    def _respect_rate_limit(self) -> None:
        if self.settings.rate_limit_per_minute <= 0:
            return
        minimum_interval = 60.0 / self.settings.rate_limit_per_minute
        elapsed = time.perf_counter() - self._last_call_started_at
        if self._last_call_started_at and elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)
        self._last_call_started_at = time.perf_counter()

    def _respect_cooldown(self) -> None:
        if self._cooldown_until <= 0:
            return
        remaining = self._cooldown_until - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self._cooldown_until = 0.0

    def _compute_backoff_seconds(self, attempts: int) -> float:
        exponent = max(attempts - 1, 0)
        return min(
            self.settings.retry_backoff_seconds * (2**exponent),
            self.settings.retry_max_backoff_seconds,
        )

    def _compute_rate_limit_wait_seconds(
        self,
        attempts: int,
        retry_after_seconds: float | None,
        waited_seconds_total: float,
    ) -> float | None:
        if attempts > self.settings.rate_limit_max_retries:
            return None
        remaining_budget = max(self.settings.rate_limit_wait_budget_seconds - waited_seconds_total, 0.0)
        if remaining_budget <= 0:
            return None

        requested_wait = retry_after_seconds or self._compute_backoff_seconds(attempts)
        wait_seconds = min(requested_wait, remaining_budget, self.settings.retry_max_backoff_seconds)
        if wait_seconds <= 0:
            return None
        return wait_seconds

    def _wait_for_retry(
        self,
        request: ProviderRequestEnvelope,
        attempts: int,
        wait_seconds: float,
        reason: str,
        http_status_code: int | None = None,
    ) -> float:
        rounded_wait = round(wait_seconds, 2)
        self._cooldown_until = max(self._cooldown_until, time.monotonic() + rounded_wait)
        if reason == "rate_limit":
            LOGGER.warning(
                "Retry cooldown started for %s page %s after HTTP %s; waiting %.2fs before attempt %s",
                request.document_id,
                request.page_number,
                http_status_code,
                rounded_wait,
                attempts + 1,
            )
        time.sleep(rounded_wait)
        return rounded_wait

    def _parse_retry_after_header(self, value: str | None) -> float | None:
        if not value:
            return None
        raw_value = value.strip()
        if not raw_value:
            return None
        try:
            return max(float(raw_value), 0.0)
        except ValueError:
            pass

        try:
            retry_at = parsedate_to_datetime(raw_value)
        except (TypeError, ValueError, IndexError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        return max(delay, 0.0)

    def _safe_response_headers(self, response: httpx.Response) -> dict[str, Any]:
        retry_after = response.headers.get("retry-after")
        return {"retry-after": retry_after} if retry_after else {}

    def _parse_response_payload(self, response: httpx.Response) -> dict[str, Any] | list[Any] | None:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, (dict, list)) else None

    def _safe_response_text(self, response: httpx.Response) -> str | None:
        text = response.text.strip()
        return self._sanitize_error(text) if text else None

    def _categorize_http_status(self, status_code: int) -> str:
        if status_code == 429:
            return "rate_limited"
        if status_code >= 500:
            return "server_error"
        return "http_error"

    def _summarize_http_error(
        self,
        status_code: int,
        error_payload: dict[str, Any] | None,
        response_text: str | None,
    ) -> str:
        if error_payload:
            nested_error = error_payload.get("error")
            if isinstance(nested_error, dict):
                message = nested_error.get("message")
                status = nested_error.get("status")
                parts = [f"HTTP {status_code}"]
                if status:
                    parts.append(str(status))
                if message:
                    parts.append(str(message))
                return self._sanitize_error(" | ".join(parts))
        if response_text:
            return self._sanitize_error(f"HTTP {status_code} | {response_text}")
        return f"HTTP {status_code} request failed"

    def _build_exception_result(
        self,
        exc: Exception,
        latency_ms: int,
        attempts: int,
        waited_seconds_total: float,
        error_category: str,
    ) -> ProviderInvocationResult:
        sanitized_error = self._sanitize_error(str(exc))
        return ProviderInvocationResult(
            status="failed",
            latency_ms=latency_ms,
            error_details=sanitized_error,
            error_category=error_category,
            attempt_count=attempts,
            retry_count=attempts - 1,
            waited_seconds_total=round(waited_seconds_total, 2),
        )

    def _extract_text(self, response_json: dict[str, Any]) -> str:
        texts: list[str] = []
        for candidate in response_json.get("candidates", []):
            content = candidate.get("content") or {}
            for part in content.get("parts", []):
                if isinstance(part, dict) and part.get("text"):
                    texts.append(part["text"])
        return "\n".join(texts).strip()

    def _extract_usage(self, response_json: dict[str, Any]) -> TokenUsage | None:
        usage = response_json.get("usageMetadata")
        if not usage:
            return None
        return TokenUsage(
            prompt_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            total_tokens=usage.get("totalTokenCount"),
            raw=usage,
        )

    def _sanitize_error(self, message: str) -> str:
        redacted = re.sub(r"key=[^'\"&\s]+", "key=***REDACTED***", message)
        api_key = self.environment.resolved_gemini_api_key
        if api_key:
            redacted = redacted.replace(api_key, "***REDACTED***")
        return redacted
