from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import Field

from ncai_pipeline.schemas.common import BaseSchemaModel, TokenUsage


class MediaPart(BaseSchemaModel):
    page_number: int | None = None
    path: str
    mime_type: str
    data_base64: str


class ProviderRequestEnvelope(BaseSchemaModel):
    document_id: str
    page_number: int | None = None
    prompt_version: str
    model_name: str
    request_payload: dict[str, Any]
    request_metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderInvocationResult(BaseSchemaModel):
    status: str
    raw_response_text: str | None = None
    raw_response_json: dict[str, Any] | None = None
    token_usage: TokenUsage | None = None
    latency_ms: int | None = None
    error_details: str | None = None
    error_category: str | None = None
    http_status_code: int | None = None
    retry_after_seconds: float | None = None
    attempt_count: int = 1
    retry_count: int = 0
    waited_seconds_total: float = 0.0
    response_headers: dict[str, Any] | None = None
    provider_error_payload: dict[str, Any] | None = None


class ProviderHealth(BaseSchemaModel):
    provider_name: str
    healthy: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ExtractionProvider(ABC):
    provider_name: str

    @abstractmethod
    def healthcheck(self) -> ProviderHealth:
        raise NotImplementedError

    @abstractmethod
    def prepare_input(self, page_paths: list[Path]) -> list[MediaPart]:
        raise NotImplementedError

    @abstractmethod
    def build_request(
        self,
        document_id: str,
        page_number: int | None,
        prompt_version: str,
        prompt_bundle: dict[str, str],
        media: list[MediaPart],
        mode: str,
    ) -> ProviderRequestEnvelope:
        raise NotImplementedError

    @abstractmethod
    def invoke(self, request: ProviderRequestEnvelope, dry_run: bool = False) -> ProviderInvocationResult:
        raise NotImplementedError

    @abstractmethod
    def parse_response(
        self,
        document_id: str,
        page_number: int | None,
        raw_response_text: str | None,
        raw_response_json: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raise NotImplementedError
