from __future__ import annotations

from ncai_pipeline.config import AppConfig, EnvironmentSettings
from ncai_pipeline.providers.base import ExtractionProvider
from ncai_pipeline.providers.gemini_provider import GeminiProvider


def build_provider(config: AppConfig, environment: EnvironmentSettings) -> ExtractionProvider:
    if config.provider.name == "gemini":
        return GeminiProvider(config.provider, environment)
    raise ValueError(f"Unsupported provider: {config.provider.name}")

