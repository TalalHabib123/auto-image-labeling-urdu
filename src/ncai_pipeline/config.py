from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    gemini_project_api: str | None = Field(default=None, alias="GEMINI_PROJECT_API")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    ncai_config_path: str | None = Field(default=None, alias="NCAI_CONFIG_PATH")

    @property
    def resolved_gemini_api_key(self) -> str | None:
        return self.gemini_api_key or self.gemini_project_api


class PipelineSettings(BaseModel):
    version: str = "0.1.0"
    input_dir: str = "data/input"
    output_root: str = "data"
    allowed_file_types: list[str] = Field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".pdf"])
    process_mode: str = "per_page"
    skip_completed: bool = True
    force_reprocess: bool = False
    dry_run: bool = False
    concurrency: int = 1
    prompt_version: str = "ocr_v1"
    pdf_render_scale: float = 2.0


class ProviderSettings(BaseModel):
    name: str = "gemini"
    model_name: str = "gemini-3-flash-preview"
    timeout_seconds: int = 90
    retry_limit: int = 3
    retry_backoff_seconds: float = 5.0
    retry_max_backoff_seconds: float = 60.0
    rate_limit_per_minute: int = 5
    rate_limit_max_retries: int = 4
    rate_limit_wait_budget_seconds: float = 300.0
    api_base_url: str = "https://generativelanguage.googleapis.com/v1beta"


class ReviewSettings(BaseModel):
    correction_format: str = "yaml"
    report_title: str = "NCAI OCR Review"


class ExportRatios(BaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    @model_validator(mode="after")
    def validate_total(self) -> "ExportRatios":
        total = round(self.train + self.val + self.test, 6)
        if total != 1.0:
            raise ValueError("export split ratios must sum to 1.0")
        return self


class ExportSettings(BaseModel):
    instruction: str = "Extract the document faithfully in original Urdu script and return structured JSON."
    split_ratios: ExportRatios = Field(default_factory=ExportRatios)
    target_format: str = "gemma_multimodal_jsonl"


class LoggingSettings(BaseModel):
    level: str = "INFO"
    structured: bool = True


class PromptTemplate(BaseModel):
    system_instruction: str
    extraction_prompt: str


class PromptRegistry(BaseModel):
    prompts: dict[str, PromptTemplate]

    def get_prompt(self, version: str) -> PromptTemplate:
        if version not in self.prompts:
            raise KeyError(f"Unknown prompt version: {version}")
        return self.prompts[version]


class AppConfig(BaseModel):
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_environment() -> EnvironmentSettings:
    return EnvironmentSettings()


def resolve_config_path(explicit_path: str | None = None, env: EnvironmentSettings | None = None) -> Path:
    env = env or load_environment()
    candidate = explicit_path or env.ncai_config_path or "config/default.yaml"
    return Path(candidate)


def load_app_config(config_path: str | None = None) -> AppConfig:
    env = load_environment()
    path = resolve_config_path(config_path, env)
    return AppConfig.model_validate(_read_yaml(path))


def load_prompts(prompts_path: str | None = None) -> PromptRegistry:
    path = Path(prompts_path or "config/prompts.yaml")
    return PromptRegistry.model_validate(_read_yaml(path))
