from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
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


class BlankPageDetectionSettings(BaseModel):
    enabled: bool = True
    white_threshold: int = 245
    dark_threshold: int = 200
    max_nonwhite_ratio: float = 0.005
    max_dark_ratio: float = 0.001
    min_mean_luma: float = 248.0
    min_pixels: int = 20_000
    sample_max_dimension: int = 512

    @model_validator(mode="after")
    def validate_blank_detection(self) -> "BlankPageDetectionSettings":
        if not 0 <= self.dark_threshold <= self.white_threshold <= 255:
            raise ValueError("pipeline.blank_page_detection thresholds must satisfy 0 <= dark <= white <= 255")
        if not 0 <= self.max_nonwhite_ratio <= 1:
            raise ValueError("pipeline.blank_page_detection.max_nonwhite_ratio must be between 0 and 1")
        if not 0 <= self.max_dark_ratio <= 1:
            raise ValueError("pipeline.blank_page_detection.max_dark_ratio must be between 0 and 1")
        if not 0 <= self.min_mean_luma <= 255:
            raise ValueError("pipeline.blank_page_detection.min_mean_luma must be between 0 and 255")
        if self.min_pixels < 0:
            raise ValueError("pipeline.blank_page_detection.min_pixels must be greater than or equal to 0")
        if self.sample_max_dimension <= 0:
            raise ValueError("pipeline.blank_page_detection.sample_max_dimension must be positive")
        return self


class PipelineSettings(BaseModel):
    version: str = "0.1.0"
    storage_mode: Literal["local", "nas"] = "local"
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
    blank_page_detection: BlankPageDetectionSettings = Field(default_factory=BlankPageDetectionSettings)

    @field_validator("storage_mode", mode="before")
    @classmethod
    def validate_storage_mode(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("allowed_file_types")
    @classmethod
    def validate_allowed_file_types(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("pipeline.allowed_file_types must include at least one extension")
        normalized = []
        for item in value:
            extension = item.strip().lower()
            if not extension.startswith("."):
                extension = f".{extension}"
            if extension not in normalized:
                normalized.append(extension)
        return normalized


class NasHealthcheckSettings(BaseModel):
    enabled: bool = True
    timeout_seconds: int = 10
    require_writable_output: bool = True

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout_seconds(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("nas.healthcheck.timeout_seconds must be positive")
        return value


class NasRetrySettings(BaseModel):
    attempts: int = 3
    backoff_seconds: float = 2.0
    max_backoff_seconds: float = 30.0

    @model_validator(mode="after")
    def validate_retry(self) -> "NasRetrySettings":
        if self.attempts <= 0:
            raise ValueError("nas.retry.attempts must be positive")
        if self.backoff_seconds <= 0:
            raise ValueError("nas.retry.backoff_seconds must be positive")
        if self.max_backoff_seconds <= 0:
            raise ValueError("nas.retry.max_backoff_seconds must be positive")
        if self.max_backoff_seconds < self.backoff_seconds:
            raise ValueError("nas.retry.max_backoff_seconds must be greater than or equal to backoff_seconds")
        return self


class NasLockSettings(BaseModel):
    enabled: bool = True
    stale_after_minutes: int = 240
    heartbeat_seconds: int = 60

    @model_validator(mode="after")
    def validate_locks(self) -> "NasLockSettings":
        if self.stale_after_minutes <= 0:
            raise ValueError("nas.locks.stale_after_minutes must be positive")
        if self.heartbeat_seconds <= 0:
            raise ValueError("nas.locks.heartbeat_seconds must be positive")
        return self


class NasTimeoutSettings(BaseModel):
    read_seconds: int = 120
    write_seconds: int = 120
    publish_seconds: int = 300

    @model_validator(mode="after")
    def validate_timeouts(self) -> "NasTimeoutSettings":
        if self.read_seconds <= 0:
            raise ValueError("nas.timeouts.read_seconds must be positive")
        if self.write_seconds <= 0:
            raise ValueError("nas.timeouts.write_seconds must be positive")
        if self.publish_seconds <= 0:
            raise ValueError("nas.timeouts.publish_seconds must be positive")
        return self


class NasInputReadinessSettings(BaseModel):
    enabled: bool = True
    stable_check_seconds: float = 5.0
    require_ready_marker: bool = False
    ready_marker_suffix: str = ".ready"

    @field_validator("stable_check_seconds")
    @classmethod
    def validate_stable_check_seconds(cls, value: float) -> float:
        if value < 0:
            raise ValueError("nas.input_readiness.stable_check_seconds must be greater than or equal to 0")
        return value

    @field_validator("ready_marker_suffix")
    @classmethod
    def validate_ready_marker_suffix(cls, value: str) -> str:
        suffix = value.strip()
        if not suffix:
            raise ValueError("nas.input_readiness.ready_marker_suffix must not be empty")
        return suffix


class NasScratchSettings(BaseModel):
    cleanup_on_success: bool = True
    keep_failed_runs_days: int = 14
    keep_completed_runs_days: int = 3

    @model_validator(mode="after")
    def validate_scratch(self) -> "NasScratchSettings":
        if self.keep_failed_runs_days < 0:
            raise ValueError("nas.scratch.keep_failed_runs_days must be greater than or equal to 0")
        if self.keep_completed_runs_days < 0:
            raise ValueError("nas.scratch.keep_completed_runs_days must be greater than or equal to 0")
        return self


class NasChecksumSettings(BaseModel):
    verify_after_copy: bool = True
    verify_after_publish: bool = True


class NasSettings(BaseModel):
    input_root: str | None = None
    output_root: str | None = None
    scratch_root: str | None = "data/nas_scratch"
    share_label: str | None = None
    publish_strategy: Literal["local_scratch_atomic", "local_scratch_create_only"] = "local_scratch_atomic"
    healthcheck: NasHealthcheckSettings = Field(default_factory=NasHealthcheckSettings)
    retry: NasRetrySettings = Field(default_factory=NasRetrySettings)
    locks: NasLockSettings = Field(default_factory=NasLockSettings)
    timeouts: NasTimeoutSettings = Field(default_factory=NasTimeoutSettings)
    input_readiness: NasInputReadinessSettings = Field(default_factory=NasInputReadinessSettings)
    scratch: NasScratchSettings = Field(default_factory=NasScratchSettings)
    checksums: NasChecksumSettings = Field(default_factory=NasChecksumSettings)

    @field_validator("input_root", "output_root", "scratch_root", "share_label")
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("publish_strategy", mode="before")
    @classmethod
    def normalize_publish_strategy(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


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


class ExportAssetSettings(BaseModel):
    page_strategy: str = "shared_copy"

    @field_validator("page_strategy")
    @classmethod
    def validate_page_strategy(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"shared_copy", "reference_working"}:
            raise ValueError("export assets.page_strategy must be one of: shared_copy, reference_working")
        return normalized


class ExportHandwrittenSettings(BaseModel):
    auto_detect: bool = True
    keywords: list[str] = Field(
        default_factory=lambda: [
            "handwritten",
            "handwriting",
            "hand-written",
            "written by hand",
            "manuscript",
            "دستی",
            "دستخط",
            "تحریر",
        ]
    )
    crop_padding: int = 24

    @field_validator("crop_padding")
    @classmethod
    def validate_crop_padding(cls, value: int) -> int:
        if value < 0:
            raise ValueError("export handwritten.crop_padding must be greater than or equal to 0")
        return value


class ExportSettings(BaseModel):
    instruction: str = "Extract the document faithfully in original Urdu script and return structured JSON."
    split_ratios: ExportRatios = Field(default_factory=ExportRatios)
    target_format: str = "gemma_multimodal_jsonl"
    dataset_types: list[str] = Field(default_factory=lambda: ["full"])
    output_formats: list[str] = Field(default_factory=lambda: ["jsonl"])
    region_source: str = "blocks"
    assets: ExportAssetSettings = Field(default_factory=ExportAssetSettings)
    handwritten: ExportHandwrittenSettings = Field(default_factory=ExportHandwrittenSettings)

    @field_validator("dataset_types")
    @classmethod
    def validate_dataset_types(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("export dataset_types must include at least one dataset type")
        valid = {"full", "page_text", "regions", "regions_blocks", "regions_sections", "handwritten"}
        normalized = []
        for item in value:
            dataset_type = item.strip().lower()
            if dataset_type not in valid:
                raise ValueError(
                    "export dataset_types entries must be one of: "
                    "full, page_text, regions, regions_blocks, regions_sections, handwritten"
                )
            normalized.append(dataset_type)
        return normalized

    @field_validator("output_formats")
    @classmethod
    def validate_output_formats(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("export output_formats must include at least one format")
        valid = {"jsonl", "json", "csv", "xlsx", "excel"}
        normalized = []
        for item in value:
            output_format = item.strip().lower()
            if output_format not in valid:
                raise ValueError("export output_formats entries must be one of: jsonl, json, csv, xlsx")
            normalized.append("xlsx" if output_format == "excel" else output_format)
        return normalized

    @field_validator("region_source")
    @classmethod
    def validate_region_source(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"blocks", "sections"}:
            raise ValueError("export region_source must be one of: blocks, sections")
        return normalized


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
    nas: NasSettings = Field(default_factory=NasSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @model_validator(mode="after")
    def validate_storage(self) -> "AppConfig":
        if self.pipeline.storage_mode == "local":
            if not self.pipeline.input_dir:
                raise ValueError("pipeline.input_dir is required in local storage mode")
            if not self.pipeline.output_root:
                raise ValueError("pipeline.output_root is required in local storage mode")
            return self

        missing = [
            name
            for name, value in {
                "nas.input_root": self.nas.input_root,
                "nas.output_root": self.nas.output_root,
                "nas.scratch_root": self.nas.scratch_root,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"{', '.join(missing)} must be configured when pipeline.storage_mode is nas")

        assert self.nas.input_root is not None
        assert self.nas.output_root is not None
        assert self.nas.scratch_root is not None
        input_root = _path_for_compare(self.nas.input_root)
        output_root = _path_for_compare(self.nas.output_root)
        scratch_root = _path_for_compare(self.nas.scratch_root)

        if input_root == output_root:
            raise ValueError("nas.input_root and nas.output_root must not point to the same directory")
        if _is_relative_to(output_root, input_root):
            raise ValueError("nas.output_root must not be inside nas.input_root")
        if _is_relative_to(input_root, output_root):
            raise ValueError("nas.input_root must not be inside nas.output_root")
        if _is_relative_to(scratch_root, input_root):
            raise ValueError("nas.scratch_root must not be inside nas.input_root")
        if _is_relative_to(scratch_root, output_root):
            raise ValueError("nas.scratch_root must not be inside nas.output_root")
        if self.nas.publish_strategy == "local_scratch_create_only" and self.nas.locks.enabled:
            raise ValueError("nas.locks.enabled must be false when nas.publish_strategy is local_scratch_create_only")
        return self


def _path_for_compare(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(str(path)).absolute()


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


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
