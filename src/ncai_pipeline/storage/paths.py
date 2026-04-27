from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def working_dir(self) -> Path:
        return self.root / "working"

    @property
    def review_dir(self) -> Path:
        return self.root / "review"

    @property
    def curated_dir(self) -> Path:
        return self.root / "curated"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def export_assets_dir(self) -> Path:
        return self.exports_dir / "assets"

    @property
    def export_page_assets_dir(self) -> Path:
        return self.export_assets_dir / "pages"

    @property
    def export_crop_assets_dir(self) -> Path:
        return self.export_assets_dir / "crops"

    @property
    def export_datasets_dir(self) -> Path:
        return self.exports_dir / "datasets"

    @property
    def manifests_dir(self) -> Path:
        return self.working_dir / "manifests"

    @property
    def rendered_pages_dir(self) -> Path:
        return self.working_dir / "rendered_pages"

    @property
    def raw_requests_dir(self) -> Path:
        return self.working_dir / "raw_requests"

    @property
    def raw_responses_dir(self) -> Path:
        return self.working_dir / "raw_responses"

    @property
    def parsed_dir(self) -> Path:
        return self.working_dir / "parsed"

    @property
    def status_dir(self) -> Path:
        return self.working_dir / "status"

    @property
    def reports_dir(self) -> Path:
        return self.review_dir / "reports"

    @property
    def correction_files_dir(self) -> Path:
        return self.review_dir / "correction_files"

    @property
    def approved_dir(self) -> Path:
        return self.curated_dir / "approved"

    @property
    def rejected_dir(self) -> Path:
        return self.curated_dir / "rejected"

    @property
    def dry_run_dir(self) -> Path:
        return self.curated_dir / "dry_run"

    @property
    def export_target_dir(self) -> Path:
        return self.exports_dir / "gemma_multimodal"

    def ensure(self) -> None:
        for path in [
            self.input_dir,
            self.manifests_dir,
            self.rendered_pages_dir,
            self.raw_requests_dir,
            self.raw_responses_dir,
            self.parsed_dir,
            self.status_dir,
            self.reports_dir,
            self.correction_files_dir,
            self.approved_dir,
            self.rejected_dir,
            self.dry_run_dir,
            self.exports_dir,
            self.export_page_assets_dir,
            self.export_crop_assets_dir,
            self.export_datasets_dir,
            self.export_target_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def document_dir(self, base: Path, document_id: str) -> Path:
        return base / document_id
