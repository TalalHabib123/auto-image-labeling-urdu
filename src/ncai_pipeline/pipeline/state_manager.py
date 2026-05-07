from __future__ import annotations

from ncai_pipeline.schemas.common import PipelineStatus
from ncai_pipeline.schemas.document import DocumentManifest
from ncai_pipeline.utils.time import utc_now


class StateManager:
    terminal_statuses = {
        PipelineStatus.REVIEW_PENDING,
        PipelineStatus.IN_REVIEW,
        PipelineStatus.APPROVED,
        PipelineStatus.REJECTED,
        PipelineStatus.SKIPPED,
    }

    def should_skip(self, existing: DocumentManifest | None, force: bool, skip_completed: bool) -> bool:
        if force or not existing:
            return False
        return skip_completed and existing.status in self.terminal_statuses

    def update_status(self, manifest: DocumentManifest, status: PipelineStatus, error_details: str | None = None) -> DocumentManifest:
        manifest.status = status
        manifest.error_details = error_details
        manifest.updated_at = utc_now()
        return manifest
