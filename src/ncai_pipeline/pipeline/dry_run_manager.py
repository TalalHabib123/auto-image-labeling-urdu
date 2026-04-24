from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ncai_pipeline.review.dry_run_policy import corrected_document_is_dry_run, review_record_is_dry_run
from ncai_pipeline.schemas.review import CorrectedDocument, ReviewRecord
from ncai_pipeline.storage.paths import DataPaths
from ncai_pipeline.storage.repository import FileRepository


@dataclass
class DryRunArtifact:
    document_id: str
    review_record_path: Path
    corrected_output_path: Path
    review_record: ReviewRecord
    corrected_document: CorrectedDocument


class DryRunArtifactManager:
    def __init__(self, data_paths: DataPaths, repository: FileRepository) -> None:
        self.data_paths = data_paths
        self.repository = repository

    def discover(self, document_id: str | None = None) -> list[DryRunArtifact]:
        artifacts: list[DryRunArtifact] = []
        for review_record_path in sorted(self.data_paths.status_dir.glob("*/review_record.json")):
            review_record = ReviewRecord.model_validate(self.repository.read_json(review_record_path))
            if document_id and review_record.document_id != document_id:
                continue
            corrected_path = Path(review_record.corrected_output_path)
            if not corrected_path.exists():
                continue
            corrected_document = CorrectedDocument.model_validate(self.repository.read_yaml(corrected_path))
            is_dry_run = review_record_is_dry_run(review_record) or corrected_document_is_dry_run(corrected_document)
            if not is_dry_run:
                continue
            artifacts.append(
                DryRunArtifact(
                    document_id=review_record.document_id,
                    review_record_path=review_record_path,
                    corrected_output_path=corrected_path,
                    review_record=review_record,
                    corrected_document=corrected_document,
                )
            )
        return artifacts

    def keep(self, document_id: str | None = None, retention_label: str = "kept") -> dict[str, object]:
        artifacts = self.discover(document_id)
        warnings: list[str] = []
        for artifact in artifacts:
            artifact.review_record.is_dry_run = True
            artifact.review_record.retention_label = retention_label
            self.repository.write_json(artifact.review_record_path, artifact.review_record)

            artifact.corrected_document.approval_metadata.is_dry_run = True
            artifact.corrected_document.approval_metadata.retention_label = retention_label
            notes = artifact.corrected_document.approval_metadata.notes
            keep_note = f"Dry-run artifacts retained with label: {retention_label}"
            if keep_note not in notes:
                notes.append(keep_note)
            self.repository.write_yaml(artifact.corrected_output_path, artifact.corrected_document)

            self._copy_then_try_remove(
                self.data_paths.approved_dir / artifact.document_id,
                self.data_paths.dry_run_dir / artifact.document_id,
                warnings,
            )
            self._copy_then_try_remove(
                self.data_paths.rejected_dir / artifact.document_id,
                self.data_paths.dry_run_dir / artifact.document_id,
                warnings,
            )

        return {
            "action": "keep",
            "documents": [artifact.document_id for artifact in artifacts],
            "count": len(artifacts),
            "retention_label": retention_label,
            "warnings": warnings,
        }

    def delete(self, document_id: str | None = None) -> dict[str, object]:
        artifacts = self.discover(document_id)
        warnings: list[str] = []
        for artifact in artifacts:
            for directory in self._document_directories(artifact.document_id):
                if directory.exists():
                    try:
                        shutil.rmtree(directory)
                    except OSError as exc:
                        warnings.append(f"Could not delete {directory}: {exc}")
        return {
            "action": "delete",
            "documents": [artifact.document_id for artifact in artifacts],
            "count": len(artifacts),
            "warnings": warnings,
        }

    def _document_directories(self, document_id: str) -> list[Path]:
        return [
            self.data_paths.manifests_dir / document_id,
            self.data_paths.rendered_pages_dir / document_id,
            self.data_paths.raw_requests_dir / document_id,
            self.data_paths.raw_responses_dir / document_id,
            self.data_paths.parsed_dir / document_id,
            self.data_paths.status_dir / document_id,
            self.data_paths.reports_dir / document_id,
            self.data_paths.correction_files_dir / document_id,
            self.data_paths.approved_dir / document_id,
            self.data_paths.rejected_dir / document_id,
            self.data_paths.dry_run_dir / document_id,
        ]

    def _copy_then_try_remove(self, source: Path, target: Path, warnings: list[str]) -> None:
        if not source.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            warnings.append(f"Preserved copy already exists at {target}; leaving it in place.")
        else:
            shutil.copytree(source, target)
        try:
            shutil.rmtree(source)
        except OSError as exc:
            warnings.append(f"Copied {source} to {target} but could not remove original: {exc}")
