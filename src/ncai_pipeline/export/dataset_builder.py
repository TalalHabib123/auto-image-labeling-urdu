from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ncai_pipeline.config import AppConfig
from ncai_pipeline.export.split_builder import assign_split
from ncai_pipeline.parsing.validators import load_corrected_document
from ncai_pipeline.review.dry_run_policy import corrected_document_is_dry_run
from ncai_pipeline.schemas.common import ReviewStatus
from ncai_pipeline.schemas.export import CuratedTrainingSample, ExportManifest, ExportStats
from ncai_pipeline.schemas.review import ReviewRecord
from ncai_pipeline.storage.paths import DataPaths
from ncai_pipeline.storage.repository import FileRepository
from ncai_pipeline.utils.time import utc_now_iso


class DatasetBuilder:
    def __init__(self, config: AppConfig, data_paths: DataPaths, repository: FileRepository) -> None:
        self.config = config
        self.data_paths = data_paths
        self.repository = repository

    def load_review_records(self) -> list[ReviewRecord]:
        records: list[ReviewRecord] = []
        for path in sorted(self.data_paths.status_dir.glob("*/review_record.json")):
            records.append(ReviewRecord.model_validate(self.repository.read_json(path)))
        return records

    def build_samples(
        self,
        include_dry_run: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[CuratedTrainingSample], ExportManifest]:
        samples: list[CuratedTrainingSample] = []
        language_counts: Counter[str] = Counter()
        review_counts: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        split_page_counts: Counter[str] = Counter()
        total_text_length = 0
        excluded_dry_run = 0
        records = self.load_review_records()
        total_records = len(records)

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "start",
                    "total_records": total_records,
                    "completed_records": 0,
                    "included_samples": 0,
                }
            )

        for index, record in enumerate(records, start=1):
            review_counts[record.review_status.value] += 1
            corrected = load_corrected_document(Path(record.corrected_output_path), self.repository)
            outcome = "skipped_not_approved"
            if corrected.approval_metadata.review_status != ReviewStatus.APPROVED:
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "record",
                            "document_id": record.document_id,
                            "completed_records": index,
                            "total_records": total_records,
                            "included_samples": len(samples),
                            "outcome": outcome,
                            "review_status": corrected.approval_metadata.review_status.value,
                        }
                    )
                continue
            if corrected_document_is_dry_run(corrected) and not include_dry_run:
                excluded_dry_run += 1
                outcome = "skipped_dry_run"
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "record",
                            "document_id": record.document_id,
                            "completed_records": index,
                            "total_records": total_records,
                            "included_samples": len(samples),
                            "outcome": outcome,
                            "review_status": corrected.approval_metadata.review_status.value,
                        }
                    )
                continue

            split = assign_split(record.document_id, self.config.export.split_ratios)
            input_media_paths = [
                str(self.data_paths.rendered_pages_dir / record.document_id / f"page_{(page.page_number or 0):04d}.png")
                for page in corrected.extracted_pages
            ]
            sample = CuratedTrainingSample(
                sample_id=f"{record.document_id}_{split}",
                document_id=record.document_id,
                page_refs=[page.page_number or 0 for page in corrected.extracted_pages],
                input_media_paths=input_media_paths,
                task_instruction=self.config.export.instruction,
                target_output=corrected.target_output or {
                    "document_id": corrected.document_id,
                    "consolidated_text": corrected.consolidated_text,
                },
                language=(corrected.extracted_pages[0].language if corrected.extracted_pages else "ur") or "ur",
                source_split=split,
                approval_metadata=corrected.approval_metadata,
            )
            samples.append(sample)
            split_counts[split] += 1
            split_page_counts[split] += len(sample.page_refs)
            language_counts[sample.language] += 1
            total_text_length += len(corrected.consolidated_text)
            target_dir = self.data_paths.dry_run_dir if corrected_document_is_dry_run(corrected) else self.data_paths.approved_dir
            self.repository.write_json(target_dir / record.document_id / "sample.json", sample)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "record",
                        "document_id": record.document_id,
                        "completed_records": index,
                        "total_records": total_records,
                        "included_samples": len(samples),
                        "outcome": "included",
                        "review_status": corrected.approval_metadata.review_status.value,
                        "split": split,
                    }
                )

        stats = ExportStats(
            sample_counts=dict(split_counts),
            page_counts=dict(split_page_counts),
            average_text_length=round(total_text_length / len(samples), 2) if samples else 0.0,
            language_counts=dict(language_counts),
            review_coverage=dict(review_counts),
            error_counts={"dry_run_excluded": excluded_dry_run} if excluded_dry_run else {},
        )
        manifest = ExportManifest(
            export_name=self.config.export.target_format,
            samples=len(samples),
            generated_at=utc_now_iso(),
            stats=stats,
        )
        return samples, manifest
