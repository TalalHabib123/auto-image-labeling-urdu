from __future__ import annotations

from pathlib import Path

from ncai_pipeline.schemas.extraction import NormalizedExtraction, RawExtractionRecord
from ncai_pipeline.storage.repository import FileRepository


class RawArtifactStore:
    def __init__(self, repository: FileRepository, requests_dir: Path, responses_dir: Path, parsed_dir: Path) -> None:
        self.repository = repository
        self.requests_dir = requests_dir
        self.responses_dir = responses_dir
        self.parsed_dir = parsed_dir

    def _scope_name(self, page_number: int | None) -> str:
        return f"page_{page_number:04d}" if page_number is not None else "document"

    def save_request(self, document_id: str, page_number: int | None, payload: dict[str, object]) -> Path:
        path = self.requests_dir / document_id / f"{self._scope_name(page_number)}.json"
        self.repository.write_json(path, payload)
        return path

    def save_raw_response(self, record: RawExtractionRecord) -> Path:
        path = self.responses_dir / record.document_id / f"{self._scope_name(record.page_number)}.json"
        self.repository.write_json(path, record)
        return path

    def save_parsed(self, extraction: NormalizedExtraction) -> Path:
        path = self.parsed_dir / extraction.document_id / f"{self._scope_name(extraction.page_number)}.json"
        self.repository.write_json(path, extraction)
        return path

