from __future__ import annotations

from pathlib import Path

from ncai_pipeline.schemas.document import DocumentManifest
from ncai_pipeline.storage.repository import FileRepository


class ManifestStore:
    def __init__(self, repository: FileRepository, manifests_dir: Path) -> None:
        self.repository = repository
        self.manifests_dir = manifests_dir

    def manifest_path(self, document_id: str) -> Path:
        return self.manifests_dir / document_id / "manifest.json"

    def save(self, manifest: DocumentManifest) -> Path:
        path = self.manifest_path(manifest.document_id)
        self.repository.write_json(path, manifest)
        return path

    def load(self, document_id: str) -> DocumentManifest | None:
        path = self.manifest_path(document_id)
        if not path.exists():
            return None
        return DocumentManifest.model_validate(self.repository.read_json(path))

