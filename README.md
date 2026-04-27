# NCAI OCR Pipeline

Production-style OCR extraction pipeline for Urdu-heavy documents with:

- modular ingestion and page normalization
- Gemini behind a provider abstraction
- raw/parsed/review artifact persistence
- YAML-based human correction workflow
- curated training sample assembly
- deterministic multi-format exports for future Gemma multimodal fine-tuning and OCR datasets

## Architecture

The pipeline is organized under `src/ncai_pipeline/` by responsibility:

- `ingestion/`: file discovery, type detection, image normalization, PDF rendering
- `providers/`: provider contract plus Gemini implementation
- `parsing/`: raw response parsing, normalization, validation
- `pipeline/`: orchestration, state transitions, run flow
- `storage/`: deterministic artifact paths and repository helpers
- `review/`: correction YAML and HTML review report generation
- `interactive/`: shared review service plus local web and terminal review surfaces
- `export/`: curated sample building, split assignment, dataset assets, and JSONL/JSON/CSV/XLSX export
- `observability/`: logging, metrics, run manifests
- `schemas/`: Pydantic models for manifests, extraction, review, and export artifacts

## Setup

1. Create or activate a Python 3.11 environment.
2. Install the project in editable mode:

```bash
pip install -e .[dev]
```

3. Copy `.env.example` to `.env` and set the Gemini API key.

Supported secret names:

- `GEMINI_PROJECT_API`
- `GEMINI_API_KEY`

## Configuration

Primary configuration lives in:

- `config/default.yaml`
- `config/prompts.yaml`

Important settings:

- input directory
- output root
- allowed file types
- provider name and Gemini model
- timeout and retry settings
- prompt version
- `per_page` or `per_document` processing
- export split ratios
- export dataset types, output formats, region source, shared asset strategy, and handwritten crop settings
- logging level

You can point to a different config file with `NCAI_CONFIG_PATH` or `--config`.

## Run Flow

### 1. Ingest and extract

```bash
python main.py run
```

Useful flags:

- `--input-dir docs`
- `--force`
- `--dry-run`

What `run` does:

1. recursively scans the input directory
2. hashes each document and creates a stable `document_id`
3. normalizes images or renders PDFs into page PNGs
4. builds Gemini extraction requests through the provider interface
5. stores raw request and raw response artifacts
6. normalizes parsed results into internal schemas
7. generates a review bundle with YAML + HTML report

### 2. Review artifacts

Artifacts are emitted under `data/`:

- `data/working/manifests/<document_id>/manifest.json`
- `data/working/rendered_pages/<document_id>/page_XXXX.png`
- `data/working/raw_requests/<document_id>/...`
- `data/working/raw_responses/<document_id>/...`
- `data/working/parsed/<document_id>/...`
- `data/review/correction_files/<document_id>/corrected.yaml`
- `data/review/reports/<document_id>/report.html`
- `data/working/status/<document_id>/review_record.json`

### 3. Interactive review module

You can review documents through the new interactive module instead of editing YAML manually.

Full web review UI:

```bash
python main.py interactive --ui web
```

Focused document review:

```bash
python main.py interactive --ui web --document-id <document_id>
```

Light terminal dashboard:

```bash
python main.py interactive --ui tui
```

The web UI provides:

- dashboard views for pipeline and review state
- latest run/export summaries
- page-by-page image review
- raw provider response inspection
- normalized extraction inspection
- editable corrected content
- manual handwritten block tagging, with automatic badges from block notes
- page verification and document approval/rejection

## Correction Workflow

The canonical correction artifact is:

- `data/review/correction_files/<document_id>/corrected.yaml`

Reviewers should edit:

- `approval_metadata`
- `consolidated_text`
- `target_output`
- any page-level extracted content that needs correction

Suggested workflow:

1. open the HTML report
2. compare the source page image against the parsed output
3. edit `corrected.yaml` manually, or use the interactive web module
4. approve or reject the document

Approve a document:

```bash
python main.py approve <document_id> --reviewer <name> --comments "ready for training"
```

Check review status:

```bash
python main.py review-status
```

## Export Flow

Only approved correction files are exported.

```bash
python main.py export
```

The default export remains the full Gemma-style JSONL dataset. You can select additional dataset types and formats from the command line or `config/default.yaml`.

```bash
python main.py export --dataset-type page_text --format csv --format xlsx
python main.py export --dataset-type regions --region-source blocks
python main.py export --dataset-type regions --region-source sections
python main.py export --dataset-type handwritten --format jsonl
```

Export outputs:

- `data/curated/approved/<document_id>/sample.json`
- `data/exports/assets/pages/<document_id>/page_XXXX.png`
- `data/exports/assets/crops/handwritten/<document_id>/page_XXXX/<block_id>.png`
- `data/exports/datasets/<dataset_id>/<format>/train.<ext>`
- `data/exports/datasets/<dataset_id>/<format>/val.<ext>`
- `data/exports/datasets/<dataset_id>/<format>/test.<ext>`
- `data/exports/export_manifest.json`
- legacy full JSONL aliases under `data/exports/gemma_multimodal/`

Each exported sample contains:

- input media paths
- task instruction
- target structured output
- approval metadata
- deterministic train/val/test split

Supported dataset types:

- `full`: one sample per approved document with the complete target output
- `page_text`: one sample per page with page image path and `full_text`
- `regions` with `--region-source blocks`: page-level block targets with bbox and handwritten flags
- `regions` with `--region-source sections`: page-level structured sections
- `handwritten`: one sample per handwritten block crop with crop path and text

Supported formats are `jsonl`, `json`, `csv`, and `xlsx`.

## Example Output Artifacts

### Review correction YAML

```yaml
document_id: example_doc
approval_metadata:
  reviewer: qa_user
  review_status: approved
consolidated_text: "..."
target_output:
  document_id: example_doc
  pages: []
```

### Exported training sample

```json
{
  "sample_id": "example_doc_train",
  "document_id": "example_doc",
  "input_media_paths": ["data/working/rendered_pages/example_doc/page_0001.png"],
  "task_instruction": "Extract the document faithfully in original Urdu script and return structured JSON.",
  "target_output": {
    "document_id": "example_doc",
    "pages": []
  },
  "language": "ur",
  "source_split": "train"
}
```

## Provider Swap Points

Provider-specific logic is isolated behind `ExtractionProvider` in `providers/base.py`.

To add another backend:

1. implement the provider contract
2. register it in `providers/factory.py`
3. keep orchestration and storage untouched

## Extension Points

- add richer JSON response schemas in the Gemini request builder
- add alternate review report renderers
- add more export adapters beyond JSONL
- add database-backed artifact indexing later without changing schema models
- add local/offline multimodal providers behind the same interface

## Assumptions

- default input root is `data/input`
- default review source of truth is YAML
- default extraction mode is `per_page`
- `extras/` stays experimental and is not used by the production pipeline
- raw outputs are always preserved even when corrected outputs exist

## Limitations

- current Gemini provider uses direct REST calls and stores inline image requests, which can make raw request artifacts large
- the interactive review UI is local-only and does not include authentication or multi-user workflow in v1
- automated tests use fake providers and dry, deterministic fixtures rather than live Gemini calls

## Terminal Commands

Detailed command reference:

- see [docs/COMMANDS.md](docs/COMMANDS.md)

Quick summary:

### `run`

Runs document discovery, normalization, extraction, parsing, and review artifact generation.

Examples:

```bash
python main.py run
python main.py run --input-dir docs
python main.py run --force
python main.py run --dry-run
```

### `interactive`

Launches the interactive review module.

Examples:

```bash
python main.py interactive --ui web
python main.py interactive --ui web --document-id <document_id>
python main.py interactive --ui tui
```

### `pipeline-status`

Shows pipeline execution state from document manifests.

Examples:

```bash
python main.py pipeline-status --status all
python main.py pipeline-status --status needs_retry
python main.py pipeline-status --status failed
```

### `review-status`

Shows review status counts from review records.

Example:

```bash
python main.py review-status
```

### `approve`

Updates document-level review state from the terminal.

Examples:

```bash
python main.py approve <document_id> --reviewer <name> --status approved
python main.py approve <document_id> --reviewer <name> --status rejected --comments "Needs manual fix"
```

### `export`

Builds curated approved samples in selected dataset types and formats.

Examples:

```bash
python main.py export
python main.py export --include-dry-run
python main.py export --dataset-type page_text --format csv --format xlsx
python main.py export --dataset-type handwritten --format jsonl
```

### `dry-run-artifacts`

Inspects, preserves, or deletes dry-run artifacts.

Examples:

```bash
python main.py dry-run-artifacts --action inspect
python main.py dry-run-artifacts --action keep --retention-label preserved
python main.py dry-run-artifacts --action delete
```
