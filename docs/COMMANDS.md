# NCAI Terminal Commands

This document explains every terminal command in the pipeline, what it does, when to use it, and example command lines.

## Entry Points

You can run the pipeline in either of these ways:

```powershell
python main.py <command> [options]
```

or, if the package is installed in editable mode:

```powershell
ncai-pipeline <command> [options]
```

## `run`

Runs the OCR pipeline over the input directory.

What it does:

- discovers input documents
- hashes files into stable `document_id` values
- normalizes images or renders PDFs into page PNGs
- calls the provider for OCR extraction
- stores raw requests, raw responses, parsed outputs, manifests, and review artifacts
- creates review bundles in `data/review/`

Common usage:

```powershell
python main.py run
python main.py run --input-dir docs
python main.py run --force
python main.py run --dry-run
python main.py run --config config/default.yaml
```

Important flags:

- `--input-dir`: override the configured input directory
- `--force`: reprocess documents even if they already completed a prior terminal state
- `--dry-run`: build review artifacts without making a live provider call
- `--config`: load a different YAML config file

NAS mode note:

- set `pipeline.storage_mode: "nas"` and configure `nas.input_root`, `nas.output_root`, and `nas.scratch_root`
- `--input-dir` is ignored in NAS mode; discovery always uses `nas.input_root`
- processing uses local scratch first, then publishes complete artifacts to `nas.output_root`
- document completion is marked by `working/status/documents/<document_id>.complete.json`

Outputs you will see:

- live document progress updates
- final run summary
- `data/working/status/latest_run.json`
- `data/working/status/<run_id>/run_manifest.json`
- `data/working/status/runs/<run_id>.json`

## `interactive`

Launches the new interactive module.

Modes:

- `--ui web`: full local browser-based review UI
- `--ui tui`: light terminal dashboard

### Web mode

Use this when you want the full page image + correction interface.

```powershell
python main.py interactive --ui web
python main.py interactive --ui web --document-id complain_page_1_e53ecdab15a9cdcd
python main.py interactive --ui web --host 127.0.0.1 --port 8765 --no-open-browser
```

Important flags:

- `--document-id`: open the UI focused on one document
- `--host`: bind address for the local server
- `--port`: local server port
- `--open-browser / --no-open-browser`: control whether the browser opens automatically

What the web UI gives you:

- pipeline and review dashboards
- latest run/export summaries
- document queue filtering
- page image viewer
- raw response and normalized extraction inspection
- editable corrected content
- automatic handwritten badges from block notes
- manual handwritten block assignment for handwritten-crop exports
- page verification
- document approval/rejection

### Terminal mode

Use this when you want a quick local dashboard inside the shell.

```powershell
python main.py interactive --ui tui
python main.py interactive --ui tui --document-id complain_page_1_e53ecdab15a9cdcd
```

Terminal mode is lighter than the web UI. It is mainly for:

- queue browsing
- reading latest run/export status
- opening the web review UI
- opening the static report
- seeing artifact paths and command hints

## `pipeline-status`

Shows pipeline execution state from document manifests.

```powershell
python main.py pipeline-status --status all
python main.py pipeline-status --status needs_retry
python main.py pipeline-status --status failed
python main.py pipeline-status --status review_pending
```

Supported values for `--status`:

- `all`
- `needs_retry`
- `failed`
- `review_pending`
- `approved`
- `rejected`
- `skipped`

This command is useful for checking where documents are in the processing lifecycle without reading `pipeline.log`.

## `review-status`

Shows review status counts from `review_record.json` files.

```powershell
python main.py review-status
```

Use this when you want a quick count of documents that are pending review, in review, approved, rejected, or marked `needs_retry` at the review layer.

## `approve`

Updates document-level review status from the terminal.

```powershell
python main.py approve complain_page_1_e53ecdab15a9cdcd --reviewer Talal --status approved
python main.py approve complain_page_1_e53ecdab15a9cdcd --reviewer Talal --status rejected --comments "Unreadable page"
python main.py approve complain_page_1_e53ecdab15a9cdcd --reviewer Talal --status in_review
```

Important flags:

- `--reviewer`: reviewer name
- `--status`: `pending`, `in_review`, `approved`, `rejected`, or `needs_retry`
- `--comments`: optional review comment
- `--config`: optional alternate config path

Note:

- the interactive web review flow is the recommended place for page-by-page verification
- this command remains available as a CLI shortcut for document-level status changes

## `export`

Builds curated samples from approved correction files and writes one or more dataset types in JSONL, JSON, CSV, or XLSX.

```powershell
python main.py export
python main.py export --include-dry-run
python main.py export --dataset-type page_text --format csv --format xlsx
python main.py export --dataset-type regions --region-source blocks
python main.py export --dataset-type regions --region-source sections
python main.py export --dataset-type handwritten --format jsonl
```

What it does:

- scans review records
- loads approved `corrected.yaml` files
- excludes dry-run artifacts by default
- assigns deterministic train/val/test splits
- copies whole-page images once into shared export assets
- generates handwritten block crops when requested
- skips whole-page export assets when `handwritten` is the only selected dataset type
- writes `sample.json` files and selected dataset exports in local/atomic modes
- writes per-format manifests and `data/exports/export_manifest.json` in local/atomic modes
- writes one run-level `export_manifest.json` in NAS create-only mode

You will see:

- live review-record export progress
- final export count summary
- the create-only NAS export run id, export folder, and manifest path when that mode is active

Important flags:

- `--dataset-type`: repeatable; supported values are `full`, `page_text`, `regions`, `regions_blocks`, `regions_sections`, and `handwritten`
- `--format`: repeatable; supported values are `jsonl`, `json`, `csv`, and `xlsx`
- `--region-source`: `blocks` or `sections` when `--dataset-type regions` is used
- `--include-dry-run`: include approved dry-run samples
- `--config`: load a different YAML config file

NAS mode note:

- approved corrections and rendered pages are read from `nas.output_root`
- export datasets and assets are assembled in `nas.scratch_root`
- atomic NAS mode publishes completed artifacts back to `nas.output_root/exports/`
- create-only NAS mode publishes to `exports/runs/export_<timestamp>_<datasets>_<formats>_<id>/`
- create-only NAS mode writes `exports/completed/<export_run_id>.complete.json` last instead of replacing `exports/export_manifest.json`
- dataset media paths are relative to the export folder, such as `assets/pages/<document_id>/page_0001.png`

Default config equivalent:

```yaml
export:
  dataset_types: ["full"]
  output_formats: ["jsonl"]
  region_source: "blocks"
  assets:
    page_strategy: "shared_copy"
  handwritten:
    auto_detect: true
    crop_padding: 24
```

Output layout:

```text
data/exports/
  assets/pages/<document_id>/page_0001.png
  assets/crops/handwritten/<document_id>/page_0001/<block_id>.png
  datasets/<dataset_id>/<format>/train.<ext>
  datasets/<dataset_id>/<format>/val.<ext>
  datasets/<dataset_id>/<format>/test.<ext>
  export_manifest.json
```

The default full JSONL export is also mirrored to `data/exports/gemma_multimodal/` for compatibility with older scripts.

NAS create-only output layout:

```text
<nas_output_root>/exports/
  runs/export_<timestamp>_<datasets>_<formats>_<id>/
    assets/pages/<document_id>/page_0001.png
    assets/crops/handwritten/<document_id>/page_0001/<block_id>.png
    datasets/<dataset_id>/<format>/train.<ext>
    datasets/<dataset_id>/<format>/val.<ext>
    datasets/<dataset_id>/<format>/test.<ext>
    export_manifest.json
  completed/<export_run_id>.complete.json
```

## `dry-run-artifacts`

Manages dry-run artifacts so placeholders do not accidentally leak into the production dataset.

Inspect dry-run artifacts:

```powershell
python main.py dry-run-artifacts --action inspect
```

Keep them but mark them:

```powershell
python main.py dry-run-artifacts --action keep --retention-label preserved
```

Delete them:

```powershell
python main.py dry-run-artifacts --action delete
python main.py dry-run-artifacts --action delete --document-id complain_page_1_e53ecdab15a9cdcd
```

Important flags:

- `--document-id`: operate on one document only
- `--retention-label`: label used when preserving dry-run artifacts
- `--rebuild-export / --no-rebuild-export`: rebuild dataset exports after the action
- `--include-dry-run-in-export`: include dry-run samples if export rebuild is triggered

## Common Workflows

### 1. First extraction pass

```powershell
python main.py run --force
python main.py pipeline-status --status all
```

### 2. Review a document interactively

```powershell
python main.py interactive --ui web --document-id complain_page_1_e53ecdab15a9cdcd
```

### 3. Quick shell-based review navigation

```powershell
python main.py interactive --ui tui
```

### 4. Check failures and retries

```powershell
python main.py pipeline-status --status needs_retry
python main.py pipeline-status --status failed
```

### 5. Export the approved dataset

```powershell
python main.py export
python main.py export --dataset-type full --dataset-type handwritten --format jsonl
```

### 6. Clean up placeholder dry-run data

```powershell
python main.py dry-run-artifacts --action inspect
python main.py dry-run-artifacts --action delete
```

### 7. Run Against A Mounted NAS Share

Example config override:

```yaml
pipeline:
  storage_mode: "nas"

nas:
  input_root: "Z:/NCAI/input"
  output_root: "Z:/NCAI/output"
  scratch_root: "data/nas_scratch"
  share_label: "ncai_z_drive"
  publish_strategy: "local_scratch_atomic"
```

Run the same commands:

```powershell
python main.py run --config config/nas.yaml
python main.py interactive --ui web --config config/nas.yaml
python main.py export --config config/nas.yaml
```

Use UNC paths such as `\\NAS-SERVER\NCAI\input` for production Windows runs. Mapped drives are fine for manual desktop sessions, but may not be visible to services, scheduled tasks, or different user accounts.


# Main Commands

```powershell
python main.py run --config config/default.yaml --force  
python main.py interactive --ui web --config config/default.yaml    
python main.py export --config config/default.yaml 
```
