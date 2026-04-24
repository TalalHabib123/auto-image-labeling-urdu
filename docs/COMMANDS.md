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

Outputs you will see:

- live document progress updates
- final run summary
- `data/working/status/latest_run.json`
- `data/working/status/<run_id>/run_manifest.json`

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

Builds curated samples and writes JSONL exports from approved correction files.

```powershell
python main.py export
python main.py export --include-dry-run
```

What it does:

- scans review records
- loads approved `corrected.yaml` files
- excludes dry-run artifacts by default
- assigns deterministic train/val/test splits
- writes `sample.json` files and JSONL exports
- writes `data/exports/gemma_multimodal/export_manifest.json`

You will see:

- live review-record export progress
- final export count summary

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
```

### 6. Clean up placeholder dry-run data

```powershell
python main.py dry-run-artifacts --action inspect
python main.py dry-run-artifacts --action delete
```
