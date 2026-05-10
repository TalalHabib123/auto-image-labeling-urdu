# NCAI OCR Pipeline Instructions

This is the short handoff guide for setting up, running, reviewing, and exporting the OCR pipeline.

## 1. Setup

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Put input files in the configured input location. Supported source files are controlled by `pipeline.allowed_file_types` in `config/default.yaml` and currently include images plus PDFs.

## 2. Environment File

Create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Set these values:

```env
GEMINI_PROJECT_API=your-gemini-api-key
GEMINI_API_KEY=your-gemini-api-key
NCAI_CONFIG_PATH=config/default.yaml
```

Only one Gemini key is required. The code checks `GEMINI_API_KEY` first and then `GEMINI_PROJECT_API`. `NCAI_CONFIG_PATH` is optional if you pass `--config` in commands.

Do not commit `.env`.

## 3. Main Config Files

Update these before running:

- `config/default.yaml`: pipeline, NAS/local storage, provider, review, export, logging.
- `config/prompts.yaml`: OCR prompt text. `pipeline.prompt_version` must match a prompt key, for example `ocr_v1`.

Important `config/default.yaml` sections:

- `pipeline.storage_mode`: use `nas` for NAS processing or `local` for local folders.
- `pipeline.input_dir`: local input directory. Ignored when `storage_mode` is `nas`.
- `pipeline.output_root`: local output root for generated artifacts.
- `pipeline.skip_completed`: when `true`, completed documents are skipped on rerun.
- `pipeline.force_reprocess`: when `true`, behaves like always passing `--force`.
- `pipeline.dry_run`: creates artifacts without calling Gemini.
- `pipeline.pdf_render_scale`: controls PDF page render resolution.
- `pipeline.blank_page_detection`: controls empty-page skipping thresholds.
- `provider.model_name`: Gemini model used for OCR.
- `provider.timeout_seconds`, `retry_limit`, `rate_limit_per_minute`: live API behavior.
- `nas.input_root`, `nas.output_root`, `nas.scratch_root`: NAS source, final output, and local scratch paths.
- `nas.publish_strategy`: `local_scratch_create_only` creates versioned completed outputs; `local_scratch_atomic` updates stable output paths.
- `export.dataset_types`: `full`, `page_text`, `regions`, `regions_blocks`, `regions_sections`, `handwritten`.
- `export.output_formats`: `jsonl`, `json`, `csv`, `xlsx`.
- `export.region_source`: `blocks` or `sections` when exporting regions.
- `export.split_ratios`: train/val/test split.
- `export.handwritten`: automatic handwritten block detection and crop padding.

## 4. Run The Pipeline

Main command:

```powershell
python main.py run --config config/default.yaml --force
```

What it does:

1. Finds documents from `nas.input_root` in NAS mode, otherwise from `pipeline.input_dir`.
2. Hashes files into stable document IDs.
3. Converts images/PDF pages into normalized page images.
4. Calls Gemini using the prompt from `config/prompts.yaml`.
5. Stores raw requests, raw responses, parsed OCR, manifests, and review files.
6. Writes review artifacts for the interactive UI.

Useful variations:

```powershell
python main.py run --config config/default.yaml
python main.py run --config config/default.yaml --dry-run
python main.py run --config config/default.yaml --input-dir data/input
```

If pages or documents fail, rerun the same `run` command. Already processed documents/pages are skipped unless `--force` or `pipeline.force_reprocess: true` is used.

## 5. Review In The Web UI

Main command:

```powershell
python main.py interactive --ui web --config config/default.yaml
```

What it does:

1. Starts a local review server, normally at `http://127.0.0.1:8765`.
2. Shows the processing queue, latest run/export summaries, page images, raw responses, parsed OCR, and correction fields.
3. Lets the reviewer edit corrected content, verify pages, tag handwritten blocks, approve, or reject documents.

Useful variations:

```powershell
python main.py interactive --ui web --config config/default.yaml --document-id <document_id>
python main.py interactive --ui web --config config/default.yaml --host 127.0.0.1 --port 8765 --no-open-browser
```

The review source of truth is the generated `corrected.yaml` plus the review record for each document.

## 6. Export The Dataset

Main command:

```powershell
python main.py export --config config/default.yaml
```

What it does:

1. Reads approved review records and corrected outputs.
2. Skips unapproved, rejected, failed, and dry-run records by default.
3. Builds deterministic train/val/test splits.
4. Copies page assets and handwritten crops when needed.
5. Writes datasets and export manifests.

Useful variations:

```powershell
python main.py export --config config/default.yaml --dataset-type page_text --format jsonl
python main.py export --config config/default.yaml --dataset-type regions --region-source blocks --format csv
python main.py export --config config/default.yaml --dataset-type handwritten --format jsonl
python main.py export --config config/default.yaml --include-dry-run
```

For normal training data, approve documents in the web UI before exporting.

## 7. Quick Checks

```powershell
python main.py pipeline-status --config config/default.yaml --status all
python main.py pipeline-status --config config/default.yaml --status failed
python main.py pipeline-status --config config/default.yaml --status needs_retry
python main.py review-status --config config/default.yaml
```

Local outputs are under `data/`. In NAS mode, final outputs are under `nas.output_root`, with scratch work under `nas.scratch_root`.
