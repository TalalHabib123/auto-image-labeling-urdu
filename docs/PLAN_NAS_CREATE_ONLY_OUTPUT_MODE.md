# Plan: NAS Create-Only Output Mode

## Purpose

The current NAS mode was implemented around local scratch processing plus atomic publishing to the NAS output root. That design assumes the NAS account can create, rename, replace, and delete files or folders.

The tested NAS account can read input files and create folders/files in the output root, but rename/replace and cleanup operations fail. This plan describes how to let the pipeline use NAS input and NAS output with limited output permissions by keeping all mutable work inside local scratch and publishing only create-only, versioned final artifacts to the NAS.

This document is the implementation plan for the create-only NAS strategy and should remain the reference for follow-up refinements.

## Permission Model From The Test

Known working operations:

- Read/list/scan input root.
- Read sample PDF from input root.
- Create output folders.
- Create output files directly.
- Read created output files when the path exists.
- Create local scratch folders and files.
- Clean local scratch folders.
- Modify existing files (Read Writes in existing files working)

Known failing or unsafe operations:

- Rename temp file to final file on NAS.
- Replace an existing file on NAS.
- Delete/cleanup NAS folders reliably.
- Remove NAS staging folders reliably.
- Release or replace NAS directory locks if that requires delete or rename.

The important conclusion is that the NAS output root must be treated as append-only or create-only. Any design that depends on temp-to-final rename, overwrite, shared "latest" files, staging cleanup, or lock cleanup will keep failing.

## Current Failure Points

Current code paths that conflict with limited NAS permissions:

- `src/ncai_pipeline/storage/repository.py`
  - `write_json`, `write_yaml`, `write_text`, and `write_bytes` route through atomic temp files and `os.replace`.
  - `atomic_publish_tree` uses `shutil.move`.
  - `copy_tree(replace=True)` and `safe_remove_tree` delete existing targets.

- `src/ncai_pipeline/storage/nas.py`
  - `_assert_nas_capabilities` explicitly requires rename/replace and delete.
  - `publish_document_artifacts` creates NAS `.staging`, moves files into final paths, writes completion markers through atomic write, then tries to delete staging.
  - `publish_export_artifacts` follows the same staging, move, replace, cleanup pattern.
  - `_replace_file` uses `os.replace`.
  - scratch cleanup is local and should stay local, but failed scratch retention uses local moves and should remain under local scratch only.

- `src/ncai_pipeline/storage/locks.py`
  - Lock acquisition uses directory creation, which works, but metadata writes use atomic replace.
  - Lock release uses `shutil.rmtree`.
  - Stale lock replacement uses rename and delete.
  - With create-only NAS permissions, NAS locks can become permanent leftovers.

- `src/ncai_pipeline/pipeline/orchestrator.py`
  - NAS run currently calls `final_paths.ensure()`, then publishes documents through the rename-based NAS publisher.
  - `_write_run_summary` writes `latest_run.json`, which is a shared mutable file.
  - Failure manifests can be written to final NAS paths through the same replace-based repository.

- `src/ncai_pipeline/cli.py`
  - `_load_runtime` configures file logging at `working/status/pipeline.log`; appending to a shared NAS log is mutable output.
  - NAS export builds in scratch but publishes to stable final export paths through the rename-based publisher.

- `src/ncai_pipeline/interactive/service.py`
  - Review edits update `corrected.yaml`, `review_record.json`, reports, parsed pages, and manifests in place.
  - That will fail on NAS when files already exist.

## Design Shift

Add a second NAS publish mode:

```yaml
nas:
  publish_strategy: "local_scratch_create_only"
```

The existing strategy can remain:

```yaml
nas:
  publish_strategy: "local_scratch_atomic"
```

The two strategies have different contracts:

| Strategy | NAS permission required | Behavior |
| --- | --- | --- |
| `local_scratch_atomic` | create, write, rename, replace, delete | Current staging plus atomic finalization design |
| `local_scratch_create_only` | read and create only | Local scratch builds final artifacts, then copies them once to unique NAS paths and writes completion markers last |

In create-only mode, "atomic publish" is replaced with "manifested completeness":

- A file or folder existing on NAS is not proof that the artifact is complete.
- Readers trust only completed versions referenced by a valid completion marker.
- Completion markers are small create-only JSON files written last.
- If a copy fails midway, the partial files stay orphaned on NAS and are ignored forever.
- A retry uses a new run/version path, so it never needs to overwrite or delete the failed attempt.

## Proposed NAS Layout

Keep the familiar top-level layout, but make write targets versioned.

Document artifacts:

```text
<nas_output_root>/
  working/
    rendered_pages/<document_id>/<run_id>/page_0001.png
    raw_requests/<document_id>/<run_id>/page_0001.json
    raw_responses/<document_id>/<run_id>/page_0001.json
    parsed/<document_id>/<run_id>/page_0001.json
    manifests/<document_id>/<run_id>.manifest.json
    status/
      review_records/<document_id>/<run_id>.review_record.json
      documents/<document_id>/<run_id>.complete.json
      runs/<run_id>.json
  review/
    correction_files/<document_id>/<run_id>.corrected.yaml
    reports/<document_id>/<run_id>.report.html
```

Export artifacts:

```text
<nas_output_root>/
  exports/
    runs/<export_run_id>/
      datasets/<dataset_id>/<format>/train.<ext>
      datasets/<dataset_id>/<format>/val.<ext>
      datasets/<dataset_id>/<format>/test.<ext>
      assets/pages/<document_id>/page_0001.png
      assets/crops/handwritten/<document_id>/page_0001/<block_id>.png
      export_manifest.json
    completed/<export_run_id>.complete.json
```

Do not write these mutable stable files in create-only mode:

- `working/status/latest_run.json`
- `exports/export_manifest.json`
- `exports/gemma_multimodal/train.jsonl` as a shared stable alias
- `working/manifests/<document_id>/manifest.json`
- `working/status/<document_id>/review_record.json`
- `review/correction_files/<document_id>/corrected.yaml`

Instead, load the latest valid version by scanning completion markers.

Export run names should be readable, for example:

```text
exports/runs/export_20260505_114522_full-page_text_jsonl-csv_1a2b3c4d/
```

Dataset records and manifest file references should use portable paths relative to the export run folder. Examples:

- `assets/pages/<document_id>/page_0001.png`
- `assets/crops/handwritten/<document_id>/page_0001/<block_id>.png`
- `datasets/<dataset_id>/<format>/train.<ext>`

When `handwritten` is the only selected dataset type, export only crop assets and omit full-page `assets/pages/...` files. If `full`, `page_text`, or region datasets are selected alongside `handwritten`, page assets are exported once for those dataset records.

## End-To-End Run Flow

### 1. Startup

1. Load config.
2. Resolve NAS runtime as today.
3. If `publish_strategy` is `local_scratch_create_only`, run the NAS output create/read probe only when `nas.healthcheck.require_writable_output` is true:
   - input root exists, is listable, and sample reads work
   - output root exists or can be created
   - output can create a unique directory
   - output can create and read back a unique final file
   - do not test rename, replace, or delete as required capabilities
   - do not fail if cleanup of the healthcheck folder is denied
4. Always verify local scratch is writable because temp files, atomic writes, and cleanup happen there.
5. Configure file logging under local scratch, not under the NAS output root.
6. Create local run scratch under `nas.scratch_root/active/<run_id>/`.

### 2. Discovery

1. Scan `nas.input_root` as today.
2. Keep input readiness checks.
3. Preserve source metadata in `DocumentManifest`.
4. Determine skip eligibility through the create-only NAS output index:
   - list `working/status/documents/<document_id>/*.complete.json`
   - parse valid completion markers
   - load the referenced manifest
   - skip only if source hash/version matches and required artifacts exist

### 3. Processing

Processing should remain local:

1. Copy source file from NAS input to local scratch.
2. Verify source hash before and after copy.
3. Render pages into local scratch.
4. Store raw requests, raw responses, parsed JSON, correction YAML, review HTML, review record, and manifest in local scratch.
5. Prepare final paths in the scratch metadata so `ReviewRecord` points to versioned NAS paths, not stable mutable paths.

### 4. Publish

For each document:

1. Build a publish plan in scratch containing source files, final NAS paths, sizes, and SHA256 values.
2. Ensure every final NAS target includes `run_id` or a unique publish id.
3. Fail fast if any planned target already exists.
4. Copy each scratch file directly to its final versioned NAS path using exclusive create.
5. Read back each copied NAS file and verify size/checksum when enabled.
6. Write the versioned manifest file directly to:

```text
working/manifests/<document_id>/<run_id>.manifest.json
```

7. Write the completion marker last:

```text
working/status/documents/<document_id>/<run_id>.complete.json
```

8. Do not move, rename, replace, or delete anything on NAS.
9. If publish fails before the completion marker, leave partial files in place and mark the document `needs_retry` in the run summary only.

### 5. Run Summary

In create-only mode:

1. Write only the immutable run summary:

```text
working/status/runs/<run_id>.json
```

2. Do not update `latest_run.json`.
3. CLI and UI should compute latest run by sorting `working/status/runs/*.json`.

The UI already has fallback logic for missing `latest_run.json`, so this is aligned with the current direction.

## Export Flow

1. Acquire no NAS delete-based export lock in create-only mode.
2. Require one export writer at a time for the first implementation, or add a create-only claim/index later.
3. Build datasets and assets in local scratch as today.
4. Use a readable export run id: `export_<timestamp>_<dataset_ids>_<formats>_<short_id>`.
5. Store dataset media paths and manifest file references relative to the export run folder.
6. Publish only export datasets, assets, and the run-level manifest to `exports/runs/<export_run_id>/...` using exclusive create.
7. Do not publish scratch-only curated/dry-run folders, per-format mini manifests, or legacy `gemma_multimodal` aliases in create-only mode.
8. For handwritten-only exports, publish crop assets only and omit full-page page assets.
9. Verify all copied files from NAS.
10. Write `exports/runs/<export_run_id>/export_manifest.json`.
11. Write `exports/completed/<export_run_id>.complete.json` last.
12. Update export summary readers to choose the newest valid completion marker instead of `exports/export_manifest.json`.

## Review And Approval Flow

The current review flow edits files in place, so it is not compatible with create-only NAS output.

Create-only review support should use revision files:

```text
review/correction_files/<document_id>/revisions/<revision_id>.corrected.yaml
working/status/review_records/<document_id>/revisions/<revision_id>.review_record.json
review/reports/<document_id>/revisions/<revision_id>.report.html
working/status/review_revisions/<document_id>/<revision_id>.complete.json
```

Review save behavior:

1. Load latest completed document version.
2. Load latest completed review revision if one exists.
3. Save edits to local scratch.
4. Publish a new review revision to unique NAS paths.
5. Write a review revision completion marker last.
6. Never overwrite the original correction YAML or review record.

Approval behavior:

1. Approval is also a new review revision.
2. Export loads the latest completed approved revision.
3. Manifest pipeline status can be derived from latest review revision instead of rewriting the original manifest.

If review edits do not need to be saved back to NAS in the first implementation, this can be deferred. The pipeline run and export plan should still avoid in-place writes.

## Locking Policy

Directory locks are not suitable for create-only NAS permissions because release requires delete and stale lock repair requires rename/delete.

Initial create-only mode should choose one of these policies:

1. Single-writer mode, recommended for the first implementation.
   - Disable NAS document, status, and export locks.
   - Document completion markers provide idempotent skip behavior.
   - Operator must not run two writers against the same output root at the same time.

2. Create-only claim files, later enhancement.
   - Writers create immutable claim files.
   - A deterministic election rule decides which claim owns the document.
   - This avoids delete but is weaker than true locks and needs careful testing.

If true multi-PC write safety is required, the NAS account needs modify/delete permissions or the project needs an external coordinator such as SQLite/Postgres/Redis.

## Repository Changes Needed

Add create-only helpers next to the existing atomic helpers:

- `create_text(path, data)`
- `create_json(path, data)`
- `create_yaml(path, data)`
- `create_bytes(path, data)`
- `copy_file_create_only(source, target)`
- `copy_tree_create_only(source, target)`

Rules for these helpers:

- Open targets with exclusive create semantics.
- Never create temp siblings on NAS.
- Never call `os.replace`, `Path.rename`, `shutil.move`, `unlink`, or `rmtree`.
- Flush and close files before verification.
- When a target exists, raise a clear create-only collision error.
- Verification reads the final file back from NAS and compares size/checksum.

The existing atomic helpers should stay for local mode and full-permission NAS mode.

## Code Areas To Change

### `src/ncai_pipeline/config.py`

- Expand `nas.publish_strategy` to include `local_scratch_create_only`.
- Add output capability or healthcheck settings for create-only mode.
- Validate that create-only mode disables or changes lock behavior unless explicitly overridden.

### `config/default.yaml`

For this NAS account, use:

```yaml
nas:
  publish_strategy: "local_scratch_create_only"
  healthcheck:
    enabled: true
    require_writable_output: false
  locks:
    enabled: false
```

With `require_writable_output: false`, startup verifies the NAS output directory exists but does not create a healthcheck probe on NAS. Scratch is still tested for full local write/rename/delete behavior.

### `src/ncai_pipeline/storage/repository.py`

- Keep current atomic write helpers.
- Add create-only write/copy helpers.
- Make PermissionError messages distinguish full atomic mode from create-only mode.

### `src/ncai_pipeline/storage/nas.py`

- Add optional create-only output healthcheck.
- Add `publish_document_artifacts_create_only`.
- Add `publish_export_artifacts_create_only`.
- Add completion marker discovery helpers.
- Add checksum/read-back verification for directly-created final files.
- Avoid NAS `.staging` entirely in create-only mode.
- Publish create-only exports with readable run ids, relative dataset paths, one run-level manifest, and no scratch-only curated/dry-run folders.

### `src/ncai_pipeline/storage/paths.py`

- Keep `DataPaths` for local and scratch paths.
- Add a small versioned NAS path helper rather than overloading every `DataPaths` property.
- The helper should produce final versioned paths for a document/run and export/run.

### `src/ncai_pipeline/storage/locks.py`

- Keep existing locks for full-permission NAS mode.
- In create-only mode, bypass locks or route to a future create-only claim system.

### `src/ncai_pipeline/pipeline/orchestrator.py`

- Branch publish behavior by `nas.publish_strategy`.
- In create-only mode, process documents in scratch exactly as today.
- Prepare review records and manifests with versioned NAS final paths.
- Use completion marker index for skip decisions.
- Write only immutable run summaries.
- Do not write `latest_run.json`.
- Do not write failed document manifests to stable NAS locations.

### `src/ncai_pipeline/cli.py`

- Configure NAS create-only logs in local scratch.
- Export through the create-only export publisher.
- Display latest run/export by marker/index discovery.

### `src/ncai_pipeline/interactive/service.py`

- Read documents through the create-only output index.
- Load latest completed document version and latest review revision.
- If write support is included, save edits as new review revisions instead of overwriting.

### `src/ncai_pipeline/export/dataset_builder.py`

- Build in scratch as today.
- Make export input discovery use latest approved review revisions.
- Publish export outputs to versioned create-only export run directories.

## Required Reader Changes

Create a `NasOutputIndex` or similar helper so readers do not hardcode old stable paths.

Responsibilities:

- List valid document completion markers.
- Load the latest valid document version per `document_id`.
- Resolve manifest, review record, correction YAML, report, rendered pages, raw responses, and parsed files for that version.
- List valid review revisions and choose latest revision.
- List valid export completion markers and choose latest export.
- Ignore corrupt or partial marker files.
- Ignore versions whose required files are missing or checksum validation fails.

This helper should be used by:

- rerun skip logic
- `pipeline-status`
- `review-status`
- interactive UI
- approval
- export

## Test Plan

Add tests with a fake create-only repository or filesystem wrapper that raises on rename, replace, delete, and overwrite.

Core tests:

- Optional create-only healthcheck passes when direct create/read works and rename/delete fail.
- `FileRepository.create_json` writes directly without temp files.
- Create-only publish never calls `os.replace`, `rename`, `move`, `unlink`, or `rmtree`.
- Partial publish without completion marker is ignored by `NasOutputIndex`.
- Retry after partial publish creates a new version path and succeeds.
- Rerun skips a document only when a valid completion marker exists.
- Run summary writes `working/status/runs/<run_id>.json` and does not require `latest_run.json`.
- Export writes under `exports/runs/<export_run_id>/` and completion marker is last.
- Export dataset rows use relative `assets/...` media paths, including handwritten `crop_path`.
- Create-only export runs do not include scratch-only `curated/`, per-format mini manifests, or legacy alias folders.
- UI latest run/export discovery works without `latest_run.json` or `exports/export_manifest.json`.
- Existing local mode tests still pass.
- Existing `local_scratch_atomic` NAS tests still pass for full-permission temp directories.

Manual NAS verification:

1. Run the existing NAS access test and confirm rename/delete failures are tolerated in create-only mode.
2. Run `python main.py run --config config/default.yaml` against a new empty NAS output folder.
3. Confirm document completion markers exist under `working/status/documents/<document_id>/`.
4. Confirm no NAS `.tmp-*` files are created by the pipeline publisher.
5. Confirm no NAS `.staging` publish folder is used.
6. Run the command again and confirm completed documents are skipped by marker discovery.
7. Approve/review only after review revision support is implemented, or keep review local for the first milestone.
8. Run export and confirm outputs appear under `exports/runs/<export_run_id>/`.

## Implementation Phases

### Phase 1: Create-Only Storage Primitives

- Add config option.
- Add create-only repository helpers.
- Add create-only healthcheck.
- Add tests that simulate rename/delete denial.

### Phase 2: Create-Only Document Publish

- Keep all document processing in local scratch.
- Publish versioned document artifacts with exclusive create.
- Write document completion marker last.
- Load/skip documents through completion marker index.
- Write immutable run summaries only.

### Phase 3: Create-Only Export Publish

- Build exports in scratch.
- Publish versioned export runs.
- Write export completion marker last.
- Update export summary readers.

### Phase 4: Review Revisions

- Save review edits and approvals as new versioned revision artifacts.
- Export latest approved revision.
- Add optimistic conflict detection based on loaded revision id.

### Phase 5: Operational Cleanup And Repair

- Add an inspect command for orphaned partial publish attempts.
- Do not delete orphaned NAS artifacts from the pipeline account.
- Optionally generate a cleanup report for a NAS admin account with delete permissions.

## Acceptance Criteria

- The pipeline can read from NAS input and process in local scratch.
- The pipeline can publish final artifacts to NAS output without rename, replace, or delete.
- No temp files are created on NAS for final JSON/YAML/text writes.
- No NAS staging cleanup is required.
- Existing partial NAS files are never overwritten.
- A partial publish is not treated as complete.
- Reruns can skip completed documents and retry incomplete ones.
- Run and export summaries work without mutable `latest` files.
- Local mode remains unchanged.
- Full-permission NAS mode can remain available for environments that support atomic rename/delete.

## Decisions Needed Before Implementation

- Is single-writer create-only mode acceptable for the first implementation?
- Should interactive review edits be part of the first create-only implementation, or can they be a second phase?
- Is a versioned NAS layout acceptable, even though stable paths such as `review/correction_files/<document_id>/corrected.yaml` will not be used in create-only mode?
- Should old orphaned partial publishes simply accumulate, or should we create an admin cleanup report format?
