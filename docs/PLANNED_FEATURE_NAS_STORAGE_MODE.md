# Planned Feature: NAS-Backed Pipeline Mode

## Summary

This document defines a future feature for running the NCAI OCR pipeline against documents stored on a NAS-mounted filesystem share while preserving the current local workflow.

The current pipeline assumes both input documents and output artifacts live on the local machine, usually under `data/`. The planned NAS mode will allow any PC on the same network to scan documents from a mounted NAS share, process them through the same ingestion, extraction, review, curation, and export stages, and publish all final artifacts back to the NAS.

The first NAS implementation should target mounted filesystem paths only, such as SMB, NFS, UNC paths, or mapped drives. Remote protocols such as SFTP, WebDAV, object storage, or NAS vendor APIs are out of scope for the first implementation.

The default behavior must remain local mode. NAS behavior should only activate when configured explicitly.

NAS mode must not change the core pipeline architecture. The NAS acts as centralized shared storage. The machine running Python acts as the compute worker. Heavy processing should happen in a local scratch workspace, and only complete validated artifacts should be published back to the NAS output root.

## Current Pipeline Path Model

The repository already has a path-oriented design that can be extended for NAS mode.

Important current modules:

- `config/default.yaml` defines `pipeline.input_dir` and `pipeline.output_root`.
- `src/ncai_pipeline/config.py` loads Pydantic settings for pipeline, provider, review, export, and logging.
- `src/ncai_pipeline/storage/paths.py` owns the standard artifact layout through `DataPaths`.
- `src/ncai_pipeline/storage/repository.py` centralizes JSON, YAML, and text reads/writes, but many modules still use direct `Path`, `glob`, `exists`, `mkdir`, `shutil`, and byte reads.
- `src/ncai_pipeline/ingestion/scanner.py` recursively scans input documents with `input_dir.rglob("*")`.
- `src/ncai_pipeline/pipeline/orchestrator.py` coordinates discovery, normalization, provider calls, parsing, review artifact creation, manifests, status files, and run summaries.
- `src/ncai_pipeline/export/dataset_builder.py` builds approved datasets and currently copies page assets and crops through direct filesystem operations.
- `src/ncai_pipeline/interactive/service.py` and `src/ncai_pipeline/interactive/web.py` read manifests, correction files, rendered pages, reports, and export summaries from configured `DataPaths`.
- CLI commands in `src/ncai_pipeline/cli.py` construct `DataPaths(Path(config.pipeline.output_root))` and therefore currently assume one output root.

Current local artifact layout:

```text
data/
  input/
  working/
    manifests/
    rendered_pages/
    raw_requests/
    raw_responses/
    parsed/
    status/
  review/
    reports/
    correction_files/
  curated/
    approved/
    rejected/
    dry_run/
  exports/
    assets/
    datasets/
    gemma_multimodal/
```

The planned feature should preserve this layout under the NAS output root, while using a separate local scratch root for temporary processing.

## Target NAS Workflow

NAS mode should use a local scratch workspace and publish complete artifacts to the NAS.

High-level flow:

1. Load config and detect `pipeline.storage_mode`.
2. If `storage_mode` is `local`, keep the current behavior unchanged.
3. If `storage_mode` is `nas`, validate mounted NAS paths before processing.
4. Healthcheck `nas.input_root`, `nas.output_root`, and `nas.scratch_root`.
5. Scan source documents recursively from `nas.input_root`.
6. Apply input readiness checks before processing any document.
7. For each document, acquire a document lock on the NAS output root.
8. Copy the source document, or the minimum required page work unit, into local scratch.
9. Verify that the copied source is stable and matches the expected source hash.
10. Render PDFs, normalize images, call the provider, parse responses, and build review artifacts locally.
11. Publish finished artifacts atomically from local scratch to a staging folder under `nas.output_root`.
12. Validate staged artifacts.
13. Move staged artifacts into their final NAS locations.
14. Write the final document manifest or completion marker last.
15. Release the document lock.
16. Write run summaries and status manifests to the NAS output root.

NAS mode should not stream all intermediate operations directly against the NAS. Slow network IO, temporary disconnects, and partial writes are expected risks. Local scratch processing keeps provider calls, PDF rendering, image crop generation, and artifact assembly more reliable.

NAS mode v1 supports safe independent runs from multiple PCs. Coordinated distributed processing under one shared run ID is out of scope unless a central queue or shared job index is added later.

## Proposed Config Schema

Add storage mode to the existing pipeline config:

```yaml
pipeline:
  storage_mode: "local" # local or nas
  input_dir: "data/input"
  output_root: "data"
```

Add a new top-level `nas` section:

```yaml
nas:
  input_root: "\\\\NAS-SERVER\\documents\\input"
  output_root: "\\\\NAS-SERVER\\documents\\ncai_output"
  scratch_root: "data/nas_scratch"
  share_label: "office_nas_documents"
  publish_strategy: "local_scratch_atomic"

  healthcheck:
    enabled: true
    timeout_seconds: 10
    require_writable_output: true

  retry:
    attempts: 3
    backoff_seconds: 2.0
    max_backoff_seconds: 30.0

  locks:
    enabled: true
    stale_after_minutes: 240
    heartbeat_seconds: 60

  timeouts:
    read_seconds: 120
    write_seconds: 120
    publish_seconds: 300

  input_readiness:
    enabled: true
    stable_check_seconds: 5
    require_ready_marker: false
    ready_marker_suffix: ".ready"

  scratch:
    cleanup_on_success: true
    keep_failed_runs_days: 14
    keep_completed_runs_days: 3

  checksums:
    verify_after_copy: true
    verify_after_publish: true
```

Local mode example:

```yaml
pipeline:
  storage_mode: "local"
  input_dir: "data/input"
  output_root: "data"
```

NAS mode example:

```yaml
pipeline:
  storage_mode: "nas"
  input_dir: "data/input"      # ignored for run discovery in NAS mode
  output_root: "data"          # ignored for final artifacts in NAS mode

nas:
  input_root: "\\\\NAS-SERVER\\NCAI\\input"
  output_root: "\\\\NAS-SERVER\\NCAI\\output"
  scratch_root: "data/nas_scratch"
  share_label: "ncai_office_nas"
  publish_strategy: "local_scratch_atomic"
```

Mapped drive example for manual desktop runs:

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

Windows guidance:

- UNC paths are recommended for production Windows runs.
- Mapped drives are allowed for manual desktop runs.
- Mapped drives may fail under Windows services, scheduled tasks, different user accounts, remote sessions, or CI runners.

Validation rules:

- `pipeline.storage_mode` must be `local` or `nas`.
- In `local` mode, `pipeline.input_dir` and `pipeline.output_root` remain required.
- In `nas` mode, `nas.input_root`, `nas.output_root`, and `nas.scratch_root` are required.
- `nas.publish_strategy` should initially support only `local_scratch_atomic`.
- Retry counts and timeout values must be positive.
- NAS input and output roots must not point to the same directory.
- NAS input root must not be inside NAS output root.
- NAS output root must not be inside NAS input root.
- Scratch root must not be inside NAS input root.
- Scratch root must not be inside NAS output root.
- Scratch root should be local storage, not a NAS path.
- Scanner must ignore NAS output, staging, lock, scratch, hidden, and system folders.
- Staging folders must be inside the same NAS output root as the final destination.
- Atomic finalization must only rely on rename/move operations within the same NAS output filesystem.
- Cross-device moves must not be treated as atomic.
- Timeout values must be treated carefully. Low-level filesystem calls may not be interruptible on all platforms, so timeout enforcement may need operation-level or stage-level wrappers.

## Storage Architecture Changes

The implementation should extend the storage layer instead of spreading NAS-specific path logic across orchestration, export, review, and UI code.

Planned additions:

- Add `NasSettings`, `NasHealthcheckSettings`, `NasRetrySettings`, `NasLockSettings`, `NasTimeoutSettings`, `NasInputReadinessSettings`, `NasScratchSettings`, and `NasChecksumSettings` models in `config.py`.
- Add `storage_mode` to `PipelineSettings`.
- Add a runtime storage resolver that returns:
  - effective input root
  - effective output root
  - local scratch root
  - final `DataPaths`
  - scratch `DataPaths`
  - storage metadata for manifests
- Extend `FileRepository` with reliable filesystem helpers:
  - `exists`
  - `glob`
  - `read_bytes`
  - `write_bytes`
  - `copy_file`
  - `copy_tree`
  - `atomic_write_json`
  - `atomic_write_yaml`
  - `atomic_write_text`
  - `atomic_write_bytes`
  - `atomic_publish_tree`
  - `verify_checksum`
  - `safe_remove_tree`
- Add NAS-specific helper functions for:
  - healthchecks
  - retry wrappers
  - input readiness checks
  - staging folder cleanup
  - scratch cleanup
  - stale lock cleanup
  - publish validation

Repository migration should prioritize correctness-critical writes first.

Must migrate first:

- manifest writes
- correction file writes
- parsed output writes
- status file writes
- approval writes
- export writes
- completion marker writes

Can remain direct temporarily where practical:

- read-only page image serving
- simple local scratch reads
- simple glob reads
- temporary local files
- non-critical local-only operations

Recommended new files:

```text
src/ncai_pipeline/storage/runtime.py
src/ncai_pipeline/storage/nas.py
src/ncai_pipeline/storage/locks.py
```

Suggested responsibilities:

- `runtime.py`: choose local or NAS runtime paths from config.
- `nas.py`: healthchecks, retry wrappers, input readiness checks, scratch cleanup policy, staging cleanup, checksum verification, and atomic publish helpers.
- `locks.py`: run, document, export, and status lock creation, stale lock handling, heartbeat updates, and lock release.

## Input Readiness Rules

NAS scanner must avoid processing files that are still being copied or modified.

A file is eligible for processing only if:

- the file extension is supported
- the path is not under output, staging, lock, scratch, hidden, or system folders
- the file size remains stable across `nas.input_readiness.stable_check_seconds`
- the modified timestamp remains stable across `nas.input_readiness.stable_check_seconds`
- if `require_ready_marker` is enabled, a matching marker file exists

Ready marker example:

```text
input/scan_001.pdf
input/scan_001.pdf.ready
```

If ready marker mode is disabled, the stable size and timestamp check is still required.

If a file fails readiness checks, the scanner should skip it for the current pass and log a clear reason.

Possible statuses:

```text
input_not_ready
input_unstable
input_marker_missing
```

Recommended operator workflow:

```text
incoming/
ready/
```

Users may copy files into `incoming/` and move them into `ready/` once the copy is complete. The scanner should be configured to scan only the ready input root if this convention is used.

## Manifest And Nested Document Handling

The scanner already scans nested documents recursively. NAS mode should preserve this behavior and store enough path metadata for traceability.

Extend `DocumentManifest` with optional fields:

```yaml
storage_mode: "local | nas"
source_root: "Z:/NCAI/input"
source_relative_path: "department_a/forms/scan_001.pdf"
source_size_bytes: 123456
source_mtime_utc: "..."
source_sha256: "..."
scratch_source_path: "data/nas_scratch/<run_id>/sources/<document_id>/scan_001.pdf"
scratch_source_sha256: "..."
nas_share_label: "ncai_z_drive"
published_at: "..."
artifact_version: 1
completion_marker_path: "working/status/documents/<document_id>.complete.json"
```

Rules:

- `source_path` remains the path used for discovery and traceability.
- `source_relative_path` must be relative to the configured input root.
- Nested source folders should not be flattened in metadata.
- Stable document IDs should remain hash-based.
- For duplicate filenames in different nested folders, manifests must still be traceable through `source_relative_path`.
- The initial implementation can keep the existing `stable_document_id(path.name, sha256)` behavior because file hash prevents most collisions, but the manifest must preserve the relative path for human debugging.
- The source file should be hashed before copy and after copy when checksum verification is enabled.
- If source hash and scratch hash differ, the document should be rejected for the current run and marked `needs_retry` with `nas_read_failed` or `input_unstable`.

## Completion Marker Rules

A document or export must not be treated as complete just because a folder exists.

A document is complete only if:

- final manifest exists
- manifest status is `completed`
- required artifacts exist
- completion marker exists
- optional checksum validation passes when enabled

Suggested document completion marker:

```text
<nas_output_root>/working/status/documents/<document_id>.complete.json
```

Suggested content:

```json
{
  "document_id": "...",
  "run_id": "...",
  "status": "completed",
  "published_at": "...",
  "artifact_version": 1,
  "required_artifacts": [
    "working/manifests/<document_id>.json",
    "review/correction_files/<document_id>.yaml"
  ],
  "checksums_verified": true
}
```

For exports, `export_manifest.json` should be written last and should act as the export completion marker.

Rerun logic should skip a NAS-published document only if the completion rules pass. If the manifest exists but the completion marker is missing, the document should be treated as incomplete and eligible for repair or retry.

## End-to-End Behavior By Command

### `run`

Local mode:

- Keep current behavior.
- Scan `pipeline.input_dir`.
- Write artifacts under `pipeline.output_root`.

NAS mode:

- Healthcheck `nas.input_root`, `nas.output_root`, and `nas.scratch_root`.
- Scan documents from `nas.input_root`.
- Apply input readiness checks before processing.
- Create run scratch directory under `nas.scratch_root/<run_id>/`.
- Use scratch `DataPaths` while processing documents.
- Acquire document lock before copying a source document into scratch.
- Copy source document to scratch.
- Verify source stability and checksum when enabled.
- Process the scratch copy through normalization, provider calls, parsing, review artifact creation, and manifest generation.
- Publish document artifacts from scratch to NAS staging under `nas.output_root/.staging/<run_id>/<document_id>/`.
- Validate staged artifacts.
- Move staged artifacts into final NAS `DataPaths`.
- Write the final document manifest and completion marker last.
- Write final run manifests under `nas.output_root/working/status/`.
- Update `latest_run.json` last.
- On NAS failures, classify documents as `needs_retry` where possible.
- One failed NAS document should not stop the rest of the batch unless the NAS root itself becomes unavailable.

### `interactive --ui web` and `interactive --ui tui`

Local mode:

- Keep current behavior.

NAS mode:

- Load manifests, review records, correction files, rendered pages, reports, and export summaries from `nas.output_root`.
- Display NAS source paths and relative source paths in the dashboard where useful.
- Serve page images from NAS-published rendered page artifacts.
- Show a clear error if a NAS artifact is unavailable.
- Continue to save review edits and approvals into the configured output root.
- Use document locks for correction file writes.
- Use optimistic concurrency checks when saving review edits.

Review save conflict handling:

- Store the checksum or version of the correction file when a reviewer loads it.
- Before saving, compare the latest correction file checksum against the loaded checksum.
- If the file changed since it was opened, reject the save and ask the reviewer to reload.
- Store `last_modified_at`, `last_modified_by`, `host`, or `username` metadata when available.

### `approve`

Local mode:

- Keep current behavior.

NAS mode:

- Load correction files from `nas.output_root/review/correction_files/`.
- Acquire document lock before editing correction files or synced manifest state.
- Check correction file version or checksum before saving.
- Write approval changes and synced manifest state back to NAS output.
- Release document lock after write.
- If a lock conflict occurs, return a clear `nas_lock_conflict` error.

### `pipeline-status` and `review-status`

Local mode:

- Keep current behavior.

NAS mode:

- Read manifests and review records from `nas.output_root`.
- Include NAS error states and `needs_retry` counts.
- Include `input_not_ready`, `input_unstable`, and `input_marker_missing` counts where available.
- Do not rely only on `latest_run.json`.
- If `latest_run.json` is missing or stale, fall back to listing run manifests under `working/status/runs/`.

Recommended status layout:

```text
working/status/
  latest_run.json
  runs/
    <run_id>.json
  documents/
    <document_id>.complete.json
```

### `export`

Local mode:

- Keep current behavior.

NAS mode:

- Load approved correction files from `nas.output_root`.
- Acquire `export.lock`.
- Build datasets and assets in local scratch.
- Publish completed datasets, page assets, crop assets, export manifests, and legacy Gemma JSONL aliases to a staging export directory under `nas.output_root/.staging/`.
- Validate expected split files, assets, and `export_manifest.json`.
- Move validated export artifacts into final `nas.output_root/exports/` locations.
- Write `exports/export_manifest.json` last.
- Release `export.lock`.
- Use atomic publish so partial exports are not mistaken for complete exports.

### `dry-run-artifacts`

Local mode:

- Keep current behavior.

NAS mode:

- Discover dry-run artifacts from NAS output.
- For delete or keep operations, use lock-aware repository operations.
- If rebuilding export, build locally and publish completed export artifacts back to NAS.
- Avoid deleting active scratch, active staging, or active lock folders.
- Provide clear warnings before removing stale staging folders.

## Failure Modes And Mitigations

| Failure Mode | Risk | Mitigation | Error Category |
| --- | --- | --- | --- |
| NAS share unavailable at startup | Run cannot find input or output root | Startup healthcheck fails fast with clear message | `nas_unavailable` |
| NAS disconnects during scan | Partial discovery or read failure | Retry scan/read operations and mark run failed if root stays unavailable | `nas_read_failed` |
| Source file is still being copied | Partial PDF/image may be processed | Stable size/mtime check and optional ready marker | `input_not_ready` |
| Source file changes during processing | Hash mismatch or inconsistent page render | Copy to scratch, hash before and after copy, reject if unstable | `input_unstable` |
| Ready marker missing | File should not be processed yet | Skip file and report readiness status | `input_marker_missing` |
| Source file read is slow | Long stalls during hashing or copy | Configured stage timeout and retry budget where technically enforceable | `nas_timeout` |
| Output write fails | Partial artifacts on NAS | Write to temp path, then atomic rename inside same NAS output root | `nas_write_failed` |
| Publish fails midway | Mixed old/new output | Publish into staging folder under same NAS output root and finalize only after validation | `nas_publish_failed` |
| Cross-device move attempted | Finalization may not be atomic | Reject config or fall back to copy plus validation plus final marker | `nas_publish_failed` |
| Two PCs process same document | Conflicting manifests and review files | Atomic document lock under NAS output root | `nas_lock_conflict` |
| Lock remains after crash | Document blocked forever | Stale lock detection using timestamp, heartbeat, and host metadata | `nas_lock_conflict` |
| Two reviewers edit same correction file | Later save may overwrite earlier save | Optimistic concurrency using checksum/version check | `review_conflict` |
| Status files updated by multiple runs | Stale or overwritten status index | Per-run manifests and optional `status.lock` for shared index updates | `nas_lock_conflict` |
| Web UI loses NAS access | Broken image/report loads | Return clear 404/503-style message in UI instead of crashing | `nas_unavailable` |
| Export crop generation reads missing page | Incomplete handwritten dataset | Count skipped crops and report in export manifest | existing skipped counts plus `nas_read_failed` |
| Scratch grows forever | Local disk fills over time | Cleanup successful scratch and retain failed scratch for a configured period | `scratch_cleanup_failed` |
| Staging folders remain after crash | Confusing incomplete artifacts | Repair or cleanup command for stale staging folders | `nas_publish_failed` |

Network-aware error categories to add:

```text
nas_unavailable
nas_timeout
nas_read_failed
nas_write_failed
nas_publish_failed
nas_lock_conflict
input_not_ready
input_unstable
input_marker_missing
review_conflict
scratch_cleanup_failed
```

Provider/API retry behavior should stay separate from NAS retry behavior. Gemini timeouts and rate limits should remain provider errors, while NAS read/write/publish failures should be storage errors.

## Atomic Publish Strategy

The first publish strategy should be `local_scratch_atomic`.

Important rule:

Atomic finalization must happen inside the same NAS output root. Staging directories must live under the same NAS share and filesystem tree as the final destination. Cross-device moves, local-to-NAS moves, or moves between different NAS shares must not be treated as atomic.

For each document:

1. Build document artifacts under scratch:

```text
<scratch_root>/<run_id>/documents/<document_id>/
```

2. Create a NAS staging directory under the same NAS output root:

```text
<nas_output_root>/.staging/<run_id>/<document_id>/
```

3. Copy completed scratch artifacts to staging.
4. Validate required files exist in staging.
5. Optionally verify staged file checksums.
6. Move files or directories into final NAS locations.
7. Write or update the final manifest last.
8. Write the completion marker last.
9. Remove staging on success.

For run-level status:

- Write run-specific status to a temp file.
- Rename temp file into final status path.
- Prefer per-run manifests under:

```text
<nas_output_root>/working/status/runs/<run_id>.json
```

- Update `latest_run.json` last.
- If multiple runs may update shared indexes, protect shared index updates with `status.lock`.

For export:

- Build the full export in scratch.
- Publish to a staging export directory under:

```text
<nas_output_root>/.staging/<run_id>/exports/
```

- Validate expected split files, assets, and `export_manifest.json`.
- Optionally verify staged file checksums.
- Move into final `exports/` locations.
- Update top-level `exports/export_manifest.json` last.
- Treat `exports/export_manifest.json` as the export completion marker.

## Locking Strategy

Add simple filesystem locks in the NAS output root.

Lock acquisition must be atomic. Existence-check-then-create is not acceptable because two PCs can pass the check at the same time.

Preferred implementation:

Use directory-based locks, because creating a directory is generally atomic on mounted filesystems.

Suggested layout:

```text
<nas_output_root>/working/locks/
  run_<run_id>.lock/
    lock.json
  document_<document_id>.lock/
    lock.json
  export.lock/
    lock.json
  status.lock/
    lock.json
```

Alternative implementation:

Use exclusive file creation, such as opening the lock file in exclusive create mode.

Conceptual behavior:

```text
create lock atomically
if lock already exists:
  inspect lock metadata
  if stale:
    replace lock safely
  else:
    report lock conflict
```

Lock file content:

```json
{
  "lock_type": "document",
  "run_id": "...",
  "document_id": "...",
  "host": "...",
  "pid": 1234,
  "created_at": "...",
  "updated_at": "...",
  "stale_after_minutes": 240
}
```

Rules:

- Document processing should acquire a document lock before copying the source into scratch.
- Review and approval writes should acquire a document lock when editing correction files or synced manifests.
- Export should acquire a single export lock.
- Shared status index updates should acquire a `status.lock` if multiple runs may update shared files.
- Lock acquisition must be atomic.
- If a lock exists and is not stale, skip or mark the document as `needs_retry` with `nas_lock_conflict`.
- If a lock is stale, replace it and record the stale lock event in logs.
- Locks should update `updated_at` periodically for long-running operations.
- Lock release should be best-effort and should not delete another process's newer lock.
- A stale lock should only be replaced after validating host, pid, timestamp, and heartbeat age where possible.

## Scratch Workspace Policy

NAS mode should use local scratch for source copies, rendered pages, normalized images, provider payloads, parsed outputs, review artifacts, and export assembly.

Suggested scratch layout:

```text
<scratch_root>/
  active/
    <run_id>/
  failed/
    <run_id>/
  completed/
    <run_id>/
```

Rules:

- Scratch root should be on local disk, not NAS.
- Successful scratch runs may be deleted after publish if `cleanup_on_success` is true.
- Failed scratch runs should be retained for debugging for `keep_failed_runs_days`.
- Completed scratch runs may be retained briefly for audit/debugging for `keep_completed_runs_days`.
- Active scratch folders must not be deleted unless their owning lock is stale.
- Cleanup should be explicit, logged, and safe.
- Cleanup failure should not invalidate successfully published NAS artifacts.

## Run And Status Indexing Strategy

Avoid treating `latest_run.json` as the source of truth.

Recommended status layout:

```text
working/status/
  latest_run.json
  runs/
    <run_id>.json
  documents/
    <document_id>.complete.json
```

Rules:

- Each run writes its own immutable or mostly immutable run manifest under `working/status/runs/`.
- `latest_run.json` is only a convenience pointer.
- If `latest_run.json` is missing, stale, or corrupt, the UI and status commands should fall back to listing run manifests.
- Shared status indexes should be updated under `status.lock` if multiple runs may write them.
- Document completion should be determined by completion rules, not by folder presence.

## Recovery And Repair Commands

NAS failures are expected operational events, not edge cases. The design should include future operator commands even if they are not implemented in Phase 1.

Possible future commands:

```text
retry-failed
repair-nas-artifacts
cleanup-staging
cleanup-scratch
clear-stale-locks
verify-nas-output
```

Suggested behavior:

### `retry-failed`

- Find documents marked `needs_retry`.
- Re-check source readiness.
- Reprocess only failed or incomplete documents.
- Respect existing locks.

### `repair-nas-artifacts`

- Find manifests without completion markers.
- Validate required artifacts.
- Rebuild completion markers if artifacts are valid.
- Mark documents incomplete if artifacts are missing.

### `cleanup-staging`

- Find staging folders older than a configured threshold.
- Ignore staging folders associated with active locks.
- Move questionable staging folders to quarantine before deletion if needed.

### `clear-stale-locks`

- List stale locks.
- Remove only locks older than `stale_after_minutes`.
- Log removed lock metadata.

### `verify-nas-output`

- Validate manifests, completion markers, required artifacts, and checksums.
- Report missing or corrupt artifacts.

## Implementation Phases

### Phase 1: Config And Runtime Path Resolution

- Add `pipeline.storage_mode` and `nas` settings models.
- Add validation for local and NAS modes.
- Add stricter path validation for input, output, and scratch roots.
- Add UNC path guidance to docs.
- Add a runtime path resolver used by CLI setup.
- Ensure all existing local tests continue to pass.

### Phase 2: Repository And NAS Filesystem Helpers

- Add repository methods for common file operations currently performed directly.
- Add atomic write helpers for JSON, YAML, text, and bytes.
- Add NAS healthcheck and retry wrappers.
- Add input readiness helpers.
- Add checksum helpers.
- Add filesystem lock helpers using atomic lock acquisition.
- Add scratch and staging cleanup helpers.

### Phase 3: Run Pipeline NAS Mode

- Update CLI runtime loading to select local or NAS `DataPaths`.
- Update discovery to use `nas.input_root` in NAS mode.
- Add input readiness checks before processing.
- Add source copy-to-scratch before normalization.
- Hash source before and after copy when enabled.
- Preserve NAS source metadata in manifests.
- Publish document artifacts from scratch to NAS output staging.
- Validate and finalize staged artifacts.
- Write document manifest and completion marker last.
- Add NAS failure classification.
- Add support for skipping completed NAS-published documents only when completion rules pass.

### Phase 4: Review And Interactive UI

- Make review service and web controller load from effective runtime `DataPaths`.
- Serve NAS-published page images and static reports.
- Save corrections and approvals through repository methods.
- Add clear missing-artifact messages for NAS disconnects.
- Add document locks for correction and approval writes.
- Add optimistic concurrency for correction file saves.

### Phase 5: Export NAS Mode

- Build export assets and dataset files in scratch.
- Publish complete exports atomically to NAS output staging.
- Protect export writes with `export.lock`.
- Validate expected export artifacts before finalization.
- Write `exports/export_manifest.json` last.
- Preserve current local export behavior.

### Phase 6: Recovery Commands, Tests, And Documentation

- Add focused tests for config, scanning, publish behavior, locks, run failure handling, interactive loading, exports, and recovery.
- Add repair/cleanup command documentation after implementation.
- Update `README.md` and `docs/COMMANDS.md` with NAS examples after code implementation.
- Keep this planned feature document as the design reference.

## Test Plan

### Config Tests

- `storage_mode: local` accepts existing config.
- `storage_mode: nas` requires `nas.input_root`, `nas.output_root`, and `nas.scratch_root`.
- Invalid storage mode fails validation.
- Invalid publish strategy fails validation.
- Scratch root inside NAS output root fails validation.
- Scratch root inside NAS input root fails validation.
- NAS output root inside NAS input root fails validation.
- NAS input root inside NAS output root fails validation.
- Retry counts and timeout values must be positive.
- UNC paths are accepted.
- Mapped drive paths are accepted for manual runs.

### Scanner Tests

- Recursive nested input paths are discovered under a temp NAS-like directory.
- `source_relative_path` is recorded for nested files.
- Unsupported files are ignored.
- Duplicate file hashes keep existing duplicate behavior.
- Files under output, staging, locks, scratch, hidden, or system folders are ignored.
- Files still changing size are skipped as not ready.
- Files still changing mtime are skipped as not ready.
- Missing ready marker skips the file when ready marker mode is enabled.

### Repository And Publish Tests

- Atomic JSON/YAML/text writes leave no partial final file on simulated failure.
- Publish copies a scratch document tree into NAS staging.
- Publish validates required files before final manifest update.
- Publish verifies checksums when enabled.
- Publish writes completion marker last.
- Retry wrapper retries transient filesystem failures.
- Stale locks can be replaced.
- Active locks prevent conflicting writes.
- Cross-device publish finalization is rejected or handled without claiming atomic rename.
- Stale staging folders can be detected safely.

### Lock Tests

- Lock acquisition is atomic.
- Existence-check-then-create is not used.
- Two simulated machines cannot acquire the same document lock.
- Lock heartbeat updates `updated_at`.
- Stale locks can be replaced after threshold.
- Non-stale locks block conflicting operations.
- Lock release does not remove a newer lock created by another process.

### Orchestrator Tests

- NAS read failure marks a document `needs_retry`.
- NAS publish failure marks a document `needs_retry`.
- Input unstable failure marks a document `needs_retry`.
- One failed NAS document does not stop the rest of the batch.
- Rerun skips completed NAS-published documents only when completion rules pass.
- Rerun retries documents with missing completion markers.
- Local mode behavior remains unchanged.

### Interactive Tests

- Dashboard loads pipeline and review state from NAS output root.
- Document context loads correction YAML, rendered page image, raw response, and report from NAS output root.
- Missing NAS page image returns a clear missing-artifact error.
- Approval updates correction and manifest state under NAS output root.
- Correction save uses document lock.
- Correction save detects stale loaded version and prevents overwrite.
- UI falls back to run manifests if `latest_run.json` is missing or stale.

### Export Tests

- Approved NAS correction files are exported.
- Export assets are generated in scratch first.
- Page assets, handwritten crops, split files, and manifests publish to NAS output root.
- Export lock prevents concurrent export writes.
- Failed publish does not leave a misleading final `export_manifest.json`.
- Export manifest is written last.
- Export checksum validation catches incomplete published files when enabled.

### Recovery And Cleanup Tests

- Stale staging folders can be listed.
- Active staging folders are not deleted.
- Stale locks can be listed and cleared.
- Failed scratch runs are retained.
- Successful scratch runs are cleaned when cleanup is enabled.
- Missing completion markers can be detected.
- Valid artifacts can be repaired by recreating completion markers.
- Invalid artifacts remain marked incomplete.

### Idempotency And Multi-PC Tests

- Two simulated machines cannot process the same document when locks are enabled.
- A rerun after interrupted publish can retry the document safely.
- Multiple independent runs can write separate run manifests.
- Shared status updates are protected by `status.lock` where required.
- Existing local behavior remains unchanged with the default config.

## Acceptance Criteria

- Local mode works exactly as it does now with the existing default config.
- NAS mode can process documents from a mounted NAS input root using local scratch.
- NAS mode does not process files that are still being copied or modified.
- All final artifacts required by review, approval, export, and interactive UI are published to NAS output.
- Nested source paths are preserved in manifests.
- NAS disconnects, input readiness failures, and lock conflicts are visible in status summaries.
- No partial document or export artifact is treated as complete.
- Completion markers or final manifests are written last.
- The web UI can review NAS-published documents without relying on local `data/` paths.
- Export can build approved datasets from NAS correction files and publish final datasets to NAS.
- Lock acquisition is atomic and safe across multiple PCs.
- Review and approval writes avoid overwriting newer edits.
- Reruns can safely skip completed documents and retry incomplete ones.
- Scratch and staging cleanup policies are defined.
- `latest_run.json` is treated as a convenience pointer, not the only source of truth.

## Assumptions And Non-Goals

Assumptions:

- The NAS is mounted and visible to Python as a normal filesystem path.
- The first implementation uses local scratch plus atomic publish.
- The current `data/` layout remains the canonical artifact layout under both local and NAS output roots.
- Existing local behavior remains the default.
- NAS input and output roots are separate folder trees.
- Local scratch has enough disk space for active processing.
- Operators understand that mapped drives may behave differently from UNC paths on Windows.

Non-goals for the first implementation:

- SFTP, object storage, WebDAV, or NAS vendor APIs.
- Multi-user authentication for the web UI.
- Database-backed indexing.
- Full distributed queue scheduling.
- Multiple PCs coordinating under one shared run ID.
- Real-time file watching on NAS input folders.
- Guaranteed interruption of low-level filesystem calls on all operating systems.
- Full migration of every read-only filesystem call into the repository abstraction in Phase 1.

## Future Extensions

After mounted-share NAS mode is stable, the storage layer can be extended with remote adapters and stronger coordination.

Possible later additions:

- SFTP-backed repository adapter.
- Object-storage-backed artifact repository.
- Central SQLite or Postgres index for multi-machine coordination.
- Redis or database-backed job queue.
- Shared distributed run coordination.
- Web UI authentication and reviewer identity management.
- Background sync service from local scratch to NAS.
- Admin command to repair stale staging folders and locks.
- Admin command to verify checksums across published artifacts.
- File watcher or inbox service for NAS input folders.
