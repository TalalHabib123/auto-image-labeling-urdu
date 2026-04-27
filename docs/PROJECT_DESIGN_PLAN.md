You are a senior Python architect and implementation agent.

Your task is to DESIGN AND IMPLEMENT a production-style, modular Python pipeline for OCR-style document extraction using the Gemini API as the first extraction engine, with the explicit goal of building a human-correctable dataset that can later be used to fine-tune a Google Gemma multimodal model for the same document-understanding task offline.

This is not a prototype script. Build it like a maintainable internal system.

========================
1. PROJECT CONTEXT
========================

We have a repository of documents, primarily containing Urdu text.
Most documents are assumed to be 1 page, but the pipeline must support multi-page documents as well.
The system will process documents one by one through the Gemini vision/document model for OCR and structured extraction.

The extracted output must satisfy BOTH of these needs:
1. Be easy for humans to manually verify and correct
2. Be exportable into a clean, versionable training dataset for later multimodal fine-tuning

The architecture must be modular and observable.
The API connector must be replaceable in the future so Gemini can be swapped out for another provider or a local model without rewriting the whole pipeline.

Do NOT build everything in one file.
Create a clean codebase with clear responsibilities.

========================
2. PRIMARY GOALS
========================

Build a pipeline that:

1. Ingests documents from a source directory/repository
2. Detects file type and handles:
   - image files (png, jpg, jpeg, webp, tif, tiff)
   - PDFs
3. Converts each document into a normalized internal representation
4. Sends each page/document to Gemini through a provider abstraction
5. Receives structured OCR/extraction output
6. Stores:
   - raw model response
   - normalized parsed response
   - review-friendly artifacts
   - correction-ready editable artifacts
7. Produces a final curated dataset export suitable for future Gemma multimodal fine-tuning
8. Supports reruns, resumability, idempotency, and future incremental updates

========================
3. CRITICAL DESIGN PRINCIPLES
========================

Follow these rules strictly:

A. Modular architecture
- Separate orchestration, IO, providers, schemas, parsing, storage, review artifacts, export, and logging
- No god-file implementation

B. Provider abstraction
- Gemini implementation must sit behind an interface/base class
- Future providers must be pluggable with minimal changes

C. Human-correction first
- Do NOT directly treat model output as final training truth
- Preserve raw outputs
- Create a clean correction layer
- Final training data must come from reviewed/corrected outputs, not raw API responses

D. Observable outputs
- Every stage must leave artifacts that can be inspected
- Logs, manifests, status files, raw responses, parsed outputs, review files, and export summaries must be persisted

E. Strong schemas
- Use explicit Pydantic models/dataclasses for all major structures
- Avoid loose dict-only design for internal system boundaries

F. Re-runnable pipeline
- If run again, already-processed documents should not be needlessly reprocessed unless forced
- Support checkpointing and incremental updates

G. Urdu-sensitive handling
- Preserve original script exactly
- Do not transliterate unless explicitly requested
- Preserve reading order as best as possible
- Preserve line/block/page structure where available
- Keep confidence/uncertainty markers where possible

H. Dataset lifecycle awareness
- The output format must support:
  1. raw extraction
  2. manual correction
  3. approved gold data
  4. export into training-ready JSONL or equivalent multimodal fine-tuning format

========================
4. EXPECTED DELIVERABLES
========================

Implement the following:

1. A runnable main entrypoint
2. A modular Python package structure
3. Configurable source/input and output directories
4. Gemini provider implementation behind an abstract provider interface
5. Multi-page handling support
6. Structured schemas for raw, normalized, reviewed, corrected, and exported data
7. Review artifact generation
8. Exporter for training dataset
9. Logging + metrics + run manifest
10. Documentation explaining architecture and extension points

========================
5. RECOMMENDED REPO STRUCTURE
========================

Create a structure close to this:

project_root/
  README.md
  pyproject.toml
  .env.example
  config/
    default.yaml
    prompts.yaml
  src/
    main.py
    pipeline/
      orchestrator.py
      stages.py
      state_manager.py
    providers/
      base.py
      gemini_provider.py
      factory.py
    ingestion/
      scanner.py
      file_types.py
      pdf_loader.py
      image_loader.py
      page_renderer.py
    schemas/
      common.py
      document.py
      extraction.py
      review.py
      export.py
    parsing/
      response_parser.py
      normalizer.py
      validators.py
    storage/
      paths.py
      repository.py
      manifests.py
      raw_store.py
    review/
      artifact_builder.py
      html_report.py
      diff_helpers.py
    export/
      dataset_builder.py
      jsonl_exporter.py
      split_builder.py
    observability/
      logging_config.py
      metrics.py
      tracing.py
    utils/
      hashing.py
      retry.py
      concurrency.py
      time.py
  data/
    input/
    working/
    review/
    curated/
    exports/
  tests/

You may refine the structure, but do not collapse responsibilities.

========================
6. REQUIRED PIPELINE FLOW
========================

Implement this logical flow:

Stage 1: Discover input documents
- Scan input directory recursively
- Assign a stable document_id
- Hash files to support deduplication/re-runs
- Build a document manifest

Stage 2: Normalize document into pages
- If image: treat as single page
- If PDF: render each page to image and maintain page order
- Store page-level artifacts in a working directory

Stage 3: Build extraction request
- Use provider abstraction
- Include configurable prompt templates
- Ask model for structured output, not free-form only
- Also store full raw textual/model output

Stage 4: Call Gemini provider
- Handle retries, transient failures, backoff, timeout, and rate limiting
- Persist raw request metadata and raw response
- Support dry-run mode where possible

Stage 5: Parse and normalize result
- Convert model response into internal schema
- Preserve page structure, blocks/lines if available
- Track parse issues and validation warnings

Stage 6: Generate review artifacts
- Create per-document review package containing:
  - original page image(s)
  - extracted structured JSON
  - raw response text
  - easy-to-edit correction file
  - human-readable HTML or markdown report
- Make it easy for a human to compare source image against extracted content

Stage 7: Human correction layer
- Define a canonical editable corrected file format
- The corrected file becomes the source of truth once approved
- Preserve revision metadata:
  - corrected_by
  - corrected_at
  - review_status
  - notes
  - source_version

Stage 8: Curated dataset assembly
- Only approved/corrected data should flow into curated dataset
- Build training examples in an export-friendly structure

Stage 9: Export for fine-tuning
- Produce JSONL or equivalent export format suitable for multimodal fine-tuning workflows
- Include image/document reference + target structured text/JSON output
- Keep export deterministic and reproducible
- Generate export manifest and stats

========================
7. DATA MODEL REQUIREMENTS
========================

Design explicit schemas for at least the following:

A. DocumentManifest
- document_id
- source_path
- file_name
- file_type
- sha256
- page_count
- status
- created_at
- updated_at
- pipeline_version

B. PageArtifact
- document_id
- page_number
- image_path
- width
- height
- render_metadata

C. RawExtractionRecord
- document_id
- page_number or document_scope
- provider_name
- model_name
- request_prompt_version
- request_metadata
- raw_response_text
- raw_response_json if available
- token_usage if available
- latency_ms
- status
- error_details

D. NormalizedExtraction
- document_id
- page_number
- full_text
- structured_content
- blocks
- lines
- tables if any
- warnings
- confidence_notes
- parser_version

E. ReviewRecord
- document_id
- review_status (pending / in_review / approved / rejected / needs_retry)
- reviewer
- comments
- corrected_output_path
- raw_output_path
- report_path

F. CuratedTrainingSample
- sample_id
- document_id
- page_refs
- input_media_paths
- task_instruction
- target_output
- language
- source_split
- approval_metadata

Make the schemas future-safe, even if some fields are null initially.

========================
8. OUTPUT STORAGE STRATEGY
========================

Design the output folders so every stage is inspectable.

Suggested layout:

data/
  input/
  working/
    manifests/
    rendered_pages/
    raw_requests/
    raw_responses/
    parsed/
  review/
    reports/
    correction_files/
  curated/
    approved/
    rejected/
  exports/
    assets/
      pages/
      crops/
    datasets/
      full/
      page_text/
      regions_blocks/
      regions_sections/
      handwritten/
    gemma_multimodal/
      train.jsonl
      val.jsonl
      test.jsonl
      export_manifest.json

Use stable, deterministic naming with document_id-based folders.

========================
9. PROVIDER ABSTRACTION REQUIREMENTS
========================

Create a provider interface similar in spirit to:

- prepare_input(document/pages)
- build_request(prompt, media)
- invoke(request)
- parse_response(raw_response)
- healthcheck()

Gemini implementation must:
- live in providers/gemini_provider.py
- be fully isolated from orchestration logic
- use environment/config-driven model selection
- support prompt versioning
- support future replacement by another provider without touching the pipeline core

Do not hardwire Gemini-specific logic across unrelated modules.

========================
10. PROMPTING REQUIREMENTS FOR EXTRACTION
========================

The Gemini extraction prompt should aim to return structured OCR-style output, not just plain text.

The prompt strategy should:
- preserve original Urdu text faithfully
- avoid translation unless explicitly asked
- preserve page and reading order
- separate uncertain text from confident text if possible
- return structured JSON-compatible output
- optionally capture:
  - document language
  - page sections
  - text blocks
  - lines
  - low-confidence spans
  - notes about unreadable or ambiguous regions

Keep prompt templates externalized in config/prompts.yaml, not buried in code.

========================
11. HUMAN REVIEW DESIGN
========================

This is one of the most important parts.

The system must generate correction-friendly artifacts, not just machine outputs.

At minimum, produce:
- source page image(s)
- extracted text/JSON
- raw model output
- correction file template
- side-by-side human-readable HTML report

The correction format should be simple and editable, such as:
- JSON
- YAML
- or structured markdown

Pick one canonical correction format and use it consistently.

The report should help a reviewer answer:
- What did the model see?
- What did the model output?
- What is the corrected gold version?
- What changed?
- Is this sample approved for training?

========================
12. TRAINING EXPORT REQUIREMENTS
========================

Design the exporter so it can generate a dataset suitable for future multimodal fine-tuning of Gemma.

Important:
- Do not assume one fixed final trainer format forever
- Build an intermediate curated representation first
- Then export into one or more training formats

Implement:
1. canonical curated sample format
2. export adapters for JSONL, JSON, CSV, and XLSX
3. dataset modes for full document targets, page text, block/section regions, and handwritten block crops
4. shared page-image assets plus separate crop assets under exports

Each exported sample should conceptually contain:
- input image/document reference
- task instruction
- target structured output
- deterministic split metadata
- approval metadata

Handwritten crop exports should contain one record per handwritten block with:
- crop path
- source page image path
- document/page/block ids
- bbox
- text
- auto/manual/effective handwriting metadata

Also generate:
- train/val/test split support
- export manifest
- dataset statistics:
  - sample counts
  - page counts
  - avg text length
  - language counts
  - review coverage
  - dataset counts
  - skipped handwritten crop counts
  - error counts

========================
13. CONFIGURATION REQUIREMENTS
========================

Use config files plus environment variables.

Support configuration for:
- input directory
- output directory
- allowed file types
- concurrency
- retry limits
- timeout
- provider selection
- Gemini model name
- prompt template version
- whether to process per-page or per-document
- whether to skip completed items
- export split ratios
- export dataset types
- export output formats
- export region source
- shared export page asset strategy
- handwritten auto-detection keywords and crop padding
- logging level

Use:
- .env for secrets
- YAML/TOML for app config
- no hardcoded secrets in code

========================
14. OBSERVABILITY REQUIREMENTS
========================

Add strong observability.

Implement:
- structured logging
- run manifest
- per-document status tracking
- error categories
- summary report at the end of a run

Track at least:
- discovered documents
- processed documents
- succeeded
- failed
- pending review
- approved
- exported
- average latency
- retry counts

Make logs readable and useful for debugging document-specific failures.

========================
15. RELIABILITY REQUIREMENTS
========================

Implement for production-minded robustness:

- Retry with backoff for provider/API failures
- Graceful handling of malformed documents
- Partial progress persistence
- Resume support after interruption
- Skip/force reprocess flags
- Validation before export
- Clean error boundaries so one bad document does not kill the whole batch

========================
16. TESTING REQUIREMENTS
========================

Add tests for:
- file discovery
- PDF page rendering path
- schema validation
- provider interface contract
- response parsing
- manifest/state transitions
- export generation
- idempotent reruns

Use mocks/fakes for Gemini API in tests.
Do not depend on live API calls for automated tests.

========================
17. DOCUMENTATION REQUIREMENTS
========================

Write a strong README that explains:
- architecture
- setup
- environment variables
- how to run ingestion
- how review artifacts work
- how to approve corrected data
- how export works
- how to swap providers
- known assumptions and limitations

Also include an “extension points” section.

========================
18. IMPLEMENTATION PRIORITY ORDER
========================

Implement in this order:

Phase 1:
- project structure
- schemas
- config
- ingestion
- PDF/image normalization
- storage layout
- logging

Phase 2:
- provider abstraction
- Gemini provider
- orchestration
- raw + parsed persistence

Phase 3:
- review artifact generation
- correction workflow
- approval states

Phase 4:
- curated dataset builder
- export pipeline
- dataset stats

Phase 5:
- tests
- README
- polish

========================
19. IMPORTANT NON-GOALS
========================

Do NOT:
- build a huge web app unless absolutely necessary
- hide raw responses
- mix corrected data with raw data carelessly
- couple the whole system to Gemini directly
- output only plain text with no structure
- create a fragile one-off notebook-style solution

========================
20. FINAL IMPLEMENTATION EXPECTATIONS
========================

When you implement, provide:

1. Clean codebase
2. Final repo tree
3. Explanation of architecture decisions
4. Notes on assumptions made
5. Example run flow
6. Example output artifacts
7. Clear instructions for how a reviewer corrects a sample
8. Example exported training sample

If any requirement is unclear, make the most reasonable engineering decision and document it explicitly in README rather than blocking.

Prioritize correctness, modularity, inspectability, and future fine-tuning readiness over speed of first implementation.
