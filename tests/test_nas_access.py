from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_SAMPLE_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def result(name: str, ok: bool, message: str = "", details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "test": name,
        "ok": ok,
        "message": message,
        "details": details or {},
    }


def safe_cleanup(path: Path) -> None:
    """
    Only deletes files/folders created under the healthcheck folder.
    This avoids accidental deletion of real NAS data.
    """
    if not path.exists():
        return

    if path.is_file():
        path.unlink()
        return

    if path.is_dir():
        for child in path.iterdir():
            safe_cleanup(child)
        path.rmdir()


def check_input_root(input_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    checks.append(result(
        "input_root_exists",
        input_root.exists(),
        f"Input root: {input_root}",
    ))

    if not input_root.exists():
        return checks

    checks.append(result(
        "input_root_is_directory",
        input_root.is_dir(),
        f"Input root is directory: {input_root}",
    ))

    if not input_root.is_dir():
        return checks

    try:
        items = list(input_root.iterdir())
        checks.append(result(
            "input_root_listable",
            True,
            f"Listed {len(items)} item(s) from input root.",
            {"sample_items": [str(p.name) for p in items[:10]]},
        ))
    except Exception as exc:
        checks.append(result(
            "input_root_listable",
            False,
            f"Could not list input root: {exc}",
        ))
        return checks

    sample_file = None

    try:
        for path in input_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SAMPLE_EXTENSIONS:
                sample_file = path
                break
    except Exception as exc:
        checks.append(result(
            "input_recursive_scan",
            False,
            f"Recursive scan failed: {exc}",
        ))
        return checks

    checks.append(result(
        "input_recursive_scan",
        True,
        "Recursive scan completed.",
    ))

    if sample_file is None:
        checks.append(result(
            "input_sample_file_read",
            True,
            "No supported sample document found. Skipping sample read test.",
        ))
        return checks

    try:
        size_before = sample_file.stat().st_size
        mtime_before = sample_file.stat().st_mtime

        with sample_file.open("rb") as f:
            sample_bytes = f.read(1024 * 1024)

        size_after = sample_file.stat().st_size
        mtime_after = sample_file.stat().st_mtime

        stable = size_before == size_after and mtime_before == mtime_after

        checks.append(result(
            "input_sample_file_read",
            True,
            f"Read sample file successfully: {sample_file}",
            {
                "sample_file": str(sample_file),
                "bytes_read": len(sample_bytes),
                "size_before": size_before,
                "size_after": size_after,
                "mtime_before": mtime_before,
                "mtime_after": mtime_after,
                "stable_during_read": stable,
            },
        ))

        checks.append(result(
            "input_sample_file_stability",
            stable,
            "Sample file size/mtime remained stable during read." if stable else "Sample file changed during read.",
            {"sample_file": str(sample_file)},
        ))

    except Exception as exc:
        checks.append(result(
            "input_sample_file_read",
            False,
            f"Could not read sample file {sample_file}: {exc}",
        ))

    return checks


def check_output_root(output_root: Path, keep_test_folder: bool = False) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    checks.append(result(
        "output_root_exists",
        output_root.exists(),
        f"Output root: {output_root}",
    ))

    if not output_root.exists():
        return checks

    checks.append(result(
        "output_root_is_directory",
        output_root.is_dir(),
        f"Output root is directory: {output_root}",
    ))

    if not output_root.is_dir():
        return checks

    test_root = output_root / ".nas_pipeline_healthcheck"
    run_id = f"test_{int(time.time())}_{os.getpid()}"
    run_root = test_root / run_id

    temp_file = run_root / "write_test.tmp"
    final_file = run_root / "write_test.json"
    staging_dir = run_root / ".staging" / "document_test"
    lock_dir = run_root / "working" / "locks" / "document_test.lock"

    try:
        run_root.mkdir(parents=True, exist_ok=False)
        checks.append(result(
            "output_create_test_folder",
            True,
            f"Created test folder: {run_root}",
        ))
    except Exception as exc:
        checks.append(result(
            "output_create_test_folder",
            False,
            f"Could not create test folder: {exc}",
        ))
        return checks

    try:
        payload = {
            "created_at": utc_now(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "purpose": "NAS pipeline write test",
        }

        temp_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        checks.append(result(
            "output_write_temp_file",
            True,
            f"Wrote temp file: {temp_file}",
        ))
    except Exception as exc:
        checks.append(result(
            "output_write_temp_file",
            False,
            f"Could not write temp file: {exc}",
        ))

    try:
        os.replace(temp_file, final_file)

        checks.append(result(
            "output_rename_temp_to_final",
            final_file.exists(),
            f"Renamed temp file to final file: {final_file}",
        ))
    except Exception as exc:
        checks.append(result(
            "output_rename_temp_to_final",
            False,
            f"Could not rename temp file to final file: {exc}",
        ))

    try:
        content = json.loads(final_file.read_text(encoding="utf-8"))

        checks.append(result(
            "output_read_back_final_file",
            True,
            "Read final JSON file back successfully.",
            content,
        ))
    except Exception as exc:
        checks.append(result(
            "output_read_back_final_file",
            False,
            f"Could not read final file back: {exc}",
        ))

    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
        staged_file = staging_dir / "artifact.json"
        staged_file.write_text(
            json.dumps({"status": "staged", "created_at": utc_now()}, indent=2),
            encoding="utf-8",
        )

        checks.append(result(
            "output_create_staging_folder",
            True,
            f"Created staging folder and artifact: {staging_dir}",
        ))
    except Exception as exc:
        checks.append(result(
            "output_create_staging_folder",
            False,
            f"Could not create staging folder/artifact: {exc}",
        ))

    try:
        lock_dir.parent.mkdir(parents=True, exist_ok=True)

        # Atomic lock acquisition pattern:
        # mkdir should fail if the lock already exists.
        lock_dir.mkdir(exist_ok=False)

        lock_payload = {
            "lock_type": "document",
            "document_id": "document_test",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }

        (lock_dir / "lock.json").write_text(
            json.dumps(lock_payload, indent=2),
            encoding="utf-8",
        )

        checks.append(result(
            "output_atomic_lock_create",
            True,
            f"Created lock directory atomically: {lock_dir}",
            lock_payload,
        ))
    except FileExistsError:
        checks.append(result(
            "output_atomic_lock_create",
            False,
            f"Lock already exists: {lock_dir}",
        ))
    except Exception as exc:
        checks.append(result(
            "output_atomic_lock_create",
            False,
            f"Could not create lock directory: {exc}",
        ))

    try:
        if final_file.exists():
            final_file.unlink()

        checks.append(result(
            "output_delete_file",
            not final_file.exists(),
            "Deleted final test file successfully.",
        ))
    except Exception as exc:
        checks.append(result(
            "output_delete_file",
            False,
            f"Could not delete final test file: {exc}",
        ))

    if not keep_test_folder:
        try:
            safe_cleanup(run_root)

            checks.append(result(
                "output_cleanup_test_folder",
                not run_root.exists(),
                "Cleaned up test folder successfully.",
            ))
        except Exception as exc:
            checks.append(result(
                "output_cleanup_test_folder",
                False,
                f"Could not clean up test folder: {exc}",
                {"test_folder": str(run_root)},
            ))
    else:
        checks.append(result(
            "output_cleanup_test_folder",
            True,
            "Skipped cleanup because --keep-test-folder was used.",
            {"test_folder": str(run_root)},
        ))

    return checks


def check_scratch_root(scratch_root: Path, keep_test_folder: bool = False) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    test_root = scratch_root / ".nas_pipeline_scratch_test" / f"test_{int(time.time())}_{os.getpid()}"
    test_file = test_root / "scratch_test.json"

    try:
        test_root.mkdir(parents=True, exist_ok=False)

        checks.append(result(
            "scratch_create_test_folder",
            True,
            f"Created scratch test folder: {test_root}",
        ))
    except Exception as exc:
        checks.append(result(
            "scratch_create_test_folder",
            False,
            f"Could not create scratch test folder: {exc}",
        ))
        return checks

    try:
        payload = {
            "created_at": utc_now(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "purpose": "Local scratch read/write test",
        }

        test_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        read_back = json.loads(test_file.read_text(encoding="utf-8"))

        checks.append(result(
            "scratch_write_and_read",
            True,
            "Scratch write/read worked.",
            read_back,
        ))
    except Exception as exc:
        checks.append(result(
            "scratch_write_and_read",
            False,
            f"Scratch write/read failed: {exc}",
        ))

    if not keep_test_folder:
        try:
            safe_cleanup(test_root)

            parent = test_root.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()

            checks.append(result(
                "scratch_cleanup_test_folder",
                not test_root.exists(),
                "Cleaned up scratch test folder successfully.",
            ))
        except Exception as exc:
            checks.append(result(
                "scratch_cleanup_test_folder",
                False,
                f"Could not clean up scratch test folder: {exc}",
                {"test_folder": str(test_root)},
            ))
    else:
        checks.append(result(
            "scratch_cleanup_test_folder",
            True,
            "Skipped cleanup because --keep-test-folder was used.",
            {"test_folder": str(test_root)},
        ))

    return checks


def print_summary(all_checks: list[dict[str, Any]]) -> None:
    passed = sum(1 for c in all_checks if c["ok"])
    failed = len(all_checks) - passed

    print("\n" + "=" * 80)
    print("NAS ACCESS TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 80)

    for check in all_checks:
        status = "PASS" if check["ok"] else "FAIL"
        print(f"[{status}] {check['test']}: {check['message']}")

    print("=" * 80)

    if failed:
        print("\nSome checks failed.")
        print("This usually means the current user/process does not have enough NAS permissions.")
        print("Required output permissions include: read, write, create folder, rename, delete, and create/remove lock folders.")
    else:
        print("\nAll checks passed. Basic NAS access pattern is compatible with the planned pipeline mode.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bare-bones NAS access test for NCAI pipeline storage pattern."
    )

    parser.add_argument(
        "--input-root",
        required=True,
        help="NAS input root path, for example: \\\\NAS-SERVER\\NCAI\\input or Z:/NCAI/input",
    )

    parser.add_argument(
        "--output-root",
        required=True,
        help="NAS output root path, for example: \\\\NAS-SERVER\\NCAI\\output or Z:/NCAI/output",
    )

    parser.add_argument(
        "--scratch-root",
        required=True,
        help="Local scratch root path, for example: data/nas_scratch or C:/NCAI/nas_scratch",
    )

    parser.add_argument(
        "--json-report",
        default=None,
        help="Optional path to save full JSON report.",
    )

    parser.add_argument(
        "--keep-test-folder",
        action="store_true",
        help="Keep test folders for debugging instead of deleting them.",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    scratch_root = Path(args.scratch_root)

    all_checks: list[dict[str, Any]] = []

    print("Running NAS access test...")
    print(f"Input root:   {input_root}")
    print(f"Output root:  {output_root}")
    print(f"Scratch root: {scratch_root}")

    all_checks.extend(check_input_root(input_root))
    all_checks.extend(check_output_root(output_root, keep_test_folder=args.keep_test_folder))
    all_checks.extend(check_scratch_root(scratch_root, keep_test_folder=args.keep_test_folder))

    report = {
        "created_at": utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "scratch_root": str(scratch_root),
        "checks": all_checks,
        "passed": sum(1 for c in all_checks if c["ok"]),
        "failed": sum(1 for c in all_checks if not c["ok"]),
    }

    if args.json_report:
        report_path = Path(args.json_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {report_path}")

    print_summary(all_checks)

    return 0 if report["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())