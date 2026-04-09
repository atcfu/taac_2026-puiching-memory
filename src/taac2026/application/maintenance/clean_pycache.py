from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
SKIPPED_ENV_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "env",
        "node_modules",
        "venv",
    }
)


@dataclass(slots=True)
class CleanupFailure:
    path: Path
    error: str


@dataclass(slots=True)
class CleanupResult:
    root: Path
    matched_dirs: list[Path]
    matched_files: int
    total_bytes: int
    dry_run: bool
    processed_dirs: list[Path]
    failures: list[CleanupFailure]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean __pycache__ directories under this repository")
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT),
        help="Directory to scan. Defaults to the repository root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be removed without deleting anything.",
    )
    parser.add_argument(
        "--include-env-dirs",
        action="store_true",
        help="Also scan common environment directories such as .venv, venv, env, .tox, and node_modules.",
    )
    return parser.parse_args(argv)


def find_pycache_dirs(root: Path, *, include_env_dirs: bool = False) -> list[Path]:
    root = root.resolve()
    if root.name == "__pycache__":
        return [root]

    skipped_dir_names = set() if include_env_dirs else set(SKIPPED_ENV_DIR_NAMES)
    matches: list[Path] = []
    for current_root, dirnames, _ in os.walk(root, topdown=True):
        dirnames[:] = sorted(directory for directory in dirnames if directory not in skipped_dir_names)
        if "__pycache__" not in dirnames:
            continue
        matches.append(Path(current_root) / "__pycache__")
        dirnames.remove("__pycache__")
    return sorted(matches)


def _count_directory_contents(directory: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for current_root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = Path(current_root) / filename
            file_count += 1
            try:
                total_bytes += file_path.stat().st_size
            except OSError:
                continue
    return file_count, total_bytes


def clean_pycache(
    root: Path,
    *,
    dry_run: bool = False,
    include_env_dirs: bool = False,
) -> CleanupResult:
    matched_dirs = find_pycache_dirs(root, include_env_dirs=include_env_dirs)
    matched_files = 0
    total_bytes = 0
    for directory in matched_dirs:
        directory_file_count, directory_total_bytes = _count_directory_contents(directory)
        matched_files += directory_file_count
        total_bytes += directory_total_bytes

    processed_dirs: list[Path] = []
    failures: list[CleanupFailure] = []
    for directory in matched_dirs:
        if dry_run:
            print(f"[dry-run] {directory}")
            processed_dirs.append(directory)
            continue

        try:
            shutil.rmtree(directory)
            print(f"[removed] {directory}")
            processed_dirs.append(directory)
        except OSError as exc:
            failures.append(CleanupFailure(path=directory, error=str(exc)))
            print(f"[failed] {directory}: {exc}", file=sys.stderr)

    return CleanupResult(
        root=root.resolve(),
        matched_dirs=matched_dirs,
        matched_files=matched_files,
        total_bytes=total_bytes,
        dry_run=dry_run,
        processed_dirs=processed_dirs,
        failures=failures,
    )


def _format_mib(total_bytes: int) -> str:
    return f"{float(total_bytes) / float(1024**2):.4f}"


def _print_summary(result: CleanupResult, *, include_env_dirs: bool) -> None:
    mode = "dry-run" if result.dry_run else "delete"
    print(f"root={result.root}")
    print(f"mode={mode}")
    print(f"matched_dirs={len(result.matched_dirs)}")
    print(f"processed_dirs={len(result.processed_dirs)}")
    print(f"matched_files={result.matched_files}")
    print(f"matched_size_mib={_format_mib(result.total_bytes)}")
    print(f"include_env_dirs={include_env_dirs}")
    print(f"failures={len(result.failures)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"root path not found: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"root path is not a directory: {root}", file=sys.stderr)
        return 2

    result = clean_pycache(
        root=root,
        dry_run=args.dry_run,
        include_env_dirs=args.include_env_dirs,
    )
    _print_summary(result, include_env_dirs=args.include_env_dirs)
    if result.failures:
        return 1
    return 0


__all__ = [
    "CleanupFailure",
    "CleanupResult",
    "clean_pycache",
    "find_pycache_dirs",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
