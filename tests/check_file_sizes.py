#!/usr/bin/env python3
"""
Automated File Size & Modularity Quality Gate for Project N.
Enforces the 400-line hard ceiling across all source files (Python, TypeScript, Dart).
"""

import sys
from pathlib import Path

MAX_ALLOWED_LINES = 400
WARNING_THRESHOLD_LINES = 350

SOURCE_DIRS = [
    "models",
    "extraction",
    "server",
    "storage",
    "rag",
    "training",
    "tests",
    "ui/src",
    "web/src",
    "app/lib",
]

EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".dart"}

EXCLUDE_PATTERNS = {
    ".g.dart",
    ".freezed.dart",
    "__init__.py",
    "site/",
    ".venv/",
    "node_modules/",
    "build/",
}


def count_lines(file_path: Path) -> int:
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    violations: list[tuple[str, int]] = []
    warnings: list[tuple[str, int]] = []
    checked_count = 0

    for rel_dir in SOURCE_DIRS:
        target_dir = root / rel_dir
        if not target_dir.exists():
            continue

        for path in target_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in EXTENSIONS:
                continue

            rel_str = str(path.relative_to(root))
            if any(exc in rel_str for exc in EXCLUDE_PATTERNS):
                continue

            line_count = count_lines(path)
            checked_count += 1

            if line_count > MAX_ALLOWED_LINES:
                violations.append((rel_str, line_count))
            elif line_count >= WARNING_THRESHOLD_LINES:
                warnings.append((rel_str, line_count))

    print(f"File Size Quality Gate: Verified {checked_count} source files.")

    if warnings:
        print(f"\n[ADVISORY] {len(warnings)} file(s) approaching 400-line ceiling (>= 350 lines):")
        for fpath, lines in sorted(warnings, key=lambda x: x[1], reverse=True):
            print(f"  • {fpath}: {lines} lines (consider decomposing)")

    if violations:
        print(f"\n[FAIL] {len(violations)} file(s) exceed {MAX_ALLOWED_LINES}-line hard limit:")
        for fpath, lines in sorted(violations, key=lambda x: x[1], reverse=True):
            print(
                f"  • {fpath}: {lines} lines (exceeds {MAX_ALLOWED_LINES} limit by {lines - MAX_ALLOWED_LINES})"
            )
        print("\nPlease decompose these files according to docs/ENGINEERING_STANDARDS.md.")
        return 1

    print("All source files adhere to the 400-line modularity mandate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
