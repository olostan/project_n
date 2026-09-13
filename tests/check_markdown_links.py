#!/usr/bin/env python3
"""
Validates all relative markdown links across project documentation.
Ensures zero 404s or broken paths exist in root and docs/.
"""

import re
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    md_files = list(root.glob("*.md")) + list((root / "docs").glob("*.md"))

    broken: list[tuple[str, str, str]] = []
    checked = 0

    for p in md_files:
        if p.is_symlink() and not p.exists():
            broken.append((str(p.relative_to(root)), "broken symlink", ""))
            continue

        content = p.read_text(encoding="utf-8")
        # Match markdown links: [text](target)
        links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
        for _text, target in links:
            target_clean = target.strip("<> \t\r\n")
            if (
                target_clean.startswith("http://")
                or target_clean.startswith("https://")
                or target_clean.startswith("mailto:")
                or target_clean.startswith("#")
                or not target_clean
            ):
                continue

            checked += 1
            path_part = target_clean.split("#")[0]
            if not path_part:
                continue

            resolved = (p.parent / path_part).resolve()
            if not resolved.exists():
                broken.append((str(p.relative_to(root)), target, str(resolved)))

    print(f"Markdown Link Check: {checked} links verified across {len(md_files)} files.")
    if broken:
        print(f"FAILED: {len(broken)} broken relative link(s) found:")
        for source, target, resolved in broken:
            print(f"  In '{source}': '{target}' -> missing '{resolved}'")
        return 1

    print("All relative documentation links resolved successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
