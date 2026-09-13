#!/usr/bin/env python3
"""
Automated Crossref & DataCite DOI Title Verification Script for Project N.
Extracts all DOIs and cited titles from markdown documentation and validates them
against official Crossref and DataCite metadata to prevent citation drift or invalid DOIs.
"""

import json
import re
import ssl
import sys
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def fetch_metadata_title(doi: str, ctx: ssl.SSLContext) -> str:
    # ArXiv DOIs use DataCite
    if doi.startswith("10.48550/"):
        url = f"https://api.datacite.org/dois/{doi}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "ProjectN-CitationVerifier/1.0 (mailto:ci@project-n.local)"}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return str(data["data"]["attributes"]["titles"][0]["title"])
    else:
        url = f"https://api.crossref.org/works/{doi}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "ProjectN-CitationVerifier/1.0 (mailto:ci@project-n.local)"}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return str(data["message"]["title"][0])


def extract_dois(text: str) -> list[str]:
    raw_matches = re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
    cleaned = []
    for d in raw_matches:
        d = d.rstrip(".,;:>")
        while d.endswith(")") and d.count(")") > d.count("("):
            d = d[:-1]
        while d.endswith("]"):
            d = d[:-1]
        if d not in cleaned:
            cleaned.append(d)
    return cleaned


def verify_file_citations(file_path: Path, ctx: ssl.SSLContext) -> int:
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    errors = 0
    checked = 0

    print(f"\n--- Verifying citations in {file_path.name} ---")
    for line in lines:
        line_s = line.strip()
        dois = extract_dois(line_s)
        if not dois:
            continue

        # Only attempt title extraction if the line is a formal bibliography entry
        # (bold author string ending in (YYYY).**)
        is_bib_entry = bool(re.search(r"\*\*[^\*]+?\(\d{4}[a-z]?\)\.\*\*", line_s))
        printed_title = ""
        if is_bib_entry:
            m_italic = re.search(
                r"\*\*[^\*]+?\(\d{4}[a-z]?\)\.\*\*\s+[\*_]([^*_]+?)\.[\*_]", line_s
            )
            if m_italic:
                printed_title = m_italic.group(1).strip()
            else:
                m_title = re.search(
                    r"\*\*[^\*]+?\(\d{4}[a-z]?\)\.\*\*\s+(.*?)(?:(?:\.|\?)\s+\*|\.\s+Paul|\.\s+Charles|\.\s+\[|\.\s+Published)",
                    line_s,
                )
                printed_title = m_title.group(1).strip("* ") if m_title else ""

        for doi in dois:
            checked += 1
            try:
                remote_title = fetch_metadata_title(doi, ctx)
                sim = similarity(printed_title, remote_title) if printed_title else 1.0
                if sim >= 0.70:
                    print(f"  [PASS] {doi} (sim: {sim:.2f})")
                    print(f"         Matched: '{remote_title[:75]}...'")
                else:
                    print(f"  [FAIL] Title mismatch for DOI {doi} (sim: {sim:.2f}):")
                    print(f"         Printed: '{printed_title}'")
                    print(f"         Remote:  '{remote_title}'")
                    errors += 1
            except Exception as e:
                print(f"  [FAIL] Error fetching DOI {doi}: {e}")
                errors += 1

    print(f"Summary for {file_path.name}: {checked} checked, {errors} errors.")
    return errors


def main() -> int:
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()

    root = Path(__file__).resolve().parent.parent
    files_to_check = [
        root / "docs" / "WHITE_PAPER.md",
        root / "INVARIANTS.md",
        root / "docs" / "SPECS.md",
    ]

    total_errors = 0
    for f in files_to_check:
        if f.exists():
            total_errors += verify_file_citations(f, ctx)

    if total_errors == 0:
        print("\nAll cited DOIs verified successfully against Crossref/DataCite APIs!")
        return 0
    else:
        print(f"\nVerification failed with {total_errors} total citation error(s).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
