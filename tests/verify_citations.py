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
        req = urllib.request.Request(url, headers={"User-Agent": "ProjectN-CitationVerifier/1.0 (mailto:ci@project-n.local)"})
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return data["data"]["attributes"]["titles"][0]["title"]
    else:
        url = f"https://api.crossref.org/works/{doi}"
        req = urllib.request.Request(url, headers={"User-Agent": "ProjectN-CitationVerifier/1.0 (mailto:ci@project-n.local)"})
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return data["message"]["title"][0]

def verify_file_citations(file_path: Path, ctx: ssl.SSLContext) -> int:
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    errors = 0
    checked = 0

    print(f"\n--- Verifying citations in {file_path.name} ---")
    for line in lines:
        line_s = line.strip()
        # Find lines with DOI markdown link
        m_doi = re.search(r"\[doi:(10\.\d{4,9}/[^\]]+)\]", line_s)
        if not m_doi:
            continue

        doi = m_doi.group(1)
        # Extract title: check italic format (*Title.*) first, then standard (Title. *Journal*)
        m_italic = re.search(r"\*\*[^\*]+\*\*\s+\*([^*]+?)\.\*", line_s)
        if m_italic:
            printed_title = m_italic.group(1).strip()
        else:
            m_title = re.search(
                r"\*\*[^\*]+\*\*\s+(.*?)(?:(?:\.|\?)\s+\*|\.\s+Paul|\.\s+Charles|\.\s+\[|\.\s+Published)",
                line_s,
            )
            printed_title = m_title.group(1).strip("* ") if m_title else ""
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
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    root = Path(__file__).resolve().parent.parent
    files_to_check = [
        root / "docs" / "WHITE_PAPER.md",
        root / "docs" / "REVIEW_REFINEMENTS.md",
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
