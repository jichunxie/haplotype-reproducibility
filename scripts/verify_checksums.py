#!/usr/bin/env python3
"""Verify the immutable release-input SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "sha256.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    expected = json.loads(MANIFEST.read_text())
    problems = []
    for relative, checksum in expected.items():
        path = ROOT / relative
        if not path.exists():
            problems.append(f"missing: {relative}")
        elif digest(path) != checksum:
            problems.append(f"checksum mismatch: {relative}")
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"Verified {len(expected)} release inputs.")


if __name__ == "__main__":
    main()
