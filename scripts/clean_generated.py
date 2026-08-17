#!/usr/bin/env python3
"""Remove only generated public-build outputs."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
targets = [
    ROOT / "paper" / "main" / "figures",
    ROOT / "paper" / "supp" / "supp-figures",
    ROOT / "paper" / "supp" / "supp-tables",
    ROOT / "paper" / "ranking-generated",
    ROOT / ".mplconfig",
]
for directory in targets:
    if directory.exists():
        for path in sorted(directory.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        directory.rmdir()
# Remove the pre-technical-check table location if it remains from an older build.
legacy_table = ROOT / "paper" / "main" / "table1_cohort_haplotype_agreement.tex"
if legacy_table.exists():
    legacy_table.unlink()
