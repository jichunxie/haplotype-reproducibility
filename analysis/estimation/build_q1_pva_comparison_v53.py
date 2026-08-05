#!/usr/bin/env python
"""Build a sanitized ROS/MAP-versus-1KGP q=1 PVA comparison artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


WORK = Path("/hpc/group/xielab/jx42/CHAP/work")
SOURCES = {
    "ROS/MAP": (
        WORK / "rosmap_factor_v51" / "rosmap_factor_v51_summary.json"
    ),
    "1000 Genomes unrelated EUR": (
        WORK / "onekg_factor_v52" / "onekg_factor_v52_summary.json"
    ),
}
OUT = WORK / "onekg_factor_v52" / "q1_pva_comparison_v53.json"
EXPECTED = {
    ("ROS/MAP", "armA"): 27,
    ("ROS/MAP", "armB"): 37,
    ("1000 Genomes unrelated EUR", "armA"): 27,
    ("1000 Genomes unrelated EUR", "armB"): 37,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


rows = []
source_metadata = {}
for cohort, path in SOURCES.items():
    source = json.load(path.open())
    if source["complete"] is not True or source["q"] != 1:
        raise RuntimeError(f"{cohort}: source is not a complete q=1 artifact")
    if source["free_threshold_fit"] is not False:
        raise RuntimeError(f"{cohort}: free-threshold fit is not allowed")
    source_metadata[cohort] = {
        "path": str(path),
        "sha256": sha256(path),
        "jobs": source.get("rcc_jobs", source.get("jobs", {})),
        "n_loci": source["n_loci_complete"],
    }
    for row in source["per_locus"]:
        pva = float(row["factor_variance_fraction"])
        if not 0.0 <= pva <= 0.990000000001:
            raise RuntimeError(
                f"{cohort} {row['arm']} {row['locus']}: invalid PVA {pva}"
            )
        rows.append({
            "cohort": cohort,
            "arm": row["arm"],
            "locus": row["locus"],
            "n_fitted_variants": row["n_fitted_variants"],
            "pva_q1": pva,
        })

groups = []
for key, expected in EXPECTED.items():
    cohort, arm = key
    selected = [
        row for row in rows
        if row["cohort"] == cohort and row["arm"] == arm
    ]
    if len(selected) != expected:
        raise RuntimeError(
            f"{cohort} {arm}: got {len(selected)} loci, expected {expected}"
        )
    values = np.asarray([row["pva_q1"] for row in selected])
    groups.append({
        "cohort": cohort,
        "arm": arm,
        "n_loci": len(values),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "n_ge_0.75": int(np.sum(values >= 0.75)),
        "n_ge_0.90": int(np.sum(values >= 0.90)),
        "n_at_uniqueness_ceiling": int(
            np.sum(values >= 0.989999999)
        ),
    })

artifact = {
    "script": "build_q1_pva_comparison_v53.py",
    "complete": True,
    "q": 1,
    "pva_definition": (
        "tr(B B') / tr(R) = sum_j ||b_j||^2 / p "
        "= 1 - sum_j psi_j / p; tr(R)=p"
    ),
    "psi_min": 0.01,
    "interpretation": (
        "fraction of total unit-diagonal latent variance assigned to the "
        "common factor; not an off-diagonal reconstruction-error measure"
    ),
    "cohort_comparison_scope": (
        "descriptive; report cohort-by-arm groups separately and do not "
        "treat 1000 Genomes as a validation sample for ROS/MAP"
    ),
    "sources": source_metadata,
    "groups": groups,
    "per_locus": sorted(
        rows, key=lambda row: (row["cohort"], row["arm"], row["locus"])
    ),
}
OUT.write_text(json.dumps(artifact, indent=2, allow_nan=False))
print(json.dumps({
    "complete": True,
    "n_rows": len(rows),
    "output": str(OUT),
}))
