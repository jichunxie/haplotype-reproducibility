#!/usr/bin/env python3
"""Summarize the density of fitted q=1 latent-scale loadings.

This diagnostic addresses row sparsity of the fitted loading vector, not
factor rank. It reads the public 1000 Genomes fixed-margin q=1 coefficient
files, checks the unit-variance parameterization, and stores normalized
rank-magnitude and cumulative loading-energy curves. No random numbers are
used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GRID = np.linspace(0.0, 1.0, 101)
EXPECTED = {"armA": 27, "armB": 37}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def interpolated_curve(values: np.ndarray) -> list[float]:
    x = np.arange(1, values.size + 1, dtype=np.float64) / values.size
    return np.interp(GRID, np.r_[0.0, x], np.r_[values[0], values]).tolist()


def summarize_locus(path: Path, arm: str) -> dict:
    with np.load(path, allow_pickle=False) as fit:
        B = fit["B"][:, 0].astype(np.float64)
        psi = fit["psi"].astype(np.float64)
    p = B.size
    if psi.shape != (p,):
        raise AssertionError(f"{arm}/{path.stem}: B/psi shape mismatch")
    if np.max(np.abs(B ** 2 + psi - 1.0)) > 1e-10:
        raise AssertionError(f"{arm}/{path.stem}: unit variance failed")
    if psi.min() < 0.01 - 1e-10:
        raise AssertionError(f"{arm}/{path.stem}: uniqueness floor failed")

    magnitude = np.sort(np.abs(B))[::-1]
    energy = magnitude ** 2
    cumulative = np.cumsum(energy) / energy.sum()
    fraction = np.arange(1, p + 1, dtype=np.float64) / p
    ascending = magnitude[::-1]
    gaps = np.diff(ascending)
    gap_index = int(np.argmax(gaps)) if gaps.size else 0
    effective_support = float(energy.sum() ** 2 / np.sum(energy ** 2))

    return {
        "cohort": "1000 Genomes unrelated EUR",
        "arm": arm,
        "locus": path.stem,
        "n_fitted_variants": p,
        "pva_q1": float(energy.mean()),
        "effective_support": effective_support,
        "effective_support_fraction": effective_support / p,
        "fraction_variants_for_90pct_energy": float(
            fraction[np.searchsorted(cumulative, 0.90)]
        ),
        "fraction_variants_for_95pct_energy": float(
            fraction[np.searchsorted(cumulative, 0.95)]
        ),
        "abs_loading_min": float(magnitude[-1]),
        "abs_loading_q05": float(np.quantile(magnitude, 0.05)),
        "abs_loading_median": float(np.median(magnitude)),
        "abs_loading_q95": float(np.quantile(magnitude, 0.95)),
        "abs_loading_max": float(magnitude[0]),
        "largest_adjacent_abs_loading_gap": (
            float(gaps[gap_index]) if gaps.size else 0.0
        ),
        "largest_gap_below": (
            float(ascending[gap_index]) if gaps.size else None
        ),
        "largest_gap_above": (
            float(ascending[gap_index + 1]) if gaps.size else None
        ),
        "n_abs_loading_lt_0_5": int(np.sum(magnitude < 0.5)),
        "n_abs_loading_lt_0_8": int(np.sum(magnitude < 0.8)),
        "rank_fraction_grid": GRID.tolist(),
        "abs_loading_by_rank_fraction": interpolated_curve(magnitude),
        "cumulative_energy_by_rank_fraction": np.interp(
            GRID,
            np.r_[0.0, fraction],
            np.r_[0.0, cumulative],
        ).tolist(),
    }


def group_summary(rows: list[dict], arm: str) -> dict:
    selected = [row for row in rows if row["arm"] == arm]

    def three(key: str) -> dict:
        values = np.asarray([row[key] for row in selected])
        return {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
        }

    return {
        "cohort": "1000 Genomes unrelated EUR",
        "arm": arm,
        "n_loci": len(selected),
        "effective_support_fraction": three(
            "effective_support_fraction"
        ),
        "fraction_variants_for_90pct_energy": three(
            "fraction_variants_for_90pct_energy"
        ),
        "fraction_variants_for_95pct_energy": three(
            "fraction_variants_for_95pct_energy"
        ),
        "abs_loading_q05": three("abs_loading_q05"),
        "largest_adjacent_abs_loading_gap": three(
            "largest_adjacent_abs_loading_gap"
        ),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for arm, expected in EXPECTED.items():
        files = sorted((args.result_root / arm).glob("*.npz"))
        if len(files) != expected:
            raise AssertionError(
                f"{arm}: expected {expected} coefficient files, got "
                f"{len(files)}"
            )
        rows.extend(summarize_locus(path, arm) for path in files)

    artifact = {
        "script": "build_q1_loading_density_v55.py",
        "complete": True,
        "cohort": "1000 Genomes unrelated EUR",
        "q": 1,
        "loading_scale": (
            "latent-scale b with b_j^2 + psi_j = 1; sparsity concerns "
            "|b_j|, not its LD-direction sign"
        ),
        "interpretation": (
            "descriptive loading-density diagnostic; does not fit or test "
            "a sparse-PCA estimator"
        ),
        "effective_support_definition": (
            "(sum_j b_j^2)^2 / sum_j b_j^4; divide by p so 1 is an "
            "equal-magnitude dense loading vector"
        ),
        "groups": [
            group_summary(rows, arm) for arm in EXPECTED
        ],
        "per_locus": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n"
    )
    print("wrote", args.output)
    for group in artifact["groups"]:
        effective = group["effective_support_fraction"]
        fraction90 = group["fraction_variants_for_90pct_energy"]
        print(
            group["arm"],
            "effective support/p min/median/max",
            f"{effective['minimum']:.4f}/"
            f"{effective['median']:.4f}/"
            f"{effective['maximum']:.4f}",
            "fraction for 90% energy min/median/max",
            f"{fraction90['minimum']:.4f}/"
            f"{fraction90['median']:.4f}/"
            f"{fraction90['maximum']:.4f}",
        )


if __name__ == "__main__":
    main()
