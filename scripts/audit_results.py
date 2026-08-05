#!/usr/bin/env python3
"""Fail-closed checks that released artifacts reproduce manuscript results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "figdata"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def close(value: float, target: float, tolerance: float = 5e-3) -> None:
    if not math.isclose(value, target, abs_tol=tolerance, rel_tol=0.0):
        raise AssertionError(f"expected {target}, observed {value}")


def strategy_row(artifact: dict, arm: str, cell: str, strategy: str) -> dict:
    rows = [
        row for row in artifact["rows"]
        if (row["arm"], row["celltype"], row["strategy"]) == (arm, cell, strategy)
    ]
    if len(rows) != 1:
        raise AssertionError(f"missing/duplicate result row: {arm}, {cell}, {strategy}")
    return rows[0]


def main() -> None:
    partners = load("partners_by_threshold_v36.json")
    assert len(partners["per_locus"]) == 38
    chr17 = next(row for row in partners["per_locus"] if row["locus"] == "17_46062125_A_C")
    assert (chr17["n_r2_ge_0.8"], chr17["n_r2_ge_0.5"]) == (1285, 2693)

    pva = load("q1_pva_comparison_v53.json")
    assert pva["complete"] and len(pva["per_locus"]) == 128
    assert min(row["pva_q1"] for row in pva["per_locus"]) >= 0.80

    cohort = load("cohort_top_haplotype_comparison_v72.json")
    by_arm = {row["arm"]: row for row in cohort["by_arm"]}
    assert by_arm["armA"]["partner_availability"] == {"same_partner_set": 22, "different_partner_set": 5}
    assert by_arm["armB"]["partner_availability"] == {"same_partner_set": 25, "different_partner_set": 12}
    arm_a_exact = sum(
        by_arm["armA"][key]["top1"]["n_exact"]
        for key in ("same_partner_set", "different_partner_set_shared_coordinates_only")
    )
    arm_b_exact = sum(
        by_arm["armB"][key]["top1"]["n_exact"]
        for key in ("same_partner_set", "different_partner_set_shared_coordinates_only")
    )
    assert (arm_a_exact, arm_b_exact) == (54, 73)

    simulation = load("ranking_simulation_v77.json")
    simulation_audit = load("ranking_simulation_audit_v77.json")
    assert simulation["n_complete_files"] == 1620
    assert simulation_audit == {
        "status": "passed",
        "n_files": 1620,
        "n_expected_files": 1620,
        "n_fit_unavailable": 0,
        "n_empirical_alt_unavailable": 2,
        "n_ld_sign_state_unavailable": 272,
        "n_certified_truth_searches": 3240,
        "n_certified_fitted_searches": 3240,
        "n_nonunique_four_partner_top_ten": 2,
        "n_problems": 0,
        "problems": [],
    }
    with (DATA / "ranking_simulation_lead_availability_v77.csv").open(newline="") as handle:
        availability = list(csv.DictReader(handle))
    unavailable = {
        (row["lead_frequency_class"], int(row["n_haplotypes"])): float(
            row["zero_alt_carrier_fraction"]
        )
        for row in availability
    }
    close(unavailable[("rare", 500)], 1 / 270, 1e-10)
    close(unavailable[("rare", 1000)], 1 / 270, 1e-10)
    close(unavailable[("rare", 2000)], 0.0, 1e-12)

    enrichment = load("strategy_enrichment_total_top10_v66.json")
    targets = {
        "top1": (2.20, 1.57, 3.47),
        "top5": (2.46, 1.67, 3.61),
        "top10": (2.33, 1.63, 3.43),
        "single_lead": (1.43, 0.65, 2.28),
    }
    for strategy, expected in targets.items():
        row = strategy_row(enrichment, "armB", "Mic", strategy)
        for key, target in zip(("enrichment", "boot_lo", "boot_hi"), expected):
            close(row[key], target, 0.015)

    simple = load("public_baseline_enrichment_v75.json")
    simple_targets = {
        "fitted_top1": (16, 2.07, 1.48, 3.60),
        "empirical_mode": (17, 2.20, 1.60, 3.64),
        "ld_sign": (17, 2.20, 1.60, 3.64),
        "single_lead": (11, 1.43, 0.69, 2.29),
    }
    for strategy, expected in simple_targets.items():
        row = strategy_row(simple, "armB", "Mic", strategy)
        assert row["tp"] == expected[0]
        for key, target in zip(("enrichment", "boot_lo", "boot_hi"), expected[1:]):
            close(row[key], target, 0.015)

    verify = load("verify_sparse_v33.json")
    assert len(verify["per_locus"]) == 27
    assert all(row["true_R_argmax_is_dense_top1"] for row in verify["per_locus"])
    assert all(row["verdict_stable_across_seeds_and_floors"] for row in verify["per_locus"])
    print("All locked artifact and manuscript-result checks passed.")


if __name__ == "__main__":
    main()
