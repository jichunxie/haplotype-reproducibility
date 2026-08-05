#!/usr/bin/env python3
"""Fail-closed audit of the complete v77 simulation grid."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import run_ranking_simulation_cell_v77 as v77


LEAD_FREQUENCY_CLASSES = ("rare", "common")
SCENARIOS = ("lead_ld_q1", "balanced_q1", "lead_ld_q2")
SAMPLE_SIZES = (500, 1000, 2000)
PARTNER_COUNTS = (4, 32, 256)
REPLICATES = 30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.input_root.rglob("replicate_*.json"))
    documents = [json.loads(path.read_text()) for path in paths]
    expected_cells = {
        (lead_class, scenario, n, partners, replicate)
        for lead_class in LEAD_FREQUENCY_CLASSES
        for scenario in SCENARIOS
        for n in SAMPLE_SIZES
        for partners in PARTNER_COUNTS
        for replicate in range(REPLICATES)
    }
    observed_cells = {
        (
            document.get("lead_frequency_class"),
            document.get("scenario"),
            document.get("n_haplotypes"),
            document.get("n_partners"),
            document.get("replicate"),
        )
        for document in documents
    }
    problems: list[str] = []
    if observed_cells != expected_cells:
        problems.append(
            f"grid mismatch: missing={len(expected_cells - observed_cells)}, "
            f"unexpected={len(observed_cells - expected_cells)}"
        )

    fit_unavailable = 0
    empirical_alt_unavailable = 0
    ld_sign_unavailable = 0
    nonunique_exact_top_ten = 0
    certified_truth_searches = 0
    certified_fitted_searches = 0
    truth_by_population: dict[tuple, tuple] = {}
    for document in documents:
        key = (
            document.get("lead_frequency_class"),
            document.get("scenario"),
            document.get("n_haplotypes"),
            document.get("n_partners"),
            document.get("replicate"),
        )
        if document.get("status") != "complete":
            problems.append(f"failed cell {key}: {document.get('error')}")
            continue
        lead_class, scenario, n, partners, replicate = key
        frequency = document["frequency_design"]
        lead_frequency = float(frequency["lead_alt_frequency"])
        if lead_class == "rare" and not 0.005 <= lead_frequency < 0.05:
            problems.append(f"rare lead frequency out of range at {key}")
        if lead_class == "common" and not 0.05 <= lead_frequency <= 0.40:
            problems.append(f"common lead frequency out of range at {key}")
        if frequency["n_rare_partners"] != partners // 2:
            problems.append(f"wrong rare-partner count at {key}")
        if frequency["n_common_partners"] != partners // 2:
            problems.append(f"wrong common-partner count at {key}")
        truth_rng = v77.seed_rng(lead_class, scenario, partners, replicate, 0)
        _, _, regenerated_probability, regenerated_class = v77.make_truth(
            truth_rng, partners, scenario, lead_class
        )
        rare_probability = regenerated_probability[1:][regenerated_class[1:] == "rare"]
        common_probability = regenerated_probability[1:][
            regenerated_class[1:] == "common"
        ]
        if not np.all((rare_probability >= 0.005) & (rare_probability < 0.05)):
            problems.append(f"regenerated rare-partner frequency out of range at {key}")
        if not np.all((common_probability >= 0.05) & (common_probability <= 0.40)):
            problems.append(f"regenerated common-partner frequency out of range at {key}")
        regenerated_summary = {
            "lead_alt_frequency": float(regenerated_probability[0]),
            "minimum_partner_alt_frequency": float(np.min(regenerated_probability[1:])),
            "median_partner_alt_frequency": float(np.median(regenerated_probability[1:])),
            "maximum_partner_alt_frequency": float(np.max(regenerated_probability[1:])),
        }
        for name, expected_value in regenerated_summary.items():
            if not math.isclose(
                float(frequency[name]), expected_value, rel_tol=0.0, abs_tol=1e-15
            ):
                problems.append(f"stored/regenerated {name} mismatch at {key}")

        population_key = (lead_class, scenario, partners, replicate)
        population_signature = (
            frequency["lead_alt_frequency"],
            frequency["minimum_partner_alt_frequency"],
            frequency["median_partner_alt_frequency"],
            frequency["maximum_partner_alt_frequency"],
            document["truth"]["mean_communality"],
        )
        previous = truth_by_population.setdefault(population_key, population_signature)
        if population_signature != previous:
            problems.append(f"nested sample sizes changed population truth at {key}")

        fit_available = bool(document["fit"]["available"])
        fit_unavailable += int(not fit_available)
        if len(document["per_state"]) != 2:
            problems.append(f"wrong number of lead states at {key}")
            continue
        for state in document["per_state"]:
            state_value = int(state["lead_state"])
            carriers = int(state["n_conditioning_haplotypes"])
            empirical_available = bool(state["empirical_mode_available"])
            if empirical_available != (carriers > 0):
                problems.append(f"empirical availability/count mismatch at {key}, state={state_value}")
            if state_value == 1 and not empirical_available:
                empirical_alt_unavailable += 1
            if not state["ld_sign_available"]:
                ld_sign_unavailable += 1

            strategies = {row["strategy"]: row for row in state["strategies"]}
            if bool(strategies["empirical_mode"]["available"]) != empirical_available:
                problems.append(f"empirical strategy availability mismatch at {key}, state={state_value}")
            if bool(strategies["ld_sign"]["available"]) != bool(state["ld_sign_available"]):
                problems.append(f"LD-sign strategy availability mismatch at {key}, state={state_value}")
            if empirical_available and strategies["empirical_mode"]["exact_true_mode"] is None:
                problems.append(f"missing empirical error at {key}, state={state_value}")

            truth_oracle = state["truth_oracle"]
            if truth_oracle["certified"]:
                certified_truth_searches += 1
            else:
                problems.append(f"uncertified truth search at {key}, state={state_value}")
            if partners == 4:
                exhaustive = truth_oracle["exhaustive_check"]
                if not exhaustive["checked"] or not exhaustive["valid"]:
                    problems.append(f"failed truth exhaustive check at {key}, state={state_value}")
                nonunique_exact_top_ten += int(not exhaustive["unique_top_ten"])

            fitted_oracle = state["fitted_oracle"]
            if fit_available:
                if fitted_oracle is None or not fitted_oracle["certified"]:
                    problems.append(f"missing fitted certification at {key}, state={state_value}")
                else:
                    certified_fitted_searches += 1
                    if partners == 4:
                        exhaustive = fitted_oracle["exhaustive_check"]
                        if not exhaustive["checked"] or not exhaustive["valid"]:
                            problems.append(f"failed fitted exhaustive check at {key}, state={state_value}")
                        nonunique_exact_top_ten += int(not exhaustive["unique_top_ten"])
            elif strategies["fitted"]["available"]:
                problems.append(f"fitted strategy present without reportable fit at {key}, state={state_value}")

    result = {
        "status": "passed" if not problems else "failed",
        "n_files": len(documents),
        "n_expected_files": len(expected_cells),
        "n_fit_unavailable": fit_unavailable,
        "n_empirical_alt_unavailable": empirical_alt_unavailable,
        "n_ld_sign_state_unavailable": ld_sign_unavailable,
        "n_certified_truth_searches": certified_truth_searches,
        "n_certified_fitted_searches": certified_fitted_searches,
        "n_nonunique_four_partner_top_ten": nonunique_exact_top_ten,
        "n_problems": len(problems),
        "problems": problems[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
