#!/usr/bin/env python3
"""Aggregate the v77 rare/common lead-frequency ranking simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEAD_FREQUENCY_CLASSES = ("rare", "common")
SCENARIOS = ("lead_ld_q1", "balanced_q1", "lead_ld_q2")
SAMPLE_SIZES = (500, 1000, 2000)
PARTNER_COUNTS = (4, 32, 256)
STRATEGIES = ("fitted", "empirical_mode", "ld_sign")
REPLICATES = 30


def load(
    root: Path, allow_incomplete: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    paths = sorted(root.rglob("replicate_*.json"))
    documents = [json.loads(path.read_text()) for path in paths]
    failed = [document for document in documents if document.get("status") != "complete"]
    if failed:
        example = failed[0]
        raise ValueError(
            f"{len(failed)} failed files; first is "
            f"{example.get('lead_frequency_class')} {example.get('scenario')} "
            f"n={example.get('n_haplotypes')} k={example.get('n_partners')} "
            f"rep={example.get('replicate')}: {example.get('error')}"
        )
    expected = (
        len(LEAD_FREQUENCY_CLASSES)
        * len(SCENARIOS)
        * len(SAMPLE_SIZES)
        * len(PARTNER_COUNTS)
        * REPLICATES
    )
    if not allow_incomplete and len(documents) != expected:
        raise ValueError(f"expected {expected} complete files, found {len(documents)}")

    strategy_rows: list[dict] = []
    panel_rows: list[dict] = []
    for document in documents:
        frequency = document["frequency_design"]
        base = {
            "lead_frequency_class": document["lead_frequency_class"],
            "scenario": document["scenario"],
            "n_haplotypes": document["n_haplotypes"],
            "n_partners": document["n_partners"],
            "replicate": document["replicate"],
            "lead_alt_frequency": frequency["lead_alt_frequency"],
            "n_monomorphic_panel_variants": frequency[
                "n_monomorphic_panel_variants"
            ],
            "fit_available": document["fit"]["available"],
        }
        panel_rows.append(base | document["truth"] | document["fit"])
        for state in document["per_state"]:
            state_base = base | {
                "lead_state": state["lead_state"],
                "n_conditioning_haplotypes": state["n_conditioning_haplotypes"],
                "empirical_mode_available": state["empirical_mode_available"],
                "empirical_mode_maximum_count": state[
                    "empirical_mode_maximum_count"
                ],
                "n_tied_empirical_modes": state["n_tied_empirical_modes"],
                "true_mode_in_empirical_mode_set": state[
                    "true_mode_in_empirical_mode_set"
                ],
                "ld_sign_available": state["ld_sign_available"],
                "n_undefined_ld_sign_partners": state[
                    "n_undefined_ld_sign_partners"
                ],
                "true_top1_in_fitted_top10": state["true_top1_in_fitted_top10"],
                "top10_overlap_fraction": state["top10_overlap_fraction"],
            }
            for strategy in state["strategies"]:
                strategy_rows.append(state_base | strategy)
    return pd.DataFrame(strategy_rows), pd.DataFrame(panel_rows), documents


def summarize_alt_state(strategies: pd.DataFrame) -> pd.DataFrame:
    alt = strategies[strategies.lead_state == 1].copy()
    keys = [
        "lead_frequency_class",
        "scenario",
        "n_haplotypes",
        "n_partners",
        "strategy",
    ]
    rows = []
    for values, group in alt.groupby(keys, sort=True):
        available = group[group.available]
        empirical_evaluable = group[group.n_conditioning_haplotypes > 0]
        common_available = empirical_evaluable[empirical_evaluable.available]
        exact = available.exact_true_mode.astype(bool) if len(available) else pd.Series()
        common_exact = (
            common_available.exact_true_mode.astype(bool)
            if len(common_available)
            else pd.Series()
        )
        hamming = pd.to_numeric(available.hamming_to_true_mode, errors="coerce").dropna()
        row = dict(zip(keys, values))
        row |= {
            "n_panels": int(len(group)),
            "n_available": int(group.available.sum()),
            "unavailable_fraction": float(1.0 - group.available.mean()),
            "n_empirical_evaluable": int(len(empirical_evaluable)),
            "unavailable_fraction_among_empirical_evaluable": (
                None
                if not len(empirical_evaluable)
                else float(1.0 - empirical_evaluable.available.mean())
            ),
            "exact_recovery_given_available": (
                None if not len(available) else float(exact.mean())
            ),
            "error_given_available": (
                None if not len(available) else float(1.0 - exact.mean())
            ),
            "exact_recovery_on_empirical_evaluable_panels": (
                None if not len(common_available) else float(common_exact.mean())
            ),
            "error_on_empirical_evaluable_panels": (
                None if not len(common_available) else float(1.0 - common_exact.mean())
            ),
            "unconditional_correct_fraction": float(
                np.mean(group.available.astype(bool) & group.exact_true_mode.eq(True))
            ),
            "median_hamming_given_available": (
                None if not len(hamming) else float(hamming.median())
            ),
            "p95_hamming_given_available": (
                None if not len(hamming) else float(np.percentile(hamming, 95))
            ),
            "median_alt_carriers": float(group.n_conditioning_haplotypes.median()),
        }
        if values[-1] == "empirical_mode":
            empirical_available = group[group.empirical_mode_available]
            row |= {
                "zero_alt_carrier_fraction": float(
                    np.mean(group.n_conditioning_haplotypes == 0)
                ),
                "tie_fraction_given_available": (
                    None
                    if not len(empirical_available)
                    else float(np.mean(empirical_available.n_tied_empirical_modes > 1))
                ),
                "true_mode_in_empirical_mode_set_given_available": (
                    None
                    if not len(empirical_available)
                    else float(
                        empirical_available.true_mode_in_empirical_mode_set.mean()
                    )
                ),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_panels(panels: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "lead_frequency_class",
        "scenario",
        "n_haplotypes",
        "n_partners",
    ]
    rows = []
    for values, group in panels.groupby(keys, sort=True):
        rows.append(
            dict(zip(keys, values))
            | {
                "n_panels": int(len(group)),
                "fit_unavailable_fraction": float(1.0 - group.fit_available.mean()),
                "median_lead_alt_frequency": float(group.lead_alt_frequency.median()),
                "median_monomorphic_panel_variants": float(
                    group.n_monomorphic_panel_variants.median()
                ),
                "fraction_with_any_monomorphic_variant": float(
                    np.mean(group.n_monomorphic_panel_variants > 0)
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_lead_availability(strategies: pd.DataFrame) -> pd.DataFrame:
    empirical = strategies[
        (strategies.lead_state == 1) & (strategies.strategy == "empirical_mode")
    ].copy()
    rows = []
    for values, group in empirical.groupby(
        ["lead_frequency_class", "n_haplotypes"], sort=True
    ):
        available = group[group.empirical_mode_available]
        rows.append(
            {
                "lead_frequency_class": str(values[0]),
                "n_haplotypes": int(values[1]),
                "n_panels": int(len(group)),
                "zero_alt_carrier_fraction": float(
                    np.mean(group.n_conditioning_haplotypes == 0)
                ),
                "median_alt_carriers": float(group.n_conditioning_haplotypes.median()),
                "p05_alt_carriers": float(
                    np.percentile(group.n_conditioning_haplotypes, 5)
                ),
                "p95_alt_carriers": float(
                    np.percentile(group.n_conditioning_haplotypes, 95)
                ),
                "tie_fraction_given_available": (
                    None
                    if not len(available)
                    else float(np.mean(available.n_tied_empirical_modes > 1))
                ),
            }
        )
    return pd.DataFrame(rows)


def json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    strategies, panels, documents = load(args.input_root, args.allow_incomplete)
    alt_summary = summarize_alt_state(strategies)
    panel_summary = summarize_panels(panels)
    lead_availability = summarize_lead_availability(strategies)
    strategies.to_csv(args.output_dir / "ranking_simulation_strategy_rows_v77.csv", index=False)
    panels.to_csv(args.output_dir / "ranking_simulation_panel_rows_v77.csv", index=False)
    alt_summary.to_csv(args.output_dir / "ranking_simulation_alt_summary_v77.csv", index=False)
    panel_summary.to_csv(
        args.output_dir / "ranking_simulation_panel_summary_v77.csv", index=False
    )
    lead_availability.to_csv(
        args.output_dir / "ranking_simulation_lead_availability_v77.csv", index=False
    )
    payload = {
        "n_complete_files": len(documents),
        "design": {
            "lead_frequency_classes": list(LEAD_FREQUENCY_CLASSES),
            "rare_alt_frequency": "log-uniform on [0.005, 0.05)",
            "common_alt_frequency": "uniform on [0.05, 0.40]",
            "partner_frequency_mix": "exactly 50% rare and 50% common",
            "scenarios": list(SCENARIOS),
            "sample_sizes": list(SAMPLE_SIZES),
            "partner_counts": list(PARTNER_COUNTS),
            "replicates": REPLICATES,
        },
        "alt_state_summary": json_records(alt_summary),
        "panel_summary": json_records(panel_summary),
        "lead_availability_summary": json_records(lead_availability),
    }
    (args.output_dir / "ranking_simulation_v77.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(json.dumps({"n_complete_files": len(documents)}, indent=2))


if __name__ == "__main__":
    main()
