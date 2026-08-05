#!/usr/bin/env python3
"""Compare ROS/MAP- and 1000 Genomes-derived AlphaGenome eQTL rankings.

All strategies are restricted to one common locus-gene universe and evaluated
against the same Fujita two-step-FDR truth.  Whole lead windows are resampled
jointly, so the public-minus-donor-panel intervals are paired by locus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_single_lead_enrichment_v64 import (
    CELL_LABEL,
    CELL_MAP,
    N_BOOT,
    SEED,
    add_truth,
    enrich,
)


RANKS = (1, 5, 10)
PANEL_COLOR = {"onekg": "#0072B2", "rosmap": "#E69F00"}
PANEL_LABEL = {"onekg": "1000 Genomes", "rosmap": "ROS/MAP"}
ARM_LABEL = {"armA": "Arm A", "armB": "Arm B"}


def collapse_fitted(path: Path, panel: str, max_rank: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    reverse = {value: key for key, value in CELL_MAP.items()}
    frame = frame.loc[
        frame.biosample_name.isin(reverse) & frame["rank"].le(max_rank)
    ].copy()
    frame["celltype"] = frame.biosample_name.map(reverse)
    keys = ["arm", "locus", "celltype", "gene_name", "rank"]
    frame = frame.groupby(keys, as_index=False).agg(
        pred0=("pred_mean_lead0", "mean"),
        pred1=("pred_mean_lead1", "mean"),
    )
    frame = frame.groupby(
        ["arm", "locus", "celltype", "gene_name"], as_index=False
    ).agg(pred0=("pred0", "mean"), pred1=("pred1", "mean"))
    strategy = f"{panel}_top{max_rank}"
    frame[f"score_{strategy}"] = np.abs(
        np.log1p(frame.pred1) - np.log1p(frame.pred0)
    )
    return frame[
        ["arm", "locus", "celltype", "gene_name", f"score_{strategy}"]
    ]


def collapse_single(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    reverse = {value: key for key, value in CELL_MAP.items()}
    frame = frame.loc[frame.biosample_name.isin(reverse)].copy()
    frame["celltype"] = frame.biosample_name.map(reverse)
    keys = ["arm", "locus", "celltype", "gene_name", "rank"]
    frame = frame.groupby(keys, as_index=False).agg(
        pred0=("pred_mean_lead0", "mean"),
        pred1=("pred_mean_lead1", "mean"),
    )
    if not frame["rank"].eq(1).all():
        raise ValueError("Single-lead predictions must contain rank 1 only")
    frame = frame.drop(columns="rank")
    frame["score_single_lead"] = np.abs(
        np.log1p(frame.pred1) - np.log1p(frame.pred0)
    )
    return frame[
        ["arm", "locus", "celltype", "gene_name", "score_single_lead"]
    ]


def assemble_common_universe(
    rosmap_top10: Path,
    onekg_top10: Path,
    single_lead: Path,
    truth_path: Path,
    variants_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    strategies: list[str] = []
    frames: list[pd.DataFrame] = []
    for panel, path in (("onekg", onekg_top10), ("rosmap", rosmap_top10)):
        for rank in RANKS:
            strategy = f"{panel}_top{rank}"
            strategies.append(strategy)
            frames.append(collapse_fitted(path, panel, rank))
    strategies.append("single_lead")
    frames.append(collapse_single(single_lead))

    keys = ["arm", "locus", "celltype", "gene_name"]
    paired = frames[0]
    for frame in frames[1:]:
        paired = paired.merge(
            frame, on=keys, how="inner", validate="one_to_one"
        )
    paired = add_truth(paired, truth_path, variants_path)
    score_columns = [f"score_{strategy}" for strategy in strategies]
    if paired[score_columns].isna().any().any():
        raise ValueError("Missing scores in the common comparison universe")
    return paired, strategies


def bootstrap_group(
    frame: pd.DataFrame,
    strategies: list[str],
    rng: np.random.Generator,
) -> tuple[dict[str, dict], list[dict]]:
    loci = sorted(frame.locus.unique())
    draws = {
        strategy: np.empty(N_BOOT, dtype=float) for strategy in strategies
    }
    for index in range(N_BOOT):
        sampled_loci = rng.choice(loci, size=len(loci), replace=True)
        sample = pd.concat(
            [frame.loc[frame.locus.eq(locus)] for locus in sampled_loci],
            ignore_index=True,
        )
        for strategy in strategies:
            draws[strategy][index] = enrich(
                sample, f"score_{strategy}"
            )["enrichment"]

    summaries = {}
    for strategy, values in draws.items():
        summaries[strategy] = {
            "boot_lo": float(np.percentile(values, 2.5)),
            "boot_hi": float(np.percentile(values, 97.5)),
            "boot_frac_le_1": float(np.mean(values <= 1)),
        }

    comparisons: list[dict] = []
    observed = {
        strategy: enrich(frame, f"score_{strategy}")["enrichment"]
        for strategy in strategies
    }
    for left_index, left in enumerate(strategies):
        for right in strategies[left_index + 1 :]:
            difference = draws[left] - draws[right]
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "difference": float(observed[left] - observed[right]),
                    "diff_boot_lo": float(np.percentile(difference, 2.5)),
                    "diff_boot_hi": float(np.percentile(difference, 97.5)),
                    "diff_boot_frac_le_0": float(
                        np.mean(difference <= 0)
                    ),
                }
            )
    return summaries, comparisons


def lookup_rows(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {
        (row["arm"], row["celltype"], row["strategy"]): row for row in rows
    }


def lookup_differences(
    comparisons: list[dict],
) -> dict[tuple[str, str, str, str], dict]:
    return {
        (
            row["arm"],
            row["celltype"],
            row["left"],
            row["right"],
        ): row
        for row in comparisons
    }


def plot_microglia(
    rows: list[dict], comparisons: list[dict], output: Path
) -> None:
    values = lookup_rows(rows)
    differences = lookup_differences(comparisons)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(6.3, 4.8),
        sharex="col",
        gridspec_kw={"height_ratios": [1.35, 1]},
    )
    x = np.arange(len(RANKS), dtype=float)
    offsets = {"onekg": -0.10, "rosmap": 0.10}
    for column, arm in enumerate(("armA", "armB")):
        ax = axes[0, column]
        for panel in ("onekg", "rosmap"):
            panel_rows = [
                values[(arm, "Mic", f"{panel}_top{rank}")] for rank in RANKS
            ]
            estimate = np.array([row["enrichment"] for row in panel_rows])
            lower = np.array([row["boot_lo"] for row in panel_rows])
            upper = np.array([row["boot_hi"] for row in panel_rows])
            ax.errorbar(
                x + offsets[panel],
                estimate,
                yerr=[estimate - lower, upper - estimate],
                fmt="o",
                color=PANEL_COLOR[panel],
                capsize=3,
                label=PANEL_LABEL[panel],
            )
        single = values[(arm, "Mic", "single_lead")]
        ax.axhline(1, color="0.45", linestyle="--", linewidth=0.9)
        ax.axhline(
            single["enrichment"],
            color="0.45",
            linestyle=":",
            linewidth=1,
            label="Single lead" if column == 0 else None,
        )
        ax.set_title(ARM_LABEL[arm])
        ax.set_ylabel("Matched-count enrichment" if column == 0 else "")
        ax.grid(axis="y", alpha=0.3)

        diff_ax = axes[1, column]
        panel_differences = []
        for rank in RANKS:
            key = (
                arm,
                "Mic",
                f"onekg_top{rank}",
                f"rosmap_top{rank}",
            )
            panel_differences.append(differences[key])
        estimate = np.array(
            [row["difference"] for row in panel_differences]
        )
        lower = np.array(
            [row["diff_boot_lo"] for row in panel_differences]
        )
        upper = np.array(
            [row["diff_boot_hi"] for row in panel_differences]
        )
        diff_ax.errorbar(
            x,
            estimate,
            yerr=[estimate - lower, upper - estimate],
            fmt="o",
            color="#333333",
            capsize=3,
        )
        diff_ax.axhline(0, color="0.45", linestyle="--", linewidth=0.9)
        diff_ax.set_xticks(x, ["Top 1", "Top 5", "Top 10"])
        diff_ax.set_ylabel(
            "1000 Genomes minus\nROS/MAP enrichment"
            if column == 0
            else ""
        )
        diff_ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_all_cells(rows: list[dict], output: Path) -> None:
    values = lookup_rows(rows)
    fig, axes = plt.subplots(
        3, 2, figsize=(6.5, 7.0), sharex=True, squeeze=False
    )
    x = np.arange(len(RANKS), dtype=float)
    offsets = {"onekg": -0.10, "rosmap": 0.10}
    for row_index, celltype in enumerate(("Ast", "Exc", "Mic")):
        for column, arm in enumerate(("armA", "armB")):
            ax = axes[row_index, column]
            for panel in ("onekg", "rosmap"):
                panel_rows = [
                    values[(arm, celltype, f"{panel}_top{rank}")]
                    for rank in RANKS
                ]
                estimate = np.array(
                    [row["enrichment"] for row in panel_rows]
                )
                lower = np.array([row["boot_lo"] for row in panel_rows])
                upper = np.array([row["boot_hi"] for row in panel_rows])
                ax.errorbar(
                    x + offsets[panel],
                    estimate,
                    yerr=[estimate - lower, upper - estimate],
                    fmt="o",
                    color=PANEL_COLOR[panel],
                    capsize=3,
                    label=PANEL_LABEL[panel],
                )
            ax.axhline(1, color="0.45", linestyle="--", linewidth=0.9)
            ax.grid(axis="y", alpha=0.3)
            if row_index == 0:
                ax.set_title(ARM_LABEL[arm])
            if column == 0:
                ax.set_ylabel(
                    f"{CELL_LABEL[celltype]}\nmatched-count enrichment"
                )
            if row_index == 2:
                ax.set_xticks(x, ["Top 1", "Top 5", "Top 10"])
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosmap-top10", type=Path, required=True)
    parser.add_argument("--onekg-top10", type=Path, required=True)
    parser.add_argument("--single-lead", type=Path, required=True)
    parser.add_argument("--egene-truth", type=Path, required=True)
    parser.add_argument("--sig-variants", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use(args.style)

    paired, strategies = assemble_common_universe(
        args.rosmap_top10,
        args.onekg_top10,
        args.single_lead,
        args.egene_truth,
        args.sig_variants,
    )
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    comparisons: list[dict] = []
    for (arm, celltype), group in paired.groupby(
        ["arm", "celltype"], sort=True
    ):
        bootstrap, contrasts = bootstrap_group(group, strategies, rng)
        for strategy in strategies:
            rows.append(
                {
                    "arm": arm,
                    "celltype": celltype,
                    "strategy": strategy,
                    **enrich(group, f"score_{strategy}"),
                    **bootstrap[strategy],
                }
            )
        comparisons.extend(
            {
                "arm": arm,
                "celltype": celltype,
                **comparison,
            }
            for comparison in contrasts
        )

    plot_microglia(
        rows,
        comparisons,
        args.output_dir
        / "figure_microglia_reference_panel_comparison_v68.pdf",
    )
    plot_all_cells(
        rows,
        args.output_dir
        / "supp_reference_panel_comparison_all_cells_v68.pdf",
    )
    summary = {
        "script": Path(__file__).name,
        "seed": SEED,
        "n_locus_bootstrap": N_BOOT,
        "unit": "locus-gene pair",
        "score": "absolute log1p lead-state contrast",
        "truth": (
            "Fujita two-step-FDR significant variant-gene pair inside the "
            "same 1-Mb window"
        ),
        "universe": (
            "intersection across ROS/MAP top 1/5/10, 1000 Genomes top "
            "1/5/10, single lead, and Fujita-tested genes"
        ),
        "strategies": strategies,
        "rows": rows,
        "paired_contrasts": comparisons,
    }
    output_json = args.output_dir / "reference_panel_enrichment_v68.json"
    output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
