#!/usr/bin/env python3
"""Compare public HaploPerturb, empirical-mode, LD-sign, and single-lead strategies.

The empirical and LD-sign state-specific backgrounds are mapped to their exact
existing fitted top-ten ranks using the v73 manifest.  This permits mixed-rank
lead-state contrasts without new AlphaGenome calls.  All strategies are
evaluated on one locus-gene universe and resampled jointly by whole locus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_single_lead_enrichment_v64 import CELL_LABEL, CELL_MAP, add_truth, enrich


SEED = 750804
N_BOOT = 10_000
STRATEGIES = ("fitted_top1", "empirical_mode", "ld_sign", "single_lead")
COLORS = {
    "fitted_top1": "#0072B2",
    "empirical_mode": "#009E73",
    "ld_sign": "#CC79A7",
    "single_lead": "#D55E00",
}
LABELS = {
    "fitted_top1": "HaploPerturb top one",
    "empirical_mode": "Empirical mode",
    "ld_sign": "LD sign",
    "single_lead": "Single lead",
}


def baseline_rank_map(path: Path, strategy: str) -> dict[tuple[str, str, int], int]:
    document = json.loads(path.read_text())
    result = {}
    for locus in document["per_locus"]:
        for row in locus["rows"]:
            if row["strategy"] != strategy:
                continue
            rank = row["fitted_top10_rank"]
            if rank is None:
                raise ValueError(
                    f"{strategy} absent from fitted top ten: {row['arm']} {row['locus']} {row['lead_state']}"
                )
            result[(row["arm"], row["locus"], int(row["lead_state"]))] = int(rank)
    return result


def fitted_rank_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    reverse = {value: key for key, value in CELL_MAP.items()}
    frame = frame.loc[frame.biosample_name.isin(reverse)].copy()
    frame["celltype"] = frame.biosample_name.map(reverse)
    keys = ["arm", "locus", "celltype", "gene_name", "rank"]
    return frame.groupby(keys, as_index=False).agg(
        pred0=("pred_mean_lead0", "mean"), pred1=("pred_mean_lead1", "mean")
    )


def assemble_fitted_strategy(
    frame: pd.DataFrame,
    strategy: str,
    ranks0: dict[tuple[str, str, int], int] | None = None,
    ranks1: dict[tuple[str, str, int], int] | None = None,
) -> pd.DataFrame:
    keys = ["arm", "locus", "celltype", "gene_name"]
    if strategy == "fitted_top1":
        selected = frame.loc[frame["rank"].eq(1)].copy()
        selected["rank0"] = 1
        selected["rank1"] = 1
        selected = selected.drop(columns="rank")
    else:
        if ranks0 is None or ranks1 is None:
            raise ValueError("state-specific rank maps are required")
        base = frame[keys].drop_duplicates()
        base["rank0"] = [ranks0[(arm, locus, 0)] for arm, locus in zip(base.arm, base.locus)]
        base["rank1"] = [ranks1[(arm, locus, 1)] for arm, locus in zip(base.arm, base.locus)]
        left = frame.rename(columns={"rank": "rank0"})[keys + ["rank0", "pred0"]]
        right = frame.rename(columns={"rank": "rank1"})[keys + ["rank1", "pred1"]]
        selected = base.merge(left, on=keys + ["rank0"], validate="one_to_one")
        selected = selected.merge(right, on=keys + ["rank1"], validate="one_to_one")
    selected[f"score_{strategy}"] = np.abs(
        np.log1p(selected.pred1) - np.log1p(selected.pred0)
    )
    return selected[keys + ["rank0", "rank1", f"score_{strategy}"]]


def assemble_single(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    reverse = {value: key for key, value in CELL_MAP.items()}
    frame = frame.loc[frame.biosample_name.isin(reverse)].copy()
    frame["celltype"] = frame.biosample_name.map(reverse)
    keys = ["arm", "locus", "celltype", "gene_name", "rank"]
    frame = frame.groupby(keys, as_index=False).agg(
        pred0=("pred_mean_lead0", "mean"), pred1=("pred_mean_lead1", "mean")
    )
    if not frame["rank"].eq(1).all():
        raise ValueError("single-lead input must contain rank one only")
    frame["score_single_lead"] = np.abs(np.log1p(frame.pred1) - np.log1p(frame.pred0))
    return frame[["arm", "locus", "celltype", "gene_name", "score_single_lead"]]


def common_universe(
    fitted_path: Path,
    single_path: Path,
    baseline_path: Path,
    truth_path: Path,
    variants_path: Path,
) -> pd.DataFrame:
    fitted = fitted_rank_predictions(fitted_path)
    frames = [assemble_fitted_strategy(fitted, "fitted_top1")]
    for strategy in ("empirical_mode", "ld_sign"):
        mapping = baseline_rank_map(baseline_path, strategy)
        frames.append(assemble_fitted_strategy(fitted, strategy, mapping, mapping))
    frames.append(assemble_single(single_path))
    keys = ["arm", "locus", "celltype", "gene_name"]
    paired = frames[0]
    for frame in frames[1:]:
        paired = paired.merge(frame, on=keys, how="inner", validate="one_to_one")
    paired = add_truth(paired, truth_path, variants_path)
    score_columns = [f"score_{strategy}" for strategy in STRATEGIES]
    if paired[score_columns].isna().any().any():
        raise ValueError("missing strategy scores")
    return paired


def bootstrap_group(frame: pd.DataFrame, rng: np.random.Generator) -> tuple[dict, list[dict]]:
    loci = sorted(frame.locus.unique())
    draws = {strategy: np.empty(N_BOOT, dtype=float) for strategy in STRATEGIES}
    for index in range(N_BOOT):
        sampled = rng.choice(loci, size=len(loci), replace=True)
        sample = pd.concat(
            [frame.loc[frame.locus.eq(locus)] for locus in sampled], ignore_index=True
        )
        for strategy in STRATEGIES:
            draws[strategy][index] = enrich(sample, f"score_{strategy}")["enrichment"]
    summaries = {
        strategy: {
            "boot_lo": float(np.percentile(values, 2.5)),
            "boot_hi": float(np.percentile(values, 97.5)),
            "boot_frac_le_1": float(np.mean(values <= 1)),
        }
        for strategy, values in draws.items()
    }
    observed = {
        strategy: enrich(frame, f"score_{strategy}")["enrichment"]
        for strategy in STRATEGIES
    }
    comparisons = []
    for left_index, left in enumerate(STRATEGIES):
        for right in STRATEGIES[left_index + 1 :]:
            difference = draws[left] - draws[right]
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "difference": float(observed[left] - observed[right]),
                    "diff_boot_lo": float(np.percentile(difference, 2.5)),
                    "diff_boot_hi": float(np.percentile(difference, 97.5)),
                    "diff_boot_frac_le_0": float(np.mean(difference <= 0)),
                }
            )
    return summaries, comparisons


def add_ranks_and_calls(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for strategy in STRATEGIES:
        score = f"score_{strategy}"
        output[f"rank_{strategy}"] = output.groupby(["arm", "celltype"])[score].rank(
            method="first", ascending=False
        )
        k_by_group = output.groupby(["arm", "celltype"])["truth"].transform("sum")
        output[f"called_{strategy}"] = output[f"rank_{strategy}"] <= k_by_group
    return output


def influence_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for (arm, celltype), group in frame.groupby(["arm", "celltype"], sort=True):
        full = {
            strategy: enrich(group, f"score_{strategy}")["enrichment"]
            for strategy in STRATEGIES
        }
        for locus in sorted(group.locus.unique()):
            reduced = group.loc[~group.locus.eq(locus)]
            for strategy in STRATEGIES:
                leaveout = enrich(reduced, f"score_{strategy}")["enrichment"]
                rows.append(
                    {
                        "arm": arm,
                        "celltype": celltype,
                        "locus": locus,
                        "strategy": strategy,
                        "full_enrichment": full[strategy],
                        "leave_one_locus_out_enrichment": leaveout,
                        "influence_full_minus_leaveout": full[strategy] - leaveout,
                        "n_pairs_removed": int(group.locus.eq(locus).sum()),
                        "n_positives_removed": int(
                            group.loc[group.locus.eq(locus), "truth"].sum()
                        ),
                    }
                )
    return rows


def plot(rows: list[dict], output: Path, cells: list[str], width: float) -> None:
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(cells), figsize=(width, 2.8), sharey=True, squeeze=False)
    offsets = np.linspace(-0.27, 0.27, len(STRATEGIES))
    for ax, cell in zip(axes[0], cells):
        subset = frame.loc[frame.celltype.eq(cell)]
        for offset, strategy in zip(offsets, STRATEGIES):
            selected = subset.loc[subset.strategy.eq(strategy)].set_index("arm").loc[["armA", "armB"]]
            x = np.array([0, 1]) + offset
            estimate = selected.enrichment.to_numpy()
            lower = selected.boot_lo.to_numpy()
            upper = selected.boot_hi.to_numpy()
            ax.errorbar(
                x,
                estimate,
                yerr=[estimate - lower, upper - estimate],
                fmt="o",
                color=COLORS[strategy],
                capsize=3,
                label=LABELS[strategy],
            )
        ax.axhline(1, color="0.45", linestyle="--", linewidth=0.9)
        ax.set_xticks([0, 1], ["Arm A", "Arm B"])
        ax.set_title(CELL_LABEL[cell])
        ax.set_ylabel("Matched-count enrichment" if cell == cells[0] else "")
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fitted-top10", type=Path, required=True)
    parser.add_argument("--single-lead", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--egene-truth", type=Path, required=True)
    parser.add_argument("--sig-variants", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use(args.style)

    paired = common_universe(
        args.fitted_top10,
        args.single_lead,
        args.baseline_manifest,
        args.egene_truth,
        args.sig_variants,
    )
    rng = np.random.default_rng(SEED)
    rows = []
    comparisons = []
    for (arm, celltype), group in paired.groupby(["arm", "celltype"], sort=True):
        boots, contrasts = bootstrap_group(group, rng)
        for strategy in STRATEGIES:
            rows.append(
                {
                    "arm": arm,
                    "celltype": celltype,
                    "strategy": strategy,
                    **enrich(group, f"score_{strategy}"),
                    **boots[strategy],
                }
            )
        comparisons.extend(
            {"arm": arm, "celltype": celltype, **contrast} for contrast in contrasts
        )

    audit = add_ranks_and_calls(paired)
    influence = pd.DataFrame(influence_rows(paired))
    audit.to_csv(args.output_dir / "public_baseline_locus_gene_audit_v75.csv", index=False)
    influence.to_csv(args.output_dir / "public_baseline_locus_influence_v75.csv", index=False)
    plot(rows, args.output_dir / "figure_microglia_public_baselines_v75.pdf", ["Mic"], 4.2)
    plot(rows, args.output_dir / "supp_public_baselines_all_cells_v75.pdf", ["Ast", "Exc", "Mic"], 7.2)
    document = {
        "script": Path(__file__).name,
        "seed": SEED,
        "n_locus_bootstrap": N_BOOT,
        "definition": {
            "baseline_assembly": "state-specific empirical/LD-sign ranks selected from existing public fitted top-ten AlphaGenome predictions; no new model query",
            "universe": "intersection of all four strategy-covered locus-gene pairs and Fujita-tested genes",
            "truth": "Fujita two-step-FDR significant variant-gene pair inside the same AlphaGenome window",
            "resampling": "paired whole-locus bootstrap",
        },
        "rows": rows,
        "paired_contrasts": comparisons,
        "locus_influence_summary": {
            "maximum_absolute_influence": float(
                influence.influence_full_minus_leaveout.abs().max()
            ),
            "microglia_armB_fitted": influence.loc[
                (influence.arm == "armB")
                & (influence.celltype == "Mic")
                & (influence.strategy == "fitted_top1")
            ].sort_values("influence_full_minus_leaveout", key=np.abs, ascending=False).head(10).to_dict("records"),
        },
    }
    (args.output_dir / "public_baseline_enrichment_v75.json").write_text(
        json.dumps(document, indent=2)
    )
    print(json.dumps({"rows": rows, "n_audit_rows": len(audit)}, indent=2))


if __name__ == "__main__":
    main()
