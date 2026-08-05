#!/usr/bin/env python3
"""Matched-count eQTL enrichment: HaploPerturb top-five, top-one, and single-lead flip.

All comparisons use the intersection of AlphaGenome-covered locus--gene pairs
and Fujita-tested genes. Truth is locus-specific: a Fujita two-step-FDR
significant variant--gene pair must fall inside the same 1,048,576-bp window.
Whole loci are resampled jointly for all three strategies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

CELL_MAP = {"Ast": "astrocyte", "Exc": "glutamatergic neuron", "Mic": "CD14-positive monocyte"}
CELL_LABEL = {"Ast": "Astrocyte", "Exc": "Excitatory neuron", "Mic": "Microglia proxy"}
SEED = 640731
N_BOOT = 10_000


def collapse(path: Path, strategy: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    reverse = {v: k for k, v in CELL_MAP.items()}
    d = d.loc[d.biosample_name.isin(reverse)].copy()
    d["celltype"] = d.biosample_name.map(reverse)
    keys = ["arm", "locus", "celltype", "gene_name", "rank"]
    d = d.groupby(keys, as_index=False).agg(
        pred0=("pred_mean_lead0", "mean"), pred1=("pred_mean_lead1", "mean")
    )
    if strategy in {"top5", "top10"}:
        d = d.groupby(["arm", "locus", "celltype", "gene_name"], as_index=False).agg(
            pred0=("pred0", "mean"), pred1=("pred1", "mean")
        )
    elif strategy == "top1":
        d = d.loc[d["rank"].eq(1)].drop(columns="rank")
    elif strategy == "single_lead":
        if not d["rank"].eq(1).all():
            raise ValueError("single-lead predictions must have rank 1 only")
        d = d.drop(columns="rank")
    else:
        raise ValueError(strategy)
    d["score"] = np.abs(np.log1p(d.pred1) - np.log1p(d.pred0))
    return d[["arm", "locus", "celltype", "gene_name", "score"]]


def add_truth(d: pd.DataFrame, truth_path: Path, variants_path: Path) -> pd.DataFrame:
    truth = pd.read_csv(truth_path).rename(columns={"gene_symbol": "gene_name"})
    truth["is_egene"] = truth.is_egene.astype(str).str.lower().eq("true")
    variants = pd.read_csv(variants_path).rename(columns={"gene_symbol": "gene_name"})
    split = variants.variant_id.str.split(":", expand=True)
    variants["chrom"], variants["pos"] = split[0].astype(str), split[1].astype(int)
    sig = {
        (cell, gene): list(zip(g.chrom.astype(str), g.pos.astype(int)))
        for (cell, gene), g in variants.groupby(["celltype", "gene_name"])
    }
    out = d.merge(truth[["gene_name", "celltype", "is_egene"]],
                  on=["gene_name", "celltype"], how="inner")

    def positive(row) -> bool:
        chrom, pos, _, _ = row.locus.split("_", 3)
        start = max(1, int(pos) - 524_288)
        end = start + 1_048_576
        return any(c == chrom and start <= p < end
                   for c, p in sig.get((row.celltype, row.gene_name), []))

    out["truth"] = out.apply(positive, axis=1)
    return out


def enrich(d: pd.DataFrame, score_col: str) -> dict:
    m, k = len(d), int(d.truth.sum())
    if not 0 < k < m:
        raise ValueError((m, k))
    top = d.sort_values([score_col, "locus", "gene_name"],
                        ascending=[False, True, True], kind="mergesort").head(k)
    tp = int(top.truth.sum())
    precision = tp / k
    base = k / m
    return {
        "M": m, "K": k, "tp": tp, "precision": precision, "recall": precision,
        "base_rate": base, "enrichment": precision / base,
        "hyper_p": float(hypergeom.sf(tp - 1, m, k, k)),
        "n_loci": int(d.locus.nunique()),
        "n_loci_tp": int(top.loc[top.truth, "locus"].nunique()),
    }


def bootstrap_strategies(d: pd.DataFrame, rng: np.random.Generator,
                         strategies: list[str]) -> tuple[dict, list[dict]]:
    loci = sorted(d.locus.unique())
    draws = {strategy: [] for strategy in strategies}
    for _ in range(N_BOOT):
        picks = rng.choice(loci, size=len(loci), replace=True)
        sample = pd.concat([d.loc[d.locus.eq(x)] for x in picks], ignore_index=True)
        for strategy in strategies:
            draws[strategy].append(enrich(sample, f"score_{strategy}")["enrichment"])
    draws = {strategy: np.asarray(values) for strategy, values in draws.items()}

    def summary(x: np.ndarray) -> dict:
        return {
            "boot_lo": float(np.percentile(x, 2.5)),
            "boot_hi": float(np.percentile(x, 97.5)),
            "boot_frac_le_1": float(np.mean(x <= 1)),
        }

    comparisons = []
    pairs = [(left, right) for left in strategies for right in strategies
             if strategies.index(left) < strategies.index(right)]
    for left, right in pairs:
        diff = draws[left] - draws[right]
        comparisons.append({
            "left": left, "right": right,
            "difference": float(enrich(d, f"score_{left}")["enrichment"]
                                - enrich(d, f"score_{right}")["enrichment"]),
            "diff_boot_lo": float(np.percentile(diff, 2.5)),
            "diff_boot_hi": float(np.percentile(diff, 97.5)),
            "diff_boot_frac_le_0": float(np.mean(diff <= 0)),
        })
    return {strategy: summary(values) for strategy, values in draws.items()}, comparisons


def plot(rows: list[dict], output: Path, cells: list[str], width: float) -> None:
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(cells), figsize=(width, 2.8), sharey=True, squeeze=False)
    colors = {"top10": "#CC79A7", "top5": "#0072B2", "top1": "#009E73",
              "single_lead": "#D55E00"}
    labels = {"top10": "HaploPerturb top-ten", "top5": "HaploPerturb top-five", "top1": "HaploPerturb top-one",
              "single_lead": "Single lead"}
    strategies = [x for x in ["top10", "top5", "top1", "single_lead"]
                  if x in set(frame.strategy)]
    for ax, cell in zip(axes[0], cells):
        sub = frame.loc[frame.celltype.eq(cell)]
        offsets = np.linspace(-0.27, 0.27, len(strategies))
        for offset, strategy in zip(offsets, strategies):
            s = sub.loc[sub.strategy.eq(strategy)].set_index("arm").loc[["armA", "armB"]]
            x = np.array([0, 1]) + offset
            y = s.enrichment.to_numpy()
            lo, hi = s.boot_lo.to_numpy(), s.boot_hi.to_numpy()
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", color=colors[strategy],
                        capsize=3, label=labels[strategy])
        ax.axhline(1, color="0.45", linestyle="--", linewidth=0.9)
        ax.set_xticks([0, 1], ["Arm A", "Arm B"])
        ax.set_title(CELL_LABEL[cell])
        ax.set_ylabel("Matched-count enrichment" if cell == cells[0] else "")
        ax.grid(axis="y", alpha=0.35)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top5", type=Path, required=True)
    ap.add_argument("--top10", type=Path)
    ap.add_argument("--single-lead", type=Path, required=True)
    ap.add_argument("--egene-truth", type=Path, required=True)
    ap.add_argument("--sig-variants", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--style", type=Path, required=True)
    ap.add_argument("--artifact-tag", default="v64")
    ap.add_argument("--figure-stem", default="strategy_enrichment")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use(args.style)

    top5 = add_truth(collapse(args.top5, "top5"), args.egene_truth, args.sig_variants)
    top1 = add_truth(collapse(args.top5, "top1"), args.egene_truth, args.sig_variants)
    single = add_truth(collapse(args.single_lead, "single_lead"), args.egene_truth, args.sig_variants)
    inputs = [("top5", top5), ("top1", top1), ("single_lead", single)]
    if args.top10:
        inputs.insert(0, ("top10", add_truth(collapse(args.top10, "top10"),
                                             args.egene_truth, args.sig_variants)))
    keys = ["arm", "locus", "celltype", "gene_name"]
    paired = None
    for strategy, frame in inputs:
        part = frame[keys + ["score", "truth"]].rename(
            columns={"score": f"score_{strategy}", "truth": f"truth_{strategy}"})
        paired = part if paired is None else paired.merge(part, on=keys, how="inner",
                                                          validate="one_to_one")
    truth_cols = [f"truth_{strategy}" for strategy, _ in inputs]
    if not all(paired[truth_cols[0]].eq(paired[col]).all() for col in truth_cols[1:]):
        raise ValueError("truth mismatch between strategies")
    paired = paired.rename(columns={truth_cols[0]: "truth"}).drop(columns=truth_cols[1:])
    strategies = [strategy for strategy, _ in inputs]

    rng = np.random.default_rng(SEED)
    rows, comparisons = [], []
    for (arm, cell), group in paired.groupby(["arm", "celltype"], sort=True):
        boots, contrasts = bootstrap_strategies(group, rng, strategies)
        for strategy in strategies:
            rows.append({"arm": arm, "celltype": cell, "strategy": strategy,
                         **enrich(group, f"score_{strategy}"), **boots[strategy]})
        comparisons.extend({"arm": arm, "celltype": cell, **contrast}
                           for contrast in contrasts)

    plot(rows, args.output_dir / f"figure_microglia_{args.figure_stem}_{args.artifact_tag}.pdf",
         ["Mic"], 3.7)
    plot(rows, args.output_dir / f"supp_{args.figure_stem}_all_cells_{args.artifact_tag}.pdf",
         ["Ast", "Exc", "Mic"], 7.2)
    summary = {
        "script": Path(__file__).name, "seed": SEED, "n_locus_bootstrap": N_BOOT,
        "unit": "locus-gene pair", "score": "absolute log1p lead-state contrast",
        "truth": "Fujita two-step-FDR significant variant-gene pair inside the same 1-Mb window",
        "universe": "intersection of AlphaGenome-covered pairs and Fujita-tested genes",
        "rows": rows, "paired_contrasts": comparisons,
    }
    (args.output_dir / f"{args.figure_stem}_{args.artifact_tag}.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
