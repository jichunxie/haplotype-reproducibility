#!/usr/bin/env python3
"""Cell-type-specific empirical-FDR tests for protected ROS/MAP v61 predictions.

The Fujita truth files enter only after scores, nulls, empirical p-values and BH
q-values are fixed. Full scored hypotheses remain on RCC; exported JSON and PDF
contain aggregate validation summaries only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

CELL_MAP = {
    "Ast": "astrocyte",
    "Exc": "glutamatergic neuron",
    "Mic": "CD14-positive monocyte",
}
CELL_LABEL = {"Ast": "Astrocyte", "Exc": "Excitatory", "Mic": "Microglia proxy"}
SEED = 620731
N_NULL = 250
EPS = 1e-8


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    q = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def empirical_p(scores: np.ndarray, null: np.ndarray) -> np.ndarray:
    null = np.sort(np.asarray(null, float))
    idx = np.searchsorted(null, np.asarray(scores, float), side="left")
    return (1.0 + len(null) - idx) / (1.0 + len(null))


def stable_rng(*tokens: str) -> np.random.Generator:
    text = "|".join(tokens).encode()
    offset = int.from_bytes(hashlib.sha256(text).digest()[:8], "little")
    return np.random.default_rng((SEED + offset) % (2**63 - 1))


def collapse_predictions(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    reverse = {v: k for k, v in CELL_MAP.items()}
    d = d.loc[d.biosample_name.isin(reverse)].copy()
    d["celltype"] = d.biosample_name.map(reverse)
    keys = ["arm", "locus", "celltype", "gene_name", "rank"]
    # Equal-weight technical RNA tracks and annotated terminal transcripts within a gene.
    d = d.groupby(keys, as_index=False).agg(
        pred0=("pred_mean_lead0", "mean"),
        pred1=("pred_mean_lead1", "mean"),
    )
    if d[["pred0", "pred1"]].isna().any().any():
        raise ValueError("missing collapsed predictions")
    return d


def make_group_scores(group: pd.DataFrame, method: str) -> list[dict]:
    arm = str(group.arm.iloc[0])
    locus = str(group.locus.iloc[0])
    cell = str(group.celltype.iloc[0])
    p0 = group.pivot(index="gene_name", columns="rank", values="pred0").sort_index()
    p1 = group.pivot(index="gene_name", columns="rank", values="pred1").sort_index()
    genes = p0.index.intersection(p1.index)
    ranks = sorted(set(p0.columns).intersection(p1.columns))
    p0 = p0.loc[genes, ranks].to_numpy(float)
    p1 = p1.loc[genes, ranks].to_numpy(float)
    r = len(ranks)
    if r < 2:
        return []
    rng = stable_rng(arm, locus, cell, method)

    if method == "top1":
        target = np.log1p(p1[:, 0]) - np.log1p(p0[:, 0])
        null = np.empty((N_NULL, len(genes)), float)
        for b in range(N_NULL):
            arr = p0 if b % 2 == 0 else p1
            i, j = rng.choice(r, size=2, replace=False)
            null[b] = np.log1p(arr[:, i]) - np.log1p(arr[:, j])
    elif method == "top5":
        target = np.log1p(p1.mean(axis=1)) - np.log1p(p0.mean(axis=1))
        null = np.empty((N_NULL, len(genes)), float)
        for b in range(N_NULL):
            arr = p0 if b % 2 == 0 else p1
            ia = rng.integers(0, r, size=r)
            ib = rng.integers(0, r, size=r)
            null[b] = np.log1p(arr[:, ia].mean(axis=1)) - np.log1p(arr[:, ib].mean(axis=1))
    else:
        raise ValueError(method)

    abs_target = np.abs(target)
    abs_null = np.abs(null)
    global_scale = float(np.median(abs_null))
    gene_scale = np.median(abs_null, axis=0)
    gene_scale = np.sqrt(0.5 * gene_scale**2 + 0.5 * global_scale**2)
    gene_scale = np.maximum(gene_scale, EPS)
    gene_target = abs_target / gene_scale
    gene_null = abs_null / gene_scale[None, :]

    target_locus_scale = max(float(np.median(abs_target)), 0.25 * global_scale, EPS)
    locus_target = abs_target / target_locus_scale
    locus_null_scale = np.maximum(np.median(abs_null, axis=1), EPS)
    locus_null = abs_null / locus_null_scale[:, None]

    rows = []
    for j, gene in enumerate(genes):
        rows.append({
            "arm": arm, "locus": locus, "celltype": cell, "gene_name": str(gene),
            "method": method, "effect": float(target[j]),
            "locus_score": float(locus_target[j]), "locus_null": locus_null[:, j],
            "gene_score": float(gene_target[j]), "gene_null": gene_null[:, j],
        })
    return rows


def score_all(collapsed: pd.DataFrame) -> list[dict]:
    records = []
    for _, group in collapsed.groupby(["arm", "locus", "celltype"], sort=True):
        records.extend(make_group_scores(group, "top1"))
        records.extend(make_group_scores(group, "top5"))
    return records


def pair_tests(records: list[dict], focus: str) -> pd.DataFrame:
    score_key, null_key = f"{focus}_score", f"{focus}_null"
    rows = []
    strata = {}
    for rec in records:
        key = (rec["arm"], rec["celltype"], rec["method"])
        strata.setdefault(key, []).append(rec[null_key])
    pools = {key: np.concatenate(value) for key, value in strata.items()}
    for rec in records:
        key = (rec["arm"], rec["celltype"], rec["method"])
        rows.append({
            "arm": rec["arm"], "locus": rec["locus"], "celltype": rec["celltype"],
            "gene_name": rec["gene_name"], "method": rec["method"], "focus": focus,
            "effect": rec["effect"], "score": rec[score_key],
            "p_value": float(empirical_p(np.array([rec[score_key]]), pools[key])[0]),
        })
    out = pd.DataFrame(rows)
    out["q_value"] = np.nan
    for _, idx in out.groupby(["arm", "celltype", "method"]).groups.items():
        out.loc[idx, "q_value"] = bh(out.loc[idx, "p_value"].to_numpy())
    return out


def gene_tests(records: list[dict]) -> pd.DataFrame:
    grouped = {}
    for rec in records:
        key = (rec["arm"], rec["celltype"], rec["method"], rec["gene_name"])
        grouped.setdefault(key, []).append(rec)
    gene_records = []
    for (arm, cell, method, gene), recs in grouped.items():
        score = max(r["gene_score"] for r in recs)
        effect = recs[int(np.argmax([r["gene_score"] for r in recs]))]["effect"]
        null = np.maximum.reduce([r["gene_null"] for r in recs])
        gene_records.append({
            "arm": arm, "celltype": cell, "method": method, "gene_name": gene,
            "focus": "gene", "effect": effect, "score": score, "null": null,
            "n_loci_searched": len(recs),
        })
    pools = {}
    for rec in gene_records:
        key = (rec["arm"], rec["celltype"], rec["method"])
        pools.setdefault(key, []).append(rec["null"])
    pools = {key: np.concatenate(value) for key, value in pools.items()}
    rows = []
    for rec in gene_records:
        key = (rec["arm"], rec["celltype"], rec["method"])
        row = {k: v for k, v in rec.items() if k != "null"}
        row["p_value"] = float(empirical_p(np.array([rec["score"]]), pools[key])[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value"] = np.nan
    for _, idx in out.groupby(["arm", "celltype", "method"]).groups.items():
        out.loc[idx, "q_value"] = bh(out.loc[idx, "p_value"].to_numpy())
    return out


def truth_tables(egene_path: Path, variant_path: Path):
    truth = pd.read_csv(egene_path)
    truth["is_egene"] = truth.is_egene.astype(str).str.lower().eq("true")
    truth = truth.rename(columns={"gene_symbol": "gene_name"})
    variants = pd.read_csv(variant_path)
    variants = variants.rename(columns={"gene_symbol": "gene_name"})
    split = variants.variant_id.str.split(":", expand=True)
    variants["chrom"] = split[0].astype(str)
    variants["pos"] = split[1].astype(int)
    return truth, variants


def add_pair_truth(tests: pd.DataFrame, truth: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    out = tests.merge(truth[["gene_name", "celltype", "is_egene"]],
                      on=["gene_name", "celltype"], how="inner")
    sig = {
        (cell, gene): list(zip(sub.chrom.astype(str), sub.pos.astype(int)))
        for (cell, gene), sub in variants.groupby(["celltype", "gene_name"])
    }
    def positive(row):
        chrom, pos, _, _ = row.locus.split("_", 3)
        pos = int(pos)
        start, end = max(1, pos - 524288), max(1, pos - 524288) + 1048576
        return any(c == chrom and start <= p < end for c, p in sig.get((row.celltype, row.gene_name), []))
    out["truth"] = out.apply(positive, axis=1)
    return out


def add_gene_truth(tests: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    out = tests.merge(truth[["gene_name", "celltype", "is_egene"]],
                      on=["gene_name", "celltype"], how="inner")
    out["truth"] = out.is_egene
    return out


def pr_points(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.sort_values("score", ascending=False).reset_index(drop=True)
    y = d.truth.to_numpy(bool)
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    positives = max(int(y.sum()), 1)
    return pd.DataFrame({
        "threshold_rank": np.arange(1, len(d) + 1),
        "precision": tp / np.maximum(tp + fp, 1),
        "recall": tp / positives,
    })


def metrics(frame: pd.DataFrame) -> dict:
    called = frame.q_value < 0.05
    y = frame.truth.astype(bool)
    tp = int((called & y).sum())
    fp = int((called & ~y).sum())
    positives = int(y.sum())
    n_calls = int(called.sum())
    negatives = int((~y).sum())
    return {
        "n_tested_against_truth": int(len(frame)), "n_truth_positive": positives,
        "n_truth_negative": negatives,
        "n_fdr_calls": n_calls, "true_positives": tp, "false_positives": fp,
        "precision_at_q05": None if n_calls == 0 else float(tp / n_calls),
        "validation_fdp_at_q05": None if n_calls == 0 else float(fp / n_calls),
        "recall_at_q05": float(tp / max(positives, 1)),
        "known_negative_call_rate_at_q05": float(fp / max(negatives, 1)),
    }


def matched_count_metrics(frame: pd.DataFrame) -> dict:
    """Call as many scored hypotheses as Fujita calls positive in this universe.

    This is a matched-stringency validation summary, not another FDR procedure:
    Fujita's published two-step-FDR flag defines truth, while the AlphaGenome
    hypotheses are ranked without using that flag.
    """
    positives = int(frame.truth.astype(bool).sum())
    order_cols = ["score"] + [c for c in ["gene_name", "locus"] if c in frame]
    ascending = [False] + [True] * (len(order_cols) - 1)
    ranked = frame.sort_values(order_cols, ascending=ascending, kind="mergesort")
    called = ranked.head(positives)
    tp = int(called.truth.astype(bool).sum())
    fp = int(len(called) - tp)
    negatives = int(len(frame) - positives)
    enrichment = None
    p_enrichment = None
    if positives and positives < len(frame):
        baseline = positives / len(frame)
        enrichment = float((tp / positives) / baseline)
        p_enrichment = float(hypergeom.sf(tp - 1, len(frame), positives, positives))
    return {
        "matched_n_calls": positives,
        "matched_true_positives": tp,
        "matched_false_positives": fp,
        "matched_precision": None if positives == 0 else float(tp / positives),
        "matched_recall": None if positives == 0 else float(tp / positives),
        "matched_validation_fdp": None if positives == 0 else float(fp / positives),
        "matched_known_negative_call_rate": float(fp / max(negatives, 1)),
        "matched_enrichment_over_prevalence": enrichment,
        "matched_enrichment_p": p_enrichment,
    }


def plot_pr(curves: pd.DataFrame, output: Path) -> None:
    plt.style.use(Path(__file__).with_name("chap_figures.mplstyle"))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
    colors = {"armA": "#0072B2", "armB": "#E69F00"}
    styles = {"top5": "-", "top1": "--"}
    for col, cell in enumerate(["Ast", "Exc", "Mic"]):
        for row, focus in enumerate(["locus", "gene"]):
            ax = axes[row, col]
            sub = curves[(curves.celltype == cell) & (curves.focus == focus)]
            for (arm, method), line in sub.groupby(["arm", "method"], sort=True):
                ax.step(line.recall, line.precision, where="post", color=colors[arm],
                        linestyle=styles[method],
                        label=f"{arm.replace('arm', 'Arm ')} {method.replace('top', 'top-')}")
            ax.set_title(CELL_LABEL[cell])
            ax.set_ylim(0.795, 1.005)
            ax.set_yticks([0.8, 0.9, 1.0])
            ax.set_xlim(-0.002, 0.152)
            ax.set_xticks([0.00, 0.05, 0.10, 0.15])
            ax.grid(axis="both", alpha=0.4)
            if row == 1:
                ax.set_xlabel("Recall")
            if col == 0:
                ax.set_ylabel(("Locus--gene" if row == 0 else "Gene") + " precision")
    for ax, letter in zip(axes.ravel(), "ABCDEF"):
        ax.text(0.02, 0.96, letter, transform=ax.transAxes, va="top", fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="lower left", fontsize=6)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--egene-truth", type=Path, required=True)
    ap.add_argument("--sig-variants", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--style", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Put the shared style beside this script only for plot_pr's stable lookup.
    if args.style.resolve() != Path(__file__).with_name("chap_figures.mplstyle").resolve():
        Path(__file__).with_name("chap_figures.mplstyle").write_text(args.style.read_text())

    collapsed = collapse_predictions(args.predictions)
    records = score_all(collapsed)
    locus_tests = pair_tests(records, "locus")
    gene = gene_tests(records)
    truth, variants = truth_tables(args.egene_truth, args.sig_variants)
    locus_eval = add_pair_truth(locus_tests, truth, variants)
    gene_eval = add_gene_truth(gene, truth)
    locus_tests.to_parquet(args.output_dir / "protected_locus_gene_tests_v62.parquet", index=False)
    gene.to_parquet(args.output_dir / "protected_gene_tests_v62.parquet", index=False)

    all_eval = pd.concat([locus_eval, gene_eval], ignore_index=True, sort=False)
    metric_rows, curve_rows = [], []
    for keys, sub in all_eval.groupby(["focus", "arm", "celltype", "method"], sort=True):
        focus, arm, cell, method = keys
        rec = {"focus": focus, "arm": arm, "celltype": cell, "method": method}
        rec.update(metrics(sub))
        rec.update(matched_count_metrics(sub))
        metric_rows.append(rec)
        pr = pr_points(sub)
        pr["focus"], pr["arm"], pr["celltype"], pr["method"] = focus, arm, cell, method
        curve_rows.append(pr)
    matched_p = np.array([r["matched_enrichment_p"] for r in metric_rows], float)
    matched_q = bh(matched_p)
    for rec, q_value in zip(metric_rows, matched_q):
        rec["matched_enrichment_q_across_24"] = float(q_value)
    curves = pd.concat(curve_rows, ignore_index=True)
    plot_pr(curves, args.output_dir / "figure_eqtl_pr_high_precision_v62.pdf")

    summary = {
        "script": Path(__file__).name, "seed": SEED, "n_internal_null_replicates": N_NULL,
        "primary_contrast": "top5", "sensitivity_contrast": "top1",
        "fdr_families": "BH separately within arm, cell type, contrast, and focus",
        "truth_definition": (
            "Fujita published significant_by_2step_FDR calls; gene focus uses the resulting "
            "cell-type-specific eGene flag and locus focus requires a significant variant-gene "
            "pair inside the prespecified locus window"
        ),
        "matched_comparison": (
            "Within each arm, cell type, contrast, and focus, call the top K AlphaGenome scores, "
            "where K is the number of Fujita FDR-positive hypotheses in the evaluated universe; "
            "hypergeometric enrichment p-values are BH-adjusted across all 24 comparisons"
        ),
        "cell_type_mapping": CELL_MAP,
        "privacy": "aggregate validation metrics and PR points only; scored hypotheses remain on RCC",
        "metrics": metric_rows,
    }
    (args.output_dir / "eqtl_fdr_recall_v62.json").write_text(json.dumps(summary, indent=2))
    curves.to_csv(args.output_dir / "eqtl_pr_points_v62.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
