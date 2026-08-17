#!/usr/bin/env python
"""Supplementary Figure 10: fitted-rank and empirical-frequency checks."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "supp" / "supp-figures" / "figure_top10_rank_validation.pdf"


def average_ranks(values):
    """Return average ranks, matching scipy.stats.rankdata(method='average')."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_rho(x, y):
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def equal_count_trend(fitted, empirical, n_bins=8):
    """Descriptive local means in equal-count bins ordered by fitted probability."""
    order = np.argsort(fitted)
    bins = np.array_split(order, n_bins)
    x = np.asarray([
        np.exp(np.mean(np.log(fitted[idx]))) for idx in bins if len(idx)
    ])
    y = np.asarray([
        np.mean(empirical[idx]) for idx in bins if len(idx)
    ])
    return x, y


def summarize_public(arm, state):
    src = json.load(open(
        HERE / f"top10_contrast_1kg_{arm}_lead{state}_v60.json"
    ))
    rows = src["per_locus"]
    by_rank = []
    rho = []
    pairs = []
    for rank in range(10):
        hs = [row["top10"][rank] for row in rows if len(row["top10"]) > rank]
        q = np.asarray([h["q1_probability"] for h in hs])
        e = np.asarray(
            [h["empirical_probability_given_lead_state"] for h in hs]
        )
        cumulative = np.asarray([
            sum(h["empirical_probability_given_lead_state"]
                for h in row["top10"][:rank + 1])
            for row in rows
        ])
        by_rank.append({
            "rank": rank + 1,
            "median_fitted_probability": float(np.median(q)),
            "fraction_observed_at_least_once": float(np.mean(e > 0)),
            "median_cumulative_empirical_probability": float(np.median(cumulative)),
            "q025_cumulative_empirical_probability": float(np.quantile(cumulative, 0.025)),
            "q975_cumulative_empirical_probability": float(np.quantile(cumulative, 0.975)),
        })
    for row in rows:
        q = [h["q1_probability"] for h in row["top10"]]
        e = [h["empirical_probability_given_lead_state"] for h in row["top10"]]
        pairs.extend({
            "fitted_probability": float(qi),
            "empirical_probability": float(ei),
        } for qi, ei in zip(q, e))
        if len(set(e)) > 1:
            rho.append(spearman_rho(q, e))
    return {
        "dataset": "1000 Genomes", "arm": arm, "lead_state": state,
        "n_loci": len(rows), "by_rank": by_rank, "spearman_rho": rho,
        "pairs": pairs,
    }


ros = json.load(open(
    HERE / "top10_contrast_summary_rosmap_v60.json"
))["datasets"]
groups = []
for state in (0, 1):
    groups.extend([
        summarize_public("armA", state),
        summarize_public("armB", state),
        next(x for x in ros if x["arm"] == "armA" and x["lead_state"] == state),
        next(x for x in ros if x["arm"] == "armB" and x["lead_state"] == state),
    ])

assert [g["n_loci"] for g in groups] == [27, 37, 27, 37] * 2
assert all(np.all(np.asarray(g["spearman_rho"]) > 0) for g in groups)

plt.style.use(HERE / "chap_figures.mplstyle")
fig, axes = plt.subplots(3, 2, figsize=(5.4, 5.7))
colors = {"1000 Genomes": "#0072B2", "ROS/MAP": "#E69F00"}
linestyles = {"armA": "-", "armB": "--"}
markers = {"armA": "o", "armB": "s"}
labels = {
    ("1000 Genomes", "armA"): "1000 Genomes, Arm A",
    ("1000 Genomes", "armB"): "1000 Genomes, Arm B",
    ("ROS/MAP", "armA"): "ROS/MAP, Arm A",
    ("ROS/MAP", "armB"): "ROS/MAP, Arm B",
}

for state in (0, 1):
    state_groups = [g for g in groups if g["lead_state"] == state]
    for g in state_groups:
        rank = np.asarray([x["rank"] for x in g["by_rank"]])
        fitted = np.asarray(
            [x["median_fitted_probability"] for x in g["by_rank"]]
        )
        observed = np.asarray(
            [x["fraction_observed_at_least_once"] for x in g["by_rank"]]
        )
        kw = dict(
            color=colors[g["dataset"]], linestyle=linestyles[g["arm"]],
            marker=markers[g["arm"]], label=labels[(g["dataset"], g["arm"])],
        )
        axes[0, state].plot(rank, fitted, **kw)
        axes[1, state].plot(rank, observed, **kw)

    # Candidate-level pairs can be shown for the public panel. ROS/MAP is
    # represented in the upper rows and in the manuscript's disclosure-safe
    # per-locus correlation summaries, but its candidate-level pairs remain
    # protected.
    for g in state_groups:
        if g["dataset"] != "1000 Genomes":
            continue
        fitted = np.asarray([x["fitted_probability"] for x in g["pairs"]])
        empirical = np.asarray([x["empirical_probability"] for x in g["pairs"]])
        axes[2, state].scatter(
            fitted, empirical, s=8, alpha=0.28,
            color=colors[g["dataset"]], marker=markers[g["arm"]],
        )
        trend_x, trend_y = equal_count_trend(fitted, empirical)
        axes[2, state].plot(
            trend_x, trend_y, color=colors[g["dataset"]],
            linestyle=linestyles[g["arm"]], marker=markers[g["arm"]],
            linewidth=1.25, markersize=3.5,
        )

for state in (0, 1):
    state_groups = [g for g in groups if g["lead_state"] == state]
    axes[0, state].set_title(
        rf"Lead state $X_0={state}$ ({'reference' if state == 0 else 'alternate'})"
    )
    axes[0, state].grid(axis="y", alpha=0.5)
    axes[0, state].set_yscale("log")
    axes[1, state].grid(axis="y", alpha=0.5)
    axes[1, state].set_ylim(-0.03, 1.03)
    axes[1, state].set_yticks([0, 0.5, 1])
    axes[1, state].set_xlabel("HaploPerturb probability rank")
    axes[1, state].set_xticks([1, 2, 4, 6, 8, 10])
    axes[2, state].plot(
        [1e-5, 1], [1e-5, 1], color="0.55", linestyle=":", linewidth=0.8,
        zorder=0,
    )
    axes[2, state].set_xscale("log")
    axes[2, state].set_yscale("symlog", linthresh=1e-3, linscale=0.7)
    axes[2, state].set_xlim(8e-6, 1.15)
    axes[2, state].set_ylim(0, 1.15)
    axes[2, state].set_xticks([1e-4, 1e-2, 1])
    axes[2, state].set_yticks([0, 1e-3, 1e-2, 1e-1, 1])
    axes[2, state].set_yticklabels(["0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1"])
    axes[2, state].set_xlabel("HaploPerturb conditional probability")
    axes[2, state].grid(alpha=0.5)
    rho_by_group = {
        (g["dataset"], g["arm"]): float(np.median(g["spearman_rho"]))
        for g in state_groups
    }
    axes[2, state].text(
        0.045, 0.84,
        "Median locus $\\rho$ (Arm A / B)\n"
        f"1000 Genomes  {rho_by_group[('1000 Genomes', 'armA')]:.2f} / "
        f"{rho_by_group[('1000 Genomes', 'armB')]:.2f}\n"
        f"ROS/MAP          {rho_by_group[('ROS/MAP', 'armA')]:.2f} / "
        f"{rho_by_group[('ROS/MAP', 'armB')]:.2f}",
        transform=axes[2, state].transAxes, va="top", fontsize=6.2,
    )

axes[0, 0].set_ylabel("Median HaploPerturb probability")
axes[1, 0].set_ylabel("Observed at least once")
axes[2, 0].set_ylabel("Empirical conditional frequency")
for ax, letter in zip(axes.ravel(), "ABCDEF"):
    ax.text(0.01, 0.97, letter, transform=ax.transAxes,
            va="top", fontweight="bold")
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
axes[0, 0].legend(handles, legend_labels, loc="upper right", fontsize=6)
fig.savefig(OUT)
print(OUT)
