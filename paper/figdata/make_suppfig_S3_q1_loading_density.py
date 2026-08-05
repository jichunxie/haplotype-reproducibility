#!/usr/bin/env python3
"""Supplementary Figure 3: density of public-panel q=1 loadings.

Sorted magnitudes diagnose gaps directly; cumulative squared-loading energy
diagnoses concentration. A sparse loading vector would drop sharply and place
most energy in a small fraction of variants. The figure uses only public 1000
Genomes coefficients summarized in q1_loading_density_v55.json.
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "q1_loading_density_v55.json"
OUT = (
    HERE.parent / "supp" / "supp-figures"
    / "S3_q1_loading_density.pdf"
)
LINEWIDTH_IN = 528.93675 / 72.27

artifact = json.load(SOURCE.open())
assert artifact["complete"] is True
assert artifact["cohort"] == "1000 Genomes unrelated EUR"
assert artifact["q"] == 1
rows = artifact["per_locus"]
groups = {group["arm"]: group for group in artifact["groups"]}
expected = {"armA": 27, "armB": 37}
for arm, count in expected.items():
    selected = [row for row in rows if row["arm"] == arm]
    assert len(selected) == count
    assert all(row["rank_fraction_grid"] == selected[0]["rank_fraction_grid"]
               for row in selected)

plt.style.use(str(HERE / "chap_figures.mplstyle"))
fig, axes = plt.subplots(
    2, 2,
    figsize=(LINEWIDTH_IN, 0.66 * LINEWIDTH_IN),
    sharex=True,
    constrained_layout=False,
)
fig.subplots_adjust(
    left=0.08, right=0.985, top=0.94, bottom=0.14,
    wspace=0.18, hspace=0.27,
)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "0.75"
arm_titles = {
    "armA": r"Arm A: $r^2\geq0.8$",
    "armB": r"Arm B: $r^2\geq0.5$",
}

for column, arm in enumerate(("armA", "armB")):
    selected = [row for row in rows if row["arm"] == arm]
    grid = np.asarray(selected[0]["rank_fraction_grid"])
    magnitude = np.asarray([
        row["abs_loading_by_rank_fraction"] for row in selected
    ])
    cumulative = np.asarray([
        row["cumulative_energy_by_rank_fraction"] for row in selected
    ])
    least_dense = min(
        selected,
        key=lambda row: row["effective_support_fraction"],
    )
    least_index = selected.index(least_dense)

    top = axes[0, column]
    for curve in magnitude:
        top.plot(grid, curve, color=GREY, lw=0.55, alpha=0.65)
    top.plot(
        grid,
        np.median(magnitude, axis=0),
        color=BLUE,
        lw=1.6,
        label="median across loci",
    )
    top.plot(
        grid,
        magnitude[least_index],
        color=ORANGE,
        lw=1.2,
        label="least-dense locus",
    )
    top.set_ylim(0.45, 1.01)
    top.set_ylabel(r"sorted loading magnitude $|b|$")
    top.set_title(
        f"{'A' if column == 0 else 'B'}   {arm_titles[arm]}",
        loc="left",
    )
    top.text(
        0.02,
        0.07,
        (
            rf"least dense: $s_{{\rm eff}}/p="
            f"{least_dense['effective_support_fraction']:.3f}$"
        ),
        transform=top.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
    )

    bottom = axes[1, column]
    for curve in cumulative:
        bottom.plot(grid, curve, color=GREY, lw=0.55, alpha=0.65)
    bottom.plot(
        grid,
        np.median(cumulative, axis=0),
        color=BLUE,
        lw=1.6,
    )
    bottom.plot(
        grid,
        cumulative[least_index],
        color=ORANGE,
        lw=1.2,
    )
    bottom.plot(
        grid,
        grid,
        color="0.25",
        lw=0.8,
        ls="--",
        label="equal-magnitude loading",
    )
    bottom.set_xlim(0.0, 1.0)
    bottom.set_ylim(0.0, 1.01)
    bottom.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    bottom.set_xticklabels(["0", "25", "50", "75", "100"])
    bottom.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    bottom.set_yticklabels(["0", "25", "50", "75", "100"])
    bottom.set_xlabel("variants retained by descending loading  (%)")
    bottom.set_ylabel(r"cumulative $\sum b_j^2$  (%)")
    bottom.set_title(
        f"{'C' if column == 0 else 'D'}   cumulative loading energy",
        loc="left",
    )
    fraction90 = groups[arm]["fraction_variants_for_90pct_energy"]
    bottom.text(
        0.98,
        0.05,
        (
            "90% energy needs\n"
            f"median {100 * fraction90['median']:.1f}% of variants"
        ),
        transform=bottom.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.7,
    )

handles, labels = axes[0, 0].get_legend_handles_labels()
diagonal_handle, diagonal_label = axes[1, 0].get_legend_handles_labels()
fig.legend(
    handles + diagonal_handle,
    labels + diagonal_label,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=3,
    frameon=False,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print("wrote", OUT)
for arm in ("armA", "armB"):
    effective = groups[arm]["effective_support_fraction"]
    fraction90 = groups[arm]["fraction_variants_for_90pct_energy"]
    selected = [row for row in rows if row["arm"] == arm]
    largest_gap = max(
        selected,
        key=lambda row: row["largest_adjacent_abs_loading_gap"],
    )
    print(
        arm,
        "effective support/p min/median/max",
        f"{effective['minimum']:.4f}/"
        f"{effective['median']:.4f}/"
        f"{effective['maximum']:.4f}",
        "90% energy fraction min/median/max",
        f"{fraction90['minimum']:.4f}/"
        f"{fraction90['median']:.4f}/"
        f"{fraction90['maximum']:.4f}",
        "largest gap",
        largest_gap["locus"],
        f"{largest_gap['largest_gap_below']:.3f}->"
        f"{largest_gap['largest_gap_above']:.3f}",
    )
