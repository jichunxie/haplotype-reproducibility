#!/usr/bin/env python3
"""Supplementary Figure 1: observed, HaploPerturb-implied, and residual q=1 correlations.

The displayed locus is selected mechanically from the public 1000 Genomes
analysis as the locus with the largest partner--partner residual RMSE among
fits containing at least 20 variants. The matrices were binned by genomic
order upstream, so this plotting script reads only a compact public artifact.
No ROS/MAP individual-level or matrix-valued data are used or exported.

Deterministic: no jitter and no random seed.
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "q1_residual_heatmap_1kg_v54.npz"
OUT = (
    HERE.parent / "supp" / "supp-figures"
    / "S2_q1_residual_heatmap.pdf"
)
LINEWIDTH_IN = 528.93675 / 72.27

with np.load(SOURCE, allow_pickle=False) as artifact:
    cohort = str(artifact["cohort"])
    arm = str(artifact["arm"])
    locus = str(artifact["locus"])
    n_variants = int(artifact["n_fitted_variants"])
    selection_rule = str(artifact["selection_rule"])
    bin_start = artifact["bin_start"].astype(np.int64)
    bin_end = artifact["bin_end"].astype(np.int64)
    bin_size = artifact["bin_size"].astype(np.int64)
    lead_bin = int(artifact["lead_bin"])
    observed = artifact["observed"].astype(np.float64)
    fitted = artifact["fitted"].astype(np.float64)
    residual = artifact["residual"].astype(np.float64)

assert cohort == "1000 Genomes unrelated EUR"
assert arm == "armB"
assert locus == "17_46062125_A_C"
assert n_variants == 2694
assert selection_rule.startswith(
    "largest partner-partner correlation-residual RMSE"
)
assert observed.shape == fitted.shape == residual.shape == (120, 120)
assert np.allclose(observed, observed.T)
assert np.allclose(fitted, fitted.T)
assert np.allclose(residual, residual.T)
assert np.allclose(observed - fitted, residual)
assert bin_size.min() == 22 and bin_size.max() == 23

plt.style.use(str(HERE / "chap_figures.mplstyle"))
fig = plt.figure(figsize=(LINEWIDTH_IN, 2.72), constrained_layout=False)
grid = fig.add_gridspec(
    2, 3,
    height_ratios=(1.0, 0.075),
    left=0.065, right=0.985, top=0.865, bottom=0.17,
    wspace=0.11, hspace=0.20,
)
axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
cax_common = fig.add_subplot(grid[1, 0:2])
cax_residual = fig.add_subplot(grid[1, 2])

common_norm = Normalize(vmin=0.0, vmax=1.0)
residual_limit = float(np.max(np.abs(residual)))
residual_norm = TwoSlopeNorm(
    vmin=-residual_limit,
    vcenter=0.0,
    vmax=residual_limit,
)
images = (
    axes[0].imshow(observed, cmap="viridis", norm=common_norm),
    axes[1].imshow(fitted, cmap="viridis", norm=common_norm),
    axes[2].imshow(residual, cmap="RdBu_r", norm=residual_norm),
)
titles = (
    r"A   empirical $\widehat{\mathbf{R}}_{X}$",
    r"B   HaploPerturb $\widehat{\mathbf{R}}_{X,1}$",
    r"C   residual $\widehat{\mathbf{R}}_{X}-\widehat{\mathbf{R}}_{X,1}$",
)

tick_bins = np.asarray([0, len(bin_start) // 2, len(bin_start) - 1])
tick_positions = 0.5 * (bin_start[tick_bins] + bin_end[tick_bins]) / 1e6
tick_labels = [f"{value:.2f}" for value in tick_positions]
for index, (ax, title) in enumerate(zip(axes, titles)):
    ax.set_title(title, loc="left")
    ax.set_xticks(tick_bins)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_bins)
    if index == 0:
        ax.set_yticklabels(tick_labels)
        ax.set_ylabel("position (Mb)")
    else:
        ax.set_yticklabels([])
    for coordinate in (lead_bin - 0.5, lead_bin + 0.5):
        ax.axvline(coordinate, color="black", lw=1.1, zorder=3)
        ax.axhline(coordinate, color="black", lw=1.1, zorder=3)
        ax.axvline(coordinate, color="white", lw=0.45, zorder=4)
        ax.axhline(coordinate, color="white", lw=0.45, zorder=4)

axes[0].annotate(
    "lead bin",
    xy=(lead_bin, -0.5),
    xytext=(0, 5),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=6.5,
)

common_bar = fig.colorbar(
    images[0],
    cax=cax_common,
    orientation="horizontal",
    ticks=[0.0, 0.5, 1.0],
)
common_bar.set_label("binary allelic correlation", fontsize=7)
residual_bar = fig.colorbar(
    images[2],
    cax=cax_residual,
    orientation="horizontal",
    ticks=[-residual_limit, 0.0, residual_limit],
    format="%.2f",
)
residual_bar.set_label("correlation residual", fontsize=7)
for colorbar in (common_bar, residual_bar):
    colorbar.ax.tick_params(labelsize=6.5, length=2)

locus_label = locus.replace("_", ":")
fig.suptitle(
    f"1000 Genomes EUR, Arm B; {locus_label}; "
    f"{n_variants:,} variants summarized in 120 genomic-order bins",
    x=0.065,
    y=0.975,
    ha="left",
    fontsize=8,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print("wrote", OUT)
print("selection:", selection_rule)
print(
    "observed min/max", f"{observed.min():.4f}/{observed.max():.4f}",
    "fitted min/max", f"{fitted.min():.4f}/{fitted.max():.4f}",
    "residual min/max", f"{residual.min():.4f}/{residual.max():.4f}",
)
