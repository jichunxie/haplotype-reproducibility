#!/usr/bin/env python3
"""Plot the phased lead--partner LD structure at the largest AD locus.

The source is the stored 1000 Genomes EUR lead-LD query table used to define
the fixed partner manifests.  The plot is deterministic and uses no random
numbers.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ld_example_chr17_source.tsv"
COUNT_SOURCE = HERE / "partners_by_threshold_v36.json"
OUT = HERE.parent / "main" / "figures" / "figure_ld_example_chr17.pdf"
LEAD_BP = 46_062_125


df = pd.read_csv(SOURCE, sep="\t")
if len(df) != 2_998 or df["ID_A"].nunique() != 1:
    raise RuntimeError("Unexpected chromosome 17 LD source table")
df["position"] = df["ID_B"].str.split(":").str[1].astype(int)
df["distance_kb"] = (df["position"] - LEAD_BP) / 1_000

arm_b = df["PHASED_R2"] >= 0.5
arm_a = df["PHASED_R2"] >= 0.8
if (int(arm_a.sum()), int(arm_b.sum())) != (1_285, 2_693):
    raise RuntimeError("Partner counts do not reproduce the fixed manifests")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
fig, (ax, ax_count) = plt.subplots(
    2,
    1,
    figsize=(6.5, 7.0),
    gridspec_kw={"height_ratios": [1.0, 2.0]},
    constrained_layout=True,
)

ax.scatter(
    df.loc[~arm_b, "distance_kb"],
    df.loc[~arm_b, "PHASED_R"],
    s=5,
    color="#b8bcc2",
    alpha=0.50,
    linewidths=0,
    label=r"Stored only: $0.1\leq r^2<0.5$",
)
ax.scatter(
    df.loc[arm_b & ~arm_a, "distance_kb"],
    df.loc[arm_b & ~arm_a, "PHASED_R"],
    s=6,
    color="#4c78a8",
    alpha=0.65,
    linewidths=0,
    label=r"Arm B only: $0.5\leq r^2<0.8$",
)
ax.scatter(
    df.loc[arm_a, "distance_kb"],
    df.loc[arm_a, "PHASED_R"],
    s=7,
    color="#d1495b",
    alpha=0.72,
    linewidths=0,
    label=r"Arm A: $r^2\geq0.8$",
)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
for value in (np.sqrt(0.5), np.sqrt(0.8), -np.sqrt(0.5), -np.sqrt(0.8)):
    ax.axhline(value, color="#737373", linewidth=0.45, linestyle=":", zorder=0)

ax.set_xlim(-530, 530)
ax.set_ylim(-1.04, 1.04)
ax.set_xlabel("Partner position relative to lead variant (kb)")
ax.set_ylabel("Signed phased correlation with lead, $r$")
ax.set_title("Lead 17:46062125:A:C and its 2,998 stored LD partners", loc="left", weight="bold")
ax.legend(loc="lower left", frameon=False, markerscale=1.7, ncol=3, fontsize=7)

with COUNT_SOURCE.open() as handle:
    count_rows = json.load(handle)["per_locus"]
count_rows = [row for row in count_rows if row["locus"] != "17_46062125_A_C"]
if len(count_rows) != 37:
    raise RuntimeError("Expected 37 non-outlier loci")
count_rows.sort(key=lambda row: (row["n_r2_ge_0.5"], row["locus"]))
y = np.arange(len(count_rows))
count_a = np.asarray([row["n_r2_ge_0.8"] for row in count_rows])
count_b = np.asarray([row["n_r2_ge_0.5"] for row in count_rows])
labels = [row["locus"].replace("_", ":") for row in count_rows]

ax_count.hlines(y, count_a, count_b, color="#c7c9cc", linewidth=0.8, zorder=1)
ax_count.scatter(
    count_b, y, s=18, marker="o", color="#0072B2",
    label=r"Arm B: $r^2\geq0.5$", zorder=2,
)
ax_count.scatter(
    count_a, y, s=18, marker="D", color="#D55E00",
    label=r"Arm A: $r^2\geq0.8$", zorder=3,
)
ax_count.set_yticks(y, labels, fontsize=5.6)
ax_count.set_ylim(-0.8, len(y) - 0.2)
ax_count.set_xlim(-4, 185)
ax_count.set_xlabel("Number of lead-linked partner variants")
ax_count.set_title(
    "Partner counts at the remaining 37 GWAS loci",
    loc="left",
    weight="bold",
)
ax_count.grid(axis="x", color="#e1e3e5", linewidth=0.5)
ax_count.legend(loc="lower right", frameon=False, fontsize=7)

ax.text(-0.11, 1.04, "A", transform=ax.transAxes, weight="bold", fontsize=10)
ax_count.text(-0.11, 1.01, "B", transform=ax_count.transAxes, weight="bold", fontsize=10)
fig.savefig(OUT)
print(OUT)
