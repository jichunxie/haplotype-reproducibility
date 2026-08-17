#!/usr/bin/env python3
"""Plot q=1 PVA and binary-correlation residual fit by cohort and LD arm.

Deterministic: reads q1_pva_comparison_v53.json, the two v54 residual summaries
and partners_by_threshold_v36.json. No jitter and no random number generation
anywhere, so re-running reproduces the figure exactly.

Both rows use the same locus ordering -- by mean PVA across the two cohorts,
smallest first, with partner count and then locus name breaking ties -- and the
same y limits. Thus the ROS/MAP and 1000 Genomes estimates for a locus share one
horizontal row, and that row is aligned between the PVA and residual panels. The
first entry is drawn at y = 0, so the locus with the lowest mean PVA anchors the
bottom of each panel. A grey line joins the two cohort estimates at every locus.
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "q1_pva_comparison_v53.json"
PARTNERS = HERE / "partners_by_threshold_v36.json"
RESIDUAL_SOURCES = {
    "ROS/MAP": HERE / "q1_pairwise_residuals_rosmap_v54.json",
    "1000 Genomes unrelated EUR": HERE / "q1_pairwise_residuals_1kg_v54.json",
}
OUT = HERE.parent / "supp" / "supp-figures" / "figure_pva_q1.pdf"
LINEWIDTH_IN = 390.0 / 72.0

artifact = json.load(SOURCE.open())
assert artifact["complete"] is True
assert artifact["q"] == 1
assert artifact["psi_min"] == 0.01

rows = artifact["per_locus"]
cohorts = ("ROS/MAP", "1000 Genomes unrelated EUR")
arms = (("armA", r"Arm A: $r^2\geq0.8$", 27, "n_r2_ge_0.8"),
        ("armB", r"Arm B: $r^2\geq0.5$", 37, "n_r2_ge_0.5"))
colors = {"ROS/MAP": "#0072B2", "1000 Genomes unrelated EUR": "#D55E00"}
markers = {"ROS/MAP": "o", "1000 Genomes unrelated EUR": "D"}
labels = {"ROS/MAP": "ROS/MAP", "1000 Genomes unrelated EUR": "1000 Genomes EUR"}

partner_artifact = json.load(PARTNERS.open())
partner_counts = {row["locus"]: row for row in partner_artifact["per_locus"]}

residual_rows = []
for cohort, source in RESIDUAL_SOURCES.items():
    residual_artifact = json.load(source.open())
    assert residual_artifact["complete"] is True
    assert residual_artifact["q"] == 1
    assert residual_artifact["cohort"] == cohort
    assert "Gauss--Hermite" in residual_artifact["quadrature"]["rule"]
    assert residual_artifact["quadrature"]["check_order"] == 1024
    assert residual_artifact["quadrature"]["final_order"] == 2048
    residual_rows.extend(residual_artifact["per_locus"])

plt.style.use(str(HERE / "chap_figures.mplstyle"))
fig, axes = plt.subplots(
    2, 2, figsize=(LINEWIDTH_IN, 5.45), sharex="row", sharey=False,
    constrained_layout=False,
    gridspec_kw={"wspace": 0.20, "hspace": 0.30,
                 "height_ratios": [1.0, 1.0]},
)

for column, (arm, title, expected, partner_key) in enumerate(arms):
    by_locus = {}
    fitted_variants = {}
    for row in rows:
        if row["arm"] == arm:
            by_locus.setdefault(row["locus"], {})[row["cohort"]] = (
                100.0 * row["pva_q1"]
            )
            if row["cohort"] == "1000 Genomes unrelated EUR":
                fitted_variants[row["locus"]] = row["n_fitted_variants"]
    assert len(by_locus) == expected
    assert all(set(values) == set(cohorts) for values in by_locus.values())

    # The arm is DEFINED by the partner threshold, so order on the shared manifest
    # count rather than on either cohort's fitted count. They agree exactly on the
    # public panel, and that agreement is asserted rather than assumed.
    n_partners = {locus: partner_counts[locus][partner_key] for locus in by_locus}
    for locus, count in n_partners.items():
        assert fitted_variants[locus] - 1 == count, (
            f"{arm} {locus}: manifest {count} partners but "
            f"{fitted_variants[locus] - 1} fitted"
        )

    residual_by_locus = {}
    for row in residual_rows:
        if row["arm"] == arm:
            residual_by_locus.setdefault(row["locus"], {})[row["cohort"]] = (
                row["correlation_residual_rmse"]
            )
    assert set(residual_by_locus) == set(by_locus)
    assert all(set(values) == set(cohorts) for values in residual_by_locus.values())

    # Primary key: mean PVA across the two cohorts, SMALLEST FIRST. The first
    # entry is drawn at y = 0, so the locus with the lowest two-cohort mean PVA
    # anchors the bottom of both panels and mean PVA increases upwards. Partner
    # count breaks ties, largest first, and the locus name breaks any remaining
    # tie so the order is total and the figure reproduces exactly.
    order = sorted(
        by_locus,
        key=lambda locus: (np.mean(list(by_locus[locus].values())),
                           -n_partners[locus],
                           locus),
    )
    y = np.arange(len(order))

    pva_ax = axes[0, column]
    for i, locus in enumerate(order):
        pva_ax.plot(
            [by_locus[locus][cohort] for cohort in cohorts],
            [i, i],
            color="0.68",
            lw=0.8,
            zorder=1,
        )
    for cohort in cohorts:
        pva_ax.plot(
            [by_locus[locus][cohort] for locus in order],
            y,
            linestyle="none",
            marker=markers[cohort],
            ms=2.8,
            color=colors[cohort],
            label=labels[cohort],
            zorder=3,
        )

    panel_letter = "A   " if column == 0 else ""
    pva_ax.set_title(f"{panel_letter}{title}", loc="left")
    pva_ax.set_xlabel(r"$\widehat{\mathrm{PVA}}(1)$  (%)")
    if column == 0:
        pva_ax.set_ylabel("loci, lowest mean PVA at bottom")
    pva_ax.set_ylim(-1.0, len(order) - 0.3)
    pva_ax.axvline(90.0, color="0.35", lw=0.8, ls="--", zorder=0)
    pva_ax.set_xlim(79.0, 100.5)
    pva_ax.set_xticks([80, 85, 90, 95, 99])
    # The ordering is by PVA, which panel A's x-axis already shows, so numeric y
    # ticks would only repeat it. Partner counts would be worse than nothing here:
    # they are no longer monotone down the axis.
    pva_ax.set_yticks([])

    worst_pva_locus = min(
        order, key=lambda locus: np.mean(list(by_locus[locus].values()))
    )
    worst_pva_index = order.index(worst_pva_locus)
    worst_pva_x = min(by_locus[worst_pva_locus].values())
    # Lift the label off the marker's own row: level with the point it would sit on
    # the grey line joining the two cohorts. Above, except in the top rows where it
    # would then run into the panel title.
    dy, va = ((-7.5, "top") if worst_pva_index >= len(order) - 2
              else (7.0, "bottom"))
    # This marker is the leftmost point of its row, so the space to its left is
    # clear while the space to its right holds the joining line and the other
    # cohort's point. Label leftwards unless the marker is against the axis limit.
    left_room = (worst_pva_x - 79.0) / (100.5 - 79.0) > 0.20
    pva_ax.annotate(
        worst_pva_locus.replace("_", ":"),
        xy=(worst_pva_x, worst_pva_index),
        xytext=(-3.5 if left_room else 3.5, dy),
        textcoords="offset points",
        fontsize=6.2,
        ha="right" if left_room else "left",
        va=va,
    )

    # Row B: same linked style as row A -- one point per cohort-locus fit with a
    # grey line joining the two cohorts at a locus -- on the row-A ordering and
    # limits, so a locus sits at the same height in both rows.
    residual_ax = axes[1, column]
    for i, locus in enumerate(order):
        residual_ax.plot(
            [residual_by_locus[locus][cohort] for cohort in cohorts],
            [i, i],
            color="0.68",
            lw=0.8,
            zorder=1,
        )
    for cohort in cohorts:
        residual_ax.plot(
            [residual_by_locus[locus][cohort] for locus in order],
            y,
            linestyle="none",
            marker=markers[cohort],
            ms=2.8,
            color=colors[cohort],
            zorder=3,
        )

    panel_letter = "B   " if column == 0 else ""
    residual_ax.set_title(f"{panel_letter}{title}", loc="left")
    residual_ax.set_xlabel("correlation residual RMSE")
    residual_ax.set_xlim(0.0, 0.365)
    residual_ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    if column == 0:
        residual_ax.set_ylabel("loci, lowest mean PVA at bottom")
    # Same limits as row A, so the two rows are read locus by locus.
    residual_ax.set_ylim(-1.0, len(order) - 0.3)
    residual_ax.set_yticks([])

    worst_locus = max(
        order, key=lambda locus: max(residual_by_locus[locus].values())
    )
    worst_cohort = max(
        cohorts, key=lambda cohort: residual_by_locus[worst_locus][cohort]
    )
    worst_x = residual_by_locus[worst_locus][worst_cohort]
    worst_index = order.index(worst_locus)
    # Same rule as row A, plus: turn the label back inside the axes when the marker
    # sits close to the right limit, or the text is clipped.
    dy, va = ((-7.5, "top") if worst_index >= len(order) - 2
              else (7.0, "bottom"))
    near_edge = worst_x > 0.82 * 0.365
    residual_ax.annotate(
        worst_locus.replace("_", ":"),
        xy=(worst_x, worst_index),
        xytext=(-1.0 if near_edge else 3.5, dy),
        textcoords="offset points",
        fontsize=6.2,
        ha="right" if near_edge else "left",
        va=va,
    )

    # The plotted values must be the artifact's values.
    for source in RESIDUAL_SOURCES.values():
        residual_artifact = json.load(source.open())
        for group in residual_artifact["groups"]:
            if group["arm"] != arm:
                continue
            plotted = np.array([
                residual_by_locus[locus][group["cohort"]] for locus in order
            ])
            assert abs(np.median(plotted) - group["median"]) < 5e-12
            assert abs(plotted.min() - group["minimum"]) < 5e-12
            assert abs(plotted.max() - group["maximum"]) < 5e-12

handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    legend_labels,
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(0.085, 0.004),
    ncol=2,
    columnspacing=1.2,
)
fig.text(
    0.995,
    0.018,
    "dashed: 90\\% PVA target; residual RMSE: smaller is better",
    ha="right",
    va="bottom",
    fontsize=6.2,
)

fig.subplots_adjust(
    left=0.145, right=0.99, top=0.955, bottom=0.105,
    wspace=0.20, hspace=0.58,
)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print("wrote", OUT)
for group in artifact["groups"]:
    print(
        group["cohort"],
        group["arm"],
        "n=", group["n_loci"],
        "min/median/max=",
        f"{100 * group['minimum']:.2f}/"
        f"{100 * group['median']:.2f}/"
        f"{100 * group['maximum']:.2f}%",
        "n>=90%=", group["n_ge_0.90"],
    )
for cohort, source in RESIDUAL_SOURCES.items():
    residual_artifact = json.load(source.open())
    for group in residual_artifact["groups"]:
        values = np.array([
            row["correlation_residual_rmse"]
            for row in residual_artifact["per_locus"]
            if row["arm"] == group["arm"]
        ])
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        print(
            cohort,
            group["arm"],
            "correlation-residual RMSE min/q25/median/q75/max=",
            f"{group['minimum']:.3f}/{q25:.3f}/{q50:.3f}/"
            f"{q75:.3f}/{group['maximum']:.3f}",
        )
