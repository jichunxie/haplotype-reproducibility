#!/usr/bin/env python3
"""Create the manuscript figures for the audited v77 ranking simulation."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


LEAD_CLASSES = ("rare", "common")
SCENARIOS = ("lead_ld_q1", "balanced_q1", "lead_ld_q2")
SAMPLE_SIZES = (500, 1000, 2000)
PARTNER_COUNTS = (4, 32, 256)
STRATEGIES = ("fitted", "empirical_mode", "ld_sign")

SCENARIO_LABEL = {
    "lead_ld_q1": "High latent dependence\ntrue $q=1$",
    "balanced_q1": "Moderate latent dependence\ntrue $q=1$",
    "lead_ld_q2": "High latent dependence\ntrue $q=2$; fit $q=1$",
}
STRATEGY_LABEL = {
    "fitted": "HaploPerturb mode",
    "empirical_mode": "Empirical mode",
    "ld_sign": "LD-sign strategy",
}
STRATEGY_COLOR = {
    "fitted": "#0072B2",
    "empirical_mode": "#009E73",
    "ld_sign": "#D55E00",
}
STRATEGY_MARKER = {"fitted": "o", "empirical_mode": "s", "ld_sign": "^"}
PARTNER_COLOR = {4: "#0072B2", 32: "#E69F00", 256: "#009E73"}
LEAD_LINESTYLE = {"rare": "-", "common": "--"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.lower() == "true"


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 7.1,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.7,
            edgecolor=edgecolor,
            facecolor=facecolor,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        transform=ax.transAxes,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color="#555555",
            connectionstyle=connectionstyle,
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def plot_design(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    ax.set_axis_off()

    add_box(
        ax,
        (0.015, 0.50),
        0.20,
        0.36,
        "Dependence structure\n\nHigh latent, $q=1$\nModerate latent, $q=1$\nHigh latent, $q=2$",
        "#EAF2F8",
        "#0072B2",
    )
    add_box(
        ax,
        (0.015, 0.09),
        0.20,
        0.31,
        "Allele frequencies\n\nLead: rare or common\nPartners: 50% rare, 50% common\n\nRare: log-uniform 0.5%--5%\nCommon: uniform 5%--40%",
        "#FFF3E6",
        "#D55E00",
        fontsize=6.7,
    )

    add_box(
        ax,
        (0.285, 0.62),
        0.18,
        0.20,
        "Known population\nconditional law",
        "#F3EAF7",
        "#CC79A7",
    )
    add_box(
        ax,
        (0.535, 0.62),
        0.18,
        0.20,
        "Certified population\nmode and top ten",
        "#F3EAF7",
        "#CC79A7",
    )
    add_arrow(ax, (0.215, 0.68), (0.285, 0.72))
    add_arrow(ax, (0.465, 0.72), (0.535, 0.72))

    add_box(
        ax,
        (0.285, 0.23),
        0.18,
        0.20,
        "One 2,000-haplotype\nmaster panel",
        "#E9F6F0",
        "#009E73",
    )
    add_box(
        ax,
        (0.535, 0.18),
        0.18,
        0.30,
        "Nested panels\n\n500 $\\subset$ 1,000\n$\\subset$ 2,000\n\nSame population truth",
        "#E9F6F0",
        "#009E73",
    )
    add_arrow(ax, (0.215, 0.28), (0.285, 0.33))
    add_arrow(ax, (0.465, 0.33), (0.535, 0.33))

    add_box(
        ax,
        (0.775, 0.56),
        0.205,
        0.24,
        "Construction strategies\n\nHaploPerturb $q=1$ mode\nEmpirical mode\nLD-sign background",
        "#FFF3E6",
        "#D55E00",
    )
    add_box(
        ax,
        (0.775, 0.18),
        0.205,
        0.24,
        "Compare with truth\n\nAvailability\nExact-mode error\nHamming distance",
        "#F2F2F2",
        "#555555",
    )
    add_arrow(ax, (0.715, 0.33), (0.775, 0.65), "arc3,rad=-0.12")
    add_arrow(ax, (0.875, 0.56), (0.875, 0.42))
    add_arrow(ax, (0.715, 0.72), (0.775, 0.30), "arc3,rad=0.18")

    ax.text(
        0.375,
        0.045,
        "$k=2^2,2^5,2^8$ partners; 30 populations per cell; both lead states evaluated",
        ha="center",
        va="center",
        fontsize=7.0,
        transform=ax.transAxes,
    )
    fig.savefig(path)
    plt.close(fig)


def plot_availability(availability_path: Path, path: Path) -> None:
    availability_rows = read_csv(availability_path)
    availability_lookup = {
        (row["lead_frequency_class"], int(row["n_haplotypes"])): row
        for row in availability_rows
    }
    fig, ax = plt.subplots(figsize=(7.2, 2.25), constrained_layout=False)
    x_availability = np.arange(len(SAMPLE_SIZES), dtype=float)
    availability_style = {
        "rare": ("#0072B2", "o", -0.04),
        "common": ("#D55E00", "s", 0.04),
    }
    for lead_class in LEAD_CLASSES:
        color, marker, offset = availability_style[lead_class]
        values = np.array(
            [
                100
                * float(
                    availability_lookup[(lead_class, n)][
                        "zero_alt_carrier_fraction"
                    ]
                )
                for n in SAMPLE_SIZES
            ]
        )
        ax.plot(
            x_availability + offset,
            values,
            color=color,
            marker=marker,
            label=f"{lead_class.capitalize()} lead",
        )
        for x, value, n in zip(x_availability + offset, values, SAMPLE_SIZES):
            numerator = round(value * 270 / 100)
            ax.annotate(
                f"{numerator}/270",
                (x, value),
                xytext=(
                    -3 if lead_class == "rare" else 3,
                    5 if value > 0 else 6,
                ),
                textcoords="offset points",
                ha="right" if lead_class == "rare" else "left",
                va="bottom",
                fontsize=6.7,
                color="#333333",
            )
    ax.set_ylabel("Empirical mode unavailable\n(no ALT carriers), %")
    ax.set_xlabel("Phased haplotypes")
    ax.set_xticks(x_availability, ("500", "1,000", "2,000"))
    ax.set_ylim(-0.05, 0.52)
    ax.set_yticks((0.0, 0.2, 0.4))
    ax.grid(axis="y", alpha=0.75)
    ax.legend(loc="upper right", ncol=2)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.24, top=0.90)
    fig.savefig(path)
    plt.close(fig)


def plot_results(alt_summary_path: Path, path: Path) -> None:
    alt_rows = read_csv(alt_summary_path)
    lookup = {
        (
            row["lead_frequency_class"],
            row["scenario"],
            int(row["n_haplotypes"]),
            int(row["n_partners"]),
            row["strategy"],
        ): row
        for row in alt_rows
    }

    fig = plt.figure(figsize=(7.2, 4.95), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.105,
        right=0.985,
        bottom=0.13,
        top=0.82,
        hspace=0.40,
        wspace=0.18,
    )
    axes = np.array(
        [[fig.add_subplot(grid[row, column]) for column in range(3)] for row in range(2)]
    )

    positions_by_k = {
        4: np.array((0.0, 1.0, 2.0)),
        32: np.array((4.0, 5.0, 6.0)),
        256: np.array((8.0, 9.0, 10.0)),
    }
    panel_letter = ord("A")
    for row_index, lead_class in enumerate(LEAD_CLASSES):
        for column_index, scenario in enumerate(SCENARIOS):
            ax = axes[row_index, column_index]
            for boundary in (3.0, 7.0):
                ax.axvline(boundary, color="#DDDDDD", linewidth=0.6, zorder=0)
            for strategy in STRATEGIES:
                for partners in PARTNER_COUNTS:
                    positions = positions_by_k[partners]
                    rows = [
                        lookup[(lead_class, scenario, n, partners, strategy)]
                        for n in SAMPLE_SIZES
                    ]
                    values = np.array(
                        [100 * float(item["error_given_available"]) for item in rows]
                    )
                    ax.plot(
                        positions,
                        values,
                        color=STRATEGY_COLOR[strategy],
                        linewidth=0.9,
                        zorder=2,
                    )
                    for x, value, item in zip(positions, values, rows):
                        complete = int(item["n_available"]) == int(item["n_panels"])
                        ax.plot(
                            x,
                            value,
                            linestyle="none",
                            marker=STRATEGY_MARKER[strategy],
                            markersize=3.7,
                            markerfacecolor=(
                                STRATEGY_COLOR[strategy] if complete else "white"
                            ),
                            markeredgecolor=STRATEGY_COLOR[strategy],
                            markeredgewidth=0.8,
                            zorder=3,
                        )
            ax.set_xlim(-0.55, 10.55)
            ax.set_ylim(-3, 103)
            ax.set_yticks((0, 25, 50, 75, 100))
            ax.grid(axis="y", alpha=0.7)
            ax.set_xticks(
                np.concatenate(tuple(positions_by_k.values())),
                ("500", "1k", "2k") * 3,
            )
            if row_index == 0:
                ax.set_title(SCENARIO_LABEL[scenario], pad=28)
            if column_index > 0:
                ax.tick_params(labelleft=False)
            if row_index == 1:
                ax.set_xlabel("Phased haplotypes")
            else:
                ax.tick_params(labelbottom=False)
            for partners, center in ((4, 1.0), (32, 5.0), (256, 9.0)):
                ax.text(
                    center,
                    1.015,
                    f"$k={partners}$",
                    ha="center",
                    va="bottom",
                    fontsize=6.7,
                    transform=ax.get_xaxis_transform(),
                )
            ax.text(
                -0.18,
                1.08,
                chr(panel_letter),
                weight="bold",
                transform=ax.transAxes,
            )
            panel_letter += 1
        axes[row_index, 0].text(
            0.025,
            0.94,
            f"{lead_class.capitalize()} lead",
            ha="left",
            va="top",
            weight="bold",
            transform=axes[row_index, 0].transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        )

    fig.text(
        0.02,
        0.45,
        "Exact-mode error among panels where the method is available, %",
        ha="center",
        va="center",
        rotation=90,
    )
    strategy_handles = [
        Line2D(
            [0],
            [0],
            color=STRATEGY_COLOR[strategy],
            marker=STRATEGY_MARKER[strategy],
            markerfacecolor=STRATEGY_COLOR[strategy],
            markeredgecolor=STRATEGY_COLOR[strategy],
            markeredgewidth=0.8,
            label=STRATEGY_LABEL[strategy],
        )
        for strategy in STRATEGIES
    ]
    availability_handle = Line2D(
        [0],
        [0],
        color="#555555",
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor="#555555",
        markeredgewidth=0.8,
        label="Open marker: fewer than 30 panels evaluable",
    )
    fig.legend(
        handles=strategy_handles + [availability_handle],
        loc="upper center",
        bbox_to_anchor=(0.55, 0.995),
        ncol=4,
    )
    fig.savefig(path)
    plt.close(fig)


def percentile(values: list[float], probability: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), probability))


def plot_hamming(strategy_rows_path: Path, path: Path) -> None:
    rows = [
        row
        for row in read_csv(strategy_rows_path)
        if row["strategy"] == "empirical_mode"
        and int(row["lead_state"]) == 1
        and as_bool(row["available"])
    ]
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        partners = int(row["n_partners"])
        grouped[
            (row["lead_frequency_class"], partners, int(row["n_haplotypes"]))
        ].append(float(row["hamming_to_true_mode"]) / partners)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.2), sharey=True, constrained_layout=False
    )
    positions = []
    labels = []
    for partners_index, partners in enumerate(PARTNER_COUNTS):
        for sample_index, sample_size in enumerate(SAMPLE_SIZES):
            positions.append(partners_index * 4 + sample_index)
            labels.append({500: "500", 1000: "1k", 2000: "2k"}[sample_size])
    for axis_index, lead_class in enumerate(LEAD_CLASSES):
        ax = axes[axis_index]
        data = [
            grouped[(lead_class, partners, sample_size)]
            for partners in PARTNER_COUNTS
            for sample_size in SAMPLE_SIZES
        ]
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.62,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#333333", "linewidth": 0.9},
            whiskerprops={"color": "#666666", "linewidth": 0.6},
            capprops={"color": "#666666", "linewidth": 0.6},
            boxprops={"edgecolor": "#666666", "linewidth": 0.6},
        )
        for patch, partners in zip(
            box["boxes"],
            [partners for partners in PARTNER_COUNTS for _ in SAMPLE_SIZES],
        ):
            patch.set_facecolor(PARTNER_COLOR[partners])
            patch.set_alpha(0.55)
        for boundary in (3.0, 7.0):
            ax.axvline(boundary, color="#DDDDDD", linewidth=0.6, zorder=0)
        ax.set_xticks(positions, labels)
        ax.set_ylim(-0.02, 0.62)
        ax.grid(axis="y", alpha=0.7)
        ax.set_title(f"{lead_class.capitalize()} lead", pad=27)
        ax.set_xlabel("Phased haplotypes")
        for partners, center in ((4, 1), (32, 5), (256, 9)):
            ax.text(
                center,
                1.015,
                f"$k={partners}$",
                ha="center",
                va="bottom",
                fontsize=7,
                transform=ax.get_xaxis_transform(),
            )
        ax.text(
            0.0,
            1.06,
            chr(ord("A") + axis_index),
            weight="bold",
            transform=ax.transAxes,
        )
    axes[0].set_ylabel("Empirical-mode Hamming distance / partners")
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.18, top=0.78, wspace=0.15)
    fig.savefig(path)
    plt.close(fig)


def plot_diagnostics(panel_rows_path: Path, path: Path) -> None:
    rows = read_csv(panel_rows_path)
    grouped: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["lead_frequency_class"],
                row["scenario"],
                int(row["n_partners"]),
                int(row["n_haplotypes"]),
            )
        ].append(row)

    fig, axes = plt.subplots(
        2, 3, figsize=(7.2, 4.6), sharex=True, sharey="row", constrained_layout=False
    )
    for column, scenario in enumerate(SCENARIOS):
        for partners in PARTNER_COUNTS:
            for lead_class in LEAD_CLASSES:
                residual = []
                pva = []
                for sample_size in SAMPLE_SIZES:
                    cell = grouped[(lead_class, scenario, partners, sample_size)]
                    residual.append(
                        np.median(
                            [float(row["binary_correlation_residual_rmse"]) for row in cell]
                        )
                    )
                    pva.append(np.median([float(row["pva"]) for row in cell]))
                axes[0, column].plot(
                    SAMPLE_SIZES,
                    residual,
                    color=PARTNER_COLOR[partners],
                    linestyle=LEAD_LINESTYLE[lead_class],
                )
                axes[1, column].plot(
                    SAMPLE_SIZES,
                    pva,
                    color=PARTNER_COLOR[partners],
                    linestyle=LEAD_LINESTYLE[lead_class],
                )
        axes[0, column].set_title(SCENARIO_LABEL[scenario])
        axes[1, column].set_xlabel("Phased haplotypes")
        for row_index in range(2):
            axes[row_index, column].set_xticks(SAMPLE_SIZES, ("500", "1,000", "2,000"))
            axes[row_index, column].grid(axis="y", alpha=0.7)
            axes[row_index, column].text(
                -0.13,
                1.06,
                chr(ord("A") + row_index * 3 + column),
                weight="bold",
                transform=axes[row_index, column].transAxes,
            )
    axes[0, 0].set_ylabel("Median residual-correlation RMSE")
    axes[1, 0].set_ylabel("Median HaploPerturb PVA")
    axes[1, 0].set_ylim(0, 1)
    partner_handles = [
        Line2D([0], [0], color=PARTNER_COLOR[k], label=f"$k={k}$")
        for k in PARTNER_COUNTS
    ]
    lead_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=LEAD_LINESTYLE[lead_class],
            label=f"{lead_class.capitalize()} lead",
        )
        for lead_class in LEAD_CLASSES
    ]
    fig.legend(
        handles=partner_handles + lead_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=5,
    )
    fig.subplots_adjust(
        left=0.11, right=0.99, bottom=0.11, top=0.84, hspace=0.20, wspace=0.18
    )
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--style", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.style:
        plt.style.use(args.style)

    plot_design(args.output_dir / "figure_ranking_simulation_design_v77.pdf")
    plot_results(
        args.input_dir / "ranking_simulation_alt_summary_v77.csv",
        args.output_dir / "figure_ranking_simulation_results_v77.pdf",
    )
    plot_availability(
        args.input_dir / "ranking_simulation_lead_availability_v77.csv",
        args.output_dir / "supp_ranking_simulation_availability_v77.pdf",
    )
    plot_hamming(
        args.input_dir / "ranking_simulation_strategy_rows_v77.csv",
        args.output_dir / "supp_ranking_simulation_hamming_v77.pdf",
    )
    plot_diagnostics(
        args.input_dir / "ranking_simulation_panel_rows_v77.csv",
        args.output_dir / "supp_ranking_simulation_diagnostics_v77.pdf",
    )


if __name__ == "__main__":
    main()
