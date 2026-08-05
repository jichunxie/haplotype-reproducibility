#!/usr/bin/env python3
"""Build the revised main Figure 7 and its supplementary companion.

The main figure retains the public-minus-donor contrasts and compares the
public-panel HaploPerturb top-one, top-five and top-ten summaries with the empirical
mode, LD-sign and single-lead strategies.  The public- versus donor-panel
enrichment estimates formerly shown as Figure 7A--B are written to a separate
supplementary figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


RANKS = (1, 5, 10)
ARMS = ("armA", "armB")
ARM_LABEL = {"armA": "Arm A", "armB": "Arm B"}
PANEL_COLOR = {"onekg": "#0072B2", "rosmap": "#E69F00"}
PANEL_LABEL = {"onekg": "1000 Genomes", "rosmap": "ROS/MAP"}
PUBLIC_STRATEGIES = (
    "fitted_top1",
    "fitted_top5",
    "fitted_top10",
    "empirical_mode",
    "ld_sign",
    "single_lead",
)
PUBLIC_LABEL = {
    "fitted_top1": "HaploPerturb top 1",
    "fitted_top5": "HaploPerturb top 5",
    "fitted_top10": "HaploPerturb top 10",
    "empirical_mode": "Empirical mode",
    "ld_sign": "LD sign",
    "single_lead": "Single lead",
}
PUBLIC_COLOR = {
    "fitted_top1": "#0072B2",
    "fitted_top5": "#56B4E9",
    "fitted_top10": "#E69F00",
    "empirical_mode": "#009E73",
    "ld_sign": "#CC79A7",
    "single_lead": "#D55E00",
}
PUBLIC_MARKER = {
    "fitted_top1": "o",
    "fitted_top5": "s",
    "fitted_top10": "^",
    "empirical_mode": "D",
    "ld_sign": "P",
    "single_lead": "X",
}


def errorbar(ax: plt.Axes, x: np.ndarray, rows: list[dict], **kwargs) -> None:
    estimate = np.asarray([row["enrichment"] for row in rows], dtype=float)
    lower = np.asarray([row["boot_lo"] for row in rows], dtype=float)
    upper = np.asarray([row["boot_hi"] for row in rows], dtype=float)
    ax.errorbar(
        x,
        estimate,
        yerr=[estimate - lower, upper - estimate],
        capsize=3,
        **kwargs,
    )


def index_rows(reference: dict, simple: dict) -> tuple[dict, dict, dict]:
    reference_rows = {
        (row["arm"], row["celltype"], row["strategy"]): row
        for row in reference["rows"]
    }
    reference_differences = {
        (row["arm"], row["celltype"], row["left"], row["right"]): row
        for row in reference["paired_contrasts"]
    }
    simple_rows = {
        (row["arm"], row["celltype"], row["strategy"]): row
        for row in simple["rows"]
    }
    return reference_rows, reference_differences, simple_rows


def plot_main(reference: dict, simple: dict, output: Path) -> None:
    reference_rows, reference_differences, simple_rows = index_rows(reference, simple)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.8),
        sharey="row",
        squeeze=False,
        constrained_layout=False,
        gridspec_kw={"height_ratios": (0.92, 1.18)},
    )
    rank_x = np.arange(len(RANKS), dtype=float)

    for column, arm in enumerate(ARMS):
        difference_ax = axes[0, column]
        differences = [
            reference_differences[
                (arm, "Mic", f"onekg_top{rank}", f"rosmap_top{rank}")
            ]
            for rank in RANKS
        ]
        estimate = np.asarray([row["difference"] for row in differences])
        lower = np.asarray([row["diff_boot_lo"] for row in differences])
        upper = np.asarray([row["diff_boot_hi"] for row in differences])
        difference_ax.errorbar(
            rank_x,
            estimate,
            yerr=[estimate - lower, upper - estimate],
            fmt="o",
            color="#333333",
            capsize=3,
        )
        difference_ax.axhline(0, color="0.45", linestyle="--", linewidth=0.9)
        difference_ax.set_title(ARM_LABEL[arm])
        difference_ax.set_xticks(rank_x, ("Top 1", "Top 5", "Top 10"))
        difference_ax.set_ylabel(
            "1000 Genomes minus\nROS/MAP enrichment" if column == 0 else ""
        )
        difference_ax.grid(axis="y", alpha=0.3)

        public_ax = axes[1, column]
        public_x = np.arange(len(PUBLIC_STRATEGIES), dtype=float)
        for x, strategy in zip(public_x, PUBLIC_STRATEGIES):
            if strategy.startswith("fitted_top"):
                rank = int(strategy.removeprefix("fitted_top"))
                row = reference_rows[(arm, "Mic", f"onekg_top{rank}")]
            else:
                row = simple_rows[(arm, "Mic", strategy)]
            errorbar(
                public_ax,
                np.asarray([x]),
                [row],
                fmt=PUBLIC_MARKER[strategy],
                color=PUBLIC_COLOR[strategy],
            )
        public_ax.axhline(1, color="0.45", linestyle="--", linewidth=0.9)
        public_ax.set_xticks(
            public_x,
            tuple(PUBLIC_LABEL[strategy] for strategy in PUBLIC_STRATEGIES),
            fontsize=6.6,
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        public_ax.set_ylabel(
            "Public-panel strategy\nmatched-count enrichment" if column == 0 else ""
        )
        public_ax.grid(axis="y", alpha=0.3)

    for index, ax in enumerate(axes.flat):
        ax.text(
            -0.17,
            1.06,
            chr(ord("A") + index),
            weight="bold",
            transform=ax.transAxes,
        )

    fig.subplots_adjust(
        left=0.15,
        right=0.985,
        bottom=0.21,
        top=0.91,
        hspace=0.52,
        wspace=0.20,
    )
    fig.savefig(output)
    plt.close(fig)


def plot_supplement(reference: dict, output: Path) -> None:
    reference_rows, _, _ = index_rows(reference, {"rows": []})
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.1),
        sharey=True,
        squeeze=False,
        constrained_layout=False,
    )
    rank_x = np.arange(len(RANKS), dtype=float)
    panel_offsets = {"onekg": -0.10, "rosmap": 0.10}

    for column, arm in enumerate(ARMS):
        ax = axes[0, column]
        for panel in ("onekg", "rosmap"):
            rows = [
                reference_rows[(arm, "Mic", f"{panel}_top{rank}")]
                for rank in RANKS
            ]
            errorbar(
                ax,
                rank_x + panel_offsets[panel],
                rows,
                fmt="o",
                color=PANEL_COLOR[panel],
                label=PANEL_LABEL[panel],
            )
        single = reference_rows[(arm, "Mic", "single_lead")]
        ax.axhline(1, color="0.45", linestyle="--", linewidth=0.9)
        ax.axhline(single["enrichment"], color="0.45", linestyle=":", linewidth=1)
        ax.set_title(ARM_LABEL[arm])
        ax.set_xticks(rank_x, ("Top 1", "Top 5", "Top 10"))
        ax.set_ylabel("Matched-count enrichment" if column == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        ax.text(-0.17, 1.06, chr(ord("A") + column), weight="bold", transform=ax.transAxes)

    legend_handles = [
        Line2D(
            [0], [0], color=PANEL_COLOR[panel], marker="o", linestyle="none",
            label=PANEL_LABEL[panel]
        )
        for panel in ("onekg", "rosmap")
    ]
    legend_handles.append(
        Line2D(
            [0], [0], color="0.45", linestyle=":", label="Single-lead enrichment"
        )
    )
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.55, 0.99), ncol=3)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.75, wspace=0.16)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--simple-summary", type=Path, required=True)
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--supp-output", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    args = parser.parse_args()
    plt.style.use(args.style)
    reference = json.loads(args.reference_summary.read_text())
    simple = json.loads(args.simple_summary.read_text())
    plot_main(reference, simple, args.main_output)
    plot_supplement(reference, args.supp_output)


if __name__ == "__main__":
    main()
