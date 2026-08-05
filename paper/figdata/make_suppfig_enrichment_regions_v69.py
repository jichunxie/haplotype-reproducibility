#!/usr/bin/env python3
"""Plot complete-transcript and terminal-exon enrichment for all cell types."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CELL_ORDER = ["Ast", "Exc", "Mic"]
CELL_LABEL = {
    "Ast": "Astrocyte",
    "Exc": "Excitatory neuron",
    "Mic": "Microglia proxy",
}
STRATEGY_ORDER = ["top10", "top5", "top1", "single_lead"]
STRATEGY_LABEL = {
    "top10": "HaploPerturb top ten",
    "top5": "HaploPerturb top five",
    "top1": "HaploPerturb top one",
    "single_lead": "Single lead",
}
STRATEGY_COLOR = {
    "top10": "#CC79A7",
    "top5": "#0072B2",
    "top1": "#009E73",
    "single_lead": "#D55E00",
}


def load_rows(path: Path) -> dict[tuple[str, str, str], dict]:
    payload = json.loads(path.read_text())
    rows = {
        (row["celltype"], row["arm"], row["strategy"]): row
        for row in payload["rows"]
    }
    expected = {
        (cell, arm, strategy)
        for cell in CELL_ORDER
        for arm in ("armA", "armB")
        for strategy in STRATEGY_ORDER
    }
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise ValueError(f"unexpected row keys; missing={missing}, extra={extra}")
    return rows


def plot_panel(ax, rows: dict, cell: str, panel: str) -> None:
    offsets = np.linspace(-0.27, 0.27, len(STRATEGY_ORDER))
    for offset, strategy in zip(offsets, STRATEGY_ORDER):
        records = [rows[(cell, arm, strategy)] for arm in ("armA", "armB")]
        x = np.array([0.0, 1.0]) + offset
        y = np.array([record["enrichment"] for record in records])
        lo = np.array([record["boot_lo"] for record in records])
        hi = np.array([record["boot_hi"] for record in records])
        ax.errorbar(
            x,
            y,
            yerr=np.vstack((y - lo, hi - y)),
            fmt="o",
            color=STRATEGY_COLOR[strategy],
            capsize=2.5,
            label=STRATEGY_LABEL[strategy],
        )
    ax.axhline(1.0, color="0.45", linestyle="--", linewidth=0.9)
    ax.set_xticks([0, 1], ["Arm A", "Arm B"])
    ax.set_xlim(-0.48, 1.48)
    ax.set_ylim(0.15, 3.85)
    ax.set_title(CELL_LABEL[cell])
    ax.grid(axis="y", alpha=0.35)
    ax.text(
        0.015,
        0.975,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--complete", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plt.style.use(args.style)
    complete = load_rows(args.complete)
    terminal = load_rows(args.terminal)

    fig, axes = plt.subplots(2, 3, figsize=(7.3189, 4.55), sharex=True, sharey=True)
    for col, cell in enumerate(CELL_ORDER):
        plot_panel(axes[0, col], complete, cell, chr(ord("A") + col))
        plot_panel(axes[1, col], terminal, cell, chr(ord("D") + col))

    axes[0, 0].set_ylabel("Complete exon union\nMatched-count enrichment")
    axes[1, 0].set_ylabel("Terminal-exon tail (1 kb)\nMatched-count enrichment")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.94), h_pad=0.12, w_pad=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
