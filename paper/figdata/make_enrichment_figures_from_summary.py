#!/usr/bin/env python3
"""Rebuild enrichment displays from disclosure-safe locked JSON summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from compare_public_baseline_enrichment_v75 import plot as plot_simple
from compare_single_lead_enrichment_v64 import plot as plot_rank

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    plt.style.use(HERE / "chap_figures.mplstyle")
    rank = json.loads((HERE / "strategy_enrichment_total_top10_v66.json").read_text())
    plot_rank(
        rank["rows"],
        ROOT / "main" / "figures" / "figure_microglia_strategy_enrichment_total_top10_v66.pdf",
        ["Mic"],
        3.7,
    )
    simple = json.loads((HERE / "public_baseline_enrichment_v75.json").read_text())
    plot_simple(
        simple["rows"],
        ROOT / "supp" / "supp-figures" / "supp_public_baselines_all_cells_v75.pdf",
        ["Ast", "Exc", "Mic"],
        7.2,
    )


if __name__ == "__main__":
    main()
