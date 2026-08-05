#!/usr/bin/env python3
"""Build every active paper figure and table from released aggregate artifacts."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "figdata"
MAIN = ROOT / "paper" / "main" / "figures"
SUPP = ROOT / "paper" / "supp" / "supp-figures"
TABLES = ROOT / "paper" / "supp" / "supp-tables"
STYLE = DATA / "chap_figures.mplstyle"
RANKING = ROOT / "paper" / "ranking-generated"


def run(*arguments: str | Path) -> None:
    command = [sys.executable, *(str(value) for value in arguments)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    MAIN.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    run(DATA / "make_ld_example_chr17.py")
    run(DATA / "make_figure_pva_q1.py")
    run(DATA / "make_figure_top10_rank_validation.py")
    run(DATA / "make_suppfig_S2_q1_residual_heatmap.py")
    run(DATA / "make_suppfig_S3_q1_loading_density.py")
    run(DATA / "make_enrichment_figures_from_summary.py")
    run(
        DATA / "make_suppfig_enrichment_regions_v69.py",
        "--complete", DATA / "strategy_enrichment_total_top10_v66.json",
        "--terminal", DATA / "strategy_enrichment_terminal_top10_v66.json",
        "--style", STYLE,
        "--output", SUPP / "supp_strategy_enrichment_full_and_terminal_top10_all_cells_v69.pdf",
    )
    run(
        DATA / "make_ranking_simulation_figures_v77.py",
        "--input-dir", DATA,
        "--output-dir", RANKING,
        "--style", STYLE,
    )
    for name in ("figure_ranking_simulation_design_v77.pdf", "figure_ranking_simulation_results_v77.pdf"):
        shutil.copy2(RANKING / name, MAIN / name)
    for name in (
        "supp_ranking_simulation_availability_v77.pdf",
        "supp_ranking_simulation_hamming_v77.pdf",
        "supp_ranking_simulation_diagnostics_v77.pdf",
    ):
        shutil.copy2(RANKING / name, SUPP / name)
    run(
        DATA / "make_figure7_reference_panel_enrichment_v78.py",
        "--reference-summary", DATA / "reference_panel_enrichment_v68.json",
        "--simple-summary", DATA / "public_baseline_enrichment_v75.json",
        "--main-output", MAIN / "figure_microglia_reference_panel_comparison_v78.pdf",
        "--supp-output", SUPP / "supp_microglia_reference_panel_enrichment_v78.pdf",
        "--style", STYLE,
    )
    run(DATA / "make_main_table1.py")
    run(DATA / "make_supptable_ST1.py")
    run(
        DATA / "make_public_baseline_table_v75.py",
        "--audit", DATA / "public_baseline_locus_gene_audit_v75.csv",
        "--output", TABLES / "ST2_public_baseline_genes_v75.tex",
    )
    expected = [
        *[MAIN / name for name in (
            "figure_ld_example_chr17.pdf",
            "figure_pva_q1.pdf",
            "figure_top10_rank_validation.pdf",
            "figure_ranking_simulation_design_v77.pdf",
            "figure_ranking_simulation_results_v77.pdf",
            "figure_microglia_strategy_enrichment_total_top10_v66.pdf",
            "figure_microglia_reference_panel_comparison_v78.pdf",
        )],
        *[SUPP / name for name in (
            "S2_q1_residual_heatmap.pdf",
            "S3_q1_loading_density.pdf",
            "supp_strategy_enrichment_full_and_terminal_top10_all_cells_v69.pdf",
            "supp_ranking_simulation_diagnostics_v77.pdf",
            "supp_ranking_simulation_hamming_v77.pdf",
            "supp_public_baselines_all_cells_v75.pdf",
            "supp_ranking_simulation_availability_v77.pdf",
            "supp_microglia_reference_panel_enrichment_v78.pdf",
        )],
        ROOT / "paper" / "main" / "table1_cohort_haplotype_agreement.tex",
        TABLES / "ST1_verify_mode.tex",
        TABLES / "ST2_public_baseline_genes_v75.tex",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.exists() or not path.stat().st_size]
    if missing:
        raise RuntimeError(f"missing or empty outputs: {missing}")
    print(f"Built {len(expected)} active figures/tables.")


if __name__ == "__main__":
    main()
