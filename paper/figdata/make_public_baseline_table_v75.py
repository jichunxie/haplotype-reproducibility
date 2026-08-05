#!/usr/bin/env python3
"""Generate the supplementary Arm B microglial locus--gene audit table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STRATEGIES = ("fitted_top1", "empirical_mode", "single_lead")


def rank_cell(row: pd.Series, strategy: str) -> str:
    value = str(int(row[f"rank_{strategy}"]))
    return rf"\textbf{{{value}}}" if bool(row[f"called_{strategy}"]) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.audit)
    frame = frame.loc[
        frame.arm.eq("armB") & frame.celltype.eq("Mic") & frame.truth
    ].copy()
    called = frame[[f"called_{strategy}" for strategy in STRATEGIES]].any(axis=1)
    frame = frame.loc[called].sort_values(["locus", "gene_name"])

    lines = [
        r"\begin{longtable}{p{0.28\linewidth}p{0.16\linewidth}rrr}",
        r"\caption{\textbf{Fujita-positive Arm B microglial locus--gene pairs selected by at least one sequence-construction strategy.} Ranks are among the 251 pairs in the common evaluation universe; bold ranks fall within the matched-count top 44 and are therefore called by that strategy. The LD-sign and empirical-mode ranks are identical. These are benchmark ranking results, not newly discovered eQTLs.}\label{tab:supp-public-baseline-genes}\\",
        r"\toprule",
        r"Lead locus & Gene & HaploPerturb & Empirical/LD & Single lead \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Lead locus & Gene & HaploPerturb & Empirical/LD & Single lead \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in frame.iterrows():
        locus = str(row.locus).replace("_", ":")
        lines.append(
            rf"\texttt{{{locus}}} & {row.gene_name} & "
            rf"{rank_cell(row, 'fitted_top1')} & "
            rf"{rank_cell(row, 'empirical_mode')} & "
            rf"{rank_cell(row, 'single_lead')} \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
