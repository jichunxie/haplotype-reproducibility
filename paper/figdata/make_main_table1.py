#!/usr/bin/env python3
"""Generate the cross-panel agreement table from the locked v72 summary."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "main" / "table1_cohort_haplotype_agreement.tex"


def integer(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", "{,}")


def main() -> None:
    artifact = json.loads((HERE / "cohort_top_haplotype_comparison_v72.json").read_text())
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Arm & Partner sets & Loci & Missing partners & Rank one exact & Top ten shared & Top-ten nearest $d$ \\",
        r"\midrule",
    ]
    for arm in artifact["by_arm"]:
        for label, key in (
            ("Same", "same_partner_set"),
            ("Different; shared only", "different_partner_set_shared_coordinates_only"),
        ):
            row = arm[key]
            missing = row["n_partners_missing_from_rosmap"]
            missing_text = "0" if label == "Same" else (
                f"{integer(missing['median'])} [{integer(missing['min'])},{integer(missing['max'])}]"
            )
            top1 = row["top1"]
            shared = row["top10_lists"]["shared_fraction_of_smaller_list"]["median"]
            nearest = row["top10_lists"]["nearest_hamming"]
            lines.append(
                f"{arm['arm'][-1]} & {label} & {row['n_loci']} & {missing_text} & "
                f"{top1['n_exact']}/{top1['n_rank_matched_haplotype_pairs']} & {shared:.2f} & "
                f"{integer(nearest['median'])} [{integer(nearest['q95'])},{integer(nearest['max'])}] \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
