#!/usr/bin/env python3
"""Combine protected fitted ranks 1--5 and 6--10 with strict coverage checks."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-five", type=Path, required=True)
    ap.add_argument("--last-five", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    a, b = pd.read_parquet(args.first_five), pd.read_parquet(args.last_five)
    if not set(a["rank"].unique()).issubset(set(range(1, 6))):
        raise ValueError("first-five input contains rank >5")
    if not set(b["rank"].unique()).issubset(set(range(6, 11))):
        raise ValueError("last-five input contains rank outside 6--10")
    out = pd.concat([a, b], ignore_index=True)
    if out.duplicated().any():
        raise ValueError("exact duplicate prediction rows")
    for (arm, locus), group in out.groupby(["arm", "locus"]):
        ranks = sorted(group["rank"].unique())
        if ranks != list(range(1, max(ranks) + 1)) or max(ranks) > 10:
            raise ValueError((arm, locus, ranks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print({"rows": len(out), "arm_loci": out[["arm", "locus"]].drop_duplicates().shape[0],
           "max_rank": int(out["rank"].max())})


if __name__ == "__main__":
    main()
