#!/usr/bin/env python
"""Count linked partners for each Alzheimer disease lead at fixed thresholds.

The stored per-locus LD table
    {ROOT}/{locus}/ld_query/lead_ld_partners.tsv.gz
holds every lead-touching pair inside a +/- 524 kb window, computed by PLINK on the 1kGP 30x EUR
superpopulation (633 samples, related included), after MAF_EUR_unrel >= 0.01 and HWE_EUR >= 1e-6,
and floored at r^2 >= 0.1. This is the table thresholded by the production partner-selection step.

r^2 is sign-invariant, so allele-orientation corrections to signed PHASED_R do not affect these
counts. Counts are reported on both scales because "threshold 0.5" can otherwise be ambiguous:
  n_r2_ge_t   -- PHASED_R2 >= t          (the paper's partner-threshold convention)
  n_absr_ge_t -- |PHASED_R| >= t, i.e. r^2 >= t^2 (the looser reading)

Deterministic. No randomness, no seed. No AlphaGenome calls; reads the original lead-partner tree.

Locked-result assertions require that at r^2 >= 0.8 the totals are
1,918 partners across 38 loci, median 9, max 1,285; at r^2 >= 0.5, 4,350 total, median 24, max 2,693.
Fails loudly on mismatch.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = "/hpc/group/xielab/cnm53/results/ldhagx_runs/AD_eQTL"
WORK = "/hpc/group/xielab/jx42/CHAP/work"
LEADS = f"{WORK}/leads_for_ldsens.json"
OUT = f"{WORK}/partners_by_threshold_v36.json"
THRESH = [0.5, 0.6, 0.7, 0.8]

leads = json.load(open(LEADS))
rows = []
for r in leads:
    loc, lead = r["locus"], r["lead"]
    lp = f"{ROOT}/{loc}/ld_query/lead_ld_partners.tsv.gz"
    if not os.path.exists(lp):
        raise SystemExit(f"missing LD table for {loc}")
    d = pd.read_csv(lp, sep="\t")
    sw = d.ID_A != lead
    d.loc[sw, ["ID_A", "ID_B"]] = d.loc[sw, ["ID_B", "ID_A"]].values
    d = d[d.ID_B != lead]
    if (d.ID_A != lead).any():
        raise SystemExit(f"{loc}: rows not lead-touching after orientation")
    # one row per candidate partner; keep the strongest if a variant appears twice
    g = d.groupby("ID_B", sort=True)["PHASED_R2"].max()
    n_dup = len(d) - len(g)
    r2 = g.to_numpy()
    row = {
        "locus": loc,
        "lead": lead,
        "n_published": len(r["ids"]) - 1,
        "n_rows_in_table": int(len(d)),
        "n_dup_collapsed": int(n_dup),
        "n_candidates_r2_ge_0.1": int(len(r2)),
        "max_r2": float(r2.max()) if len(r2) else float("nan"),
    }
    for t in THRESH:
        row[f"n_r2_ge_{t}"] = int((r2 >= t).sum())
        row[f"n_absr_ge_{t}"] = int((r2 >= t * t).sum())
    rows.append(row)

df = pd.DataFrame(rows).sort_values("n_r2_ge_0.8", ascending=False).reset_index(drop=True)

# The 27 Arm A loci have at least one partner at r^2 >= 0.8.
in27 = df["n_r2_ge_0.8"] > 0
df["in_27"] = in27

# Locked partner-count assertions.
chk = {
    0.8: (1918, 9, 1285),
    0.5: (4350, 24, 2693),
}
for t, (tot, med, mx) in chk.items():
    c = df[f"n_r2_ge_{t}"]
    got = (int(c.sum()), int(np.median(c)), int(c.max()))
    if got != (tot, med, mx):
        raise SystemExit(
            f"REPRODUCTION FAILED at r2>={t}: total/median/max = {got}, expected {(tot, med, mx)}"
        )
if int(in27.sum()) != 27:
    raise SystemExit(f"REPRODUCTION FAILED: {int(in27.sum())} loci with >=1 partner at r2>=0.8, expected 27")
print("partner-count assertions passed (r2>=0.8, r2>=0.5, and 27 Arm A loci)\n")

cols = ["locus", "n_published"] + [f"n_r2_ge_{t}" for t in THRESH] + ["n_candidates_r2_ge_0.1", "in_27"]
with pd.option_context("display.width", 200, "display.max_rows", 60):
    print(df[cols].to_string(index=False))

sub = df[in27]
print("\n--- summary over the 27 loci (r^2 scale) ---")
for t in THRESH:
    c = sub[f"n_r2_ge_{t}"]
    print(f"r2 >= {t}: total {int(c.sum()):>6}  median {np.median(c):>7.1f}  "
          f"IQR [{np.percentile(c, 25):.1f}, {np.percentile(c, 75):.1f}]  "
          f"min {int(c.min()):>4}  max {int(c.max()):>5}  loci with 0 partners {int((c == 0).sum())}")
print("\n--- same, on the |r| scale (r^2 >= t^2) ---")
for t in THRESH:
    c = sub[f"n_absr_ge_{t}"]
    print(f"|r| >= {t}: total {int(c.sum()):>6}  median {np.median(c):>7.1f}  "
          f"min {int(c.min()):>4}  max {int(c.max()):>5}")
print("\n--- all 38 loci, r^2 scale ---")
for t in THRESH:
    c = df[f"n_r2_ge_{t}"]
    print(f"r2 >= {t}: total {int(c.sum()):>6}  median {np.median(c):>7.1f}  "
          f"loci with >=1 partner {int((c > 0).sum())}/38")

meta = {
    "script": "count_partners_v36.py",
    "date": "2026-07-27",
    "question": "partner count per locus at LD thresholds 0.5/0.6/0.7/0.8",
    "source_table": f"{ROOT}/<locus>/ld_query/lead_ld_partners.tsv.gz",
    "panel": "1kGP 30x, EUR superpopulation, 633 samples (related included)",
    "panel_note": ("ld_v24 re-derives phased LD on the 503-sample EUR UNRELATED panel for the "
                   "r^2>=0.8 sets only; these counts come from the stored 633-sample table, which is "
                   "the table the partner-selection step actually thresholds"),
    "window_kb": 524,
    "filters": {"maf_min": 0.01, "maf_key": "MAF_EUR_unrel", "hwe_min": 1e-06,
                "hwe_key": "HWE_EUR", "r2_floor_of_stored_table": 0.1},
    "sign_note": "r^2 is sign-invariant; the known PHASED_R sign error does not affect these counts",
    "dedup": "one row per candidate partner, max PHASED_R2 if duplicated",
    "deterministic": True,
    "seed": None,
    "n_loci_total": int(len(df)),
    "n_loci_in_27": int(in27.sum()),
    "reproduces": "locked counts at r2>=0.8 (1918/9/1285) and r2>=0.5 (4350/24/2693)",
}
json.dump({"meta": meta, "per_locus": df.to_dict(orient="records")}, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
