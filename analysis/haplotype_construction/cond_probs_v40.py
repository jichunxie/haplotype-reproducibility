#!/usr/bin/env python
"""Normalize stored haplotype probabilities by the lead-state marginal.

The source artifacts contain joint probabilities Pr(X=x, X_0=1). This script
also reports the conditional values Pr(X=x | X_0=1) by dividing by the lead
alternate-allele frequency p_0 used to define the fitted threshold.

Because p_0 is constant across candidates at one locus and lead state, this
normalization changes absolute probabilities but not rankings, top-one
identity, or log-probability gaps. The script retains both forms for audit.

Validation checks require conditional probabilities in (0, 1], non-positive
conditional log probabilities, and an unchanged top-two log gap. The
calculation is deterministic and makes no AlphaGenome calls.
"""
import json
import os

import numpy as np
import pandas as pd

W = "/hpc/group/xielab/jx42/CHAP/work"
V24 = f"{W}/ld_v24"
OUT = f"{W}/cond_probs_v40.json"

leads = {r["locus"]: r["lead"] for r in json.load(open(f"{W}/leads_for_ldsens.json"))}


def p0_of(loc):
    A = pd.read_csv(f"{V24}/{loc}.afreq", sep="\t")
    A.columns = [c.lstrip("#") for c in A.columns]
    fc = "ALT_FREQS" if "ALT_FREQS" in A.columns else "ALT_FREQ"
    v = A.loc[A.ID.astype(str) == leads[loc], fc]
    if not len(v):
        raise SystemExit(f"{loc}: lead {leads[loc]} absent from .afreq")
    return float(v.iloc[0])


v28 = {r["locus"]: r for r in json.load(open(f"{W}/topM_v28.json"))["per_locus"]}
v31 = {r["locus"]: r for r in json.load(open(f"{W}/verify_mode_v31.json"))["per_locus"]}
v33 = {r["locus"]: r for r in json.load(open(f"{W}/verify_sparse_v33.json"))["per_locus"]}

rows = []
for loc in sorted(set(v28) | set(v31) | set(v33)):
    p0 = p0_of(loc)
    lp0 = float(np.log(p0))
    o = {"locus": loc, "p_0": p0, "log_p_0": lp0}

    if loc in v28:
        r = v28[loc]
        o["factor_top1_prob_joint"] = r["top1_prob"]
        o["factor_top1_prob_cond"] = r["top1_prob"] / p0
        o["factor_chain_mass_joint"] = r["chain_mass"]
        o["factor_chain_mass_cond"] = r["chain_mass"] / p0
        lp = np.asarray(r["log_probs"], dtype=float)
        o["factor_logp_cond_top20"] = (lp - lp0)[:20].tolist()
        # differences must survive the shift exactly
        assert abs((lp[0] - lp[1]) - ((lp - lp0)[0] - (lp - lp0)[1])) < 1e-12

    if loc in v31:
        r = v31[loc]
        for k in ("true_R_p_mode", "true_R_p_runner_up"):
            if r.get(k) is not None:
                o[k + "_joint"] = r[k]
                o[k + "_cond"] = r[k] / p0

    if loc in v33:
        r = v33[loc]
        if r.get("true_R_p_top") is not None:
            o["true_R_p_top_joint"] = r["true_R_p_top"]
            o["true_R_p_top_cond"] = r["true_R_p_top"] / p0
        if r.get("true_R_logp"):
            tl = np.asarray(r["true_R_logp"], dtype=float)
            o["true_R_logp_cond"] = (tl - lp0).tolist()
        o["true_R_logp_gap"] = r.get("true_R_logp_gap")   # a difference: unchanged
    rows.append(o)

# ---- asserts
for o in rows:
    for k, v in o.items():
        if k.endswith("_cond") and isinstance(v, float):
            if not (0.0 < v <= 1.0 + 1e-9):
                raise SystemExit(f"{o['locus']}: corrected {k} = {v!r} outside (0, 1]")
        if k.endswith("_cond") and isinstance(v, list):
            if max(v) > 1e-9:
                raise SystemExit(f"{o['locus']}: corrected {k} has a positive log-probability")

def summarise(key):
    v = np.asarray([o[key] for o in rows if key in o], dtype=float)
    return f"median {np.median(v):.3f}  min {v.min():.3f}  max {v.max():.3f}  (n={len(v)})"

print("p_0 (lead alternate-allele frequency): " + summarise("p_0"))
print()
for a, b, lab in [("factor_top1_prob_joint", "factor_top1_prob_cond", "factor-model top-1"),
                  ("true_R_p_top_joint", "true_R_p_top_cond", "working-R top-1"),
                  ("factor_chain_mass_joint", "factor_chain_mass_cond", "factor chain mass")]:
    print(f"{lab}:")
    print(f"   as stored (JOINT):     {summarise(a)}")
    print(f"   corrected (CONDITIONAL): {summarise(b)}")

json.dump({"meta": {"script": "cond_probs_v40.py", "date": "2026-07-27",
                    "defect": "build_topM_v28.py:84 labels a JOINT probability as conditional; the "
                              "division by p_0 = Phi(-tau_0) is never applied. Same omission in the "
                              "GHK rectangle evaluation.",
                    "correction": "prob -> prob / p_0 ; logprob -> logprob - log p_0",
                    "unaffected": "all log-probability DIFFERENCES: rankings, top-1 identity, "
                                  "lead-over-runner-up gap, gap/SE, rel_logprob_*, logp_deficit_*",
                    "sources": ["topM_v28.json", "verify_mode_v31.json", "verify_sparse_v33.json"],
                    "p_0_source": "ld_v24 <locus>.afreq, lead ALT frequency -- the same file the "
                                  "thresholds come from",
                    "deterministic": True, "seed": None},
           "per_locus": rows}, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}  ({len(rows)} loci)")
