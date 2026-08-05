#!/usr/bin/env python
"""Discover and certify the top 10 q=1 haplotypes, then count them in phased data.

The fitted law is conditional on X[:, 0] == 1. Candidate discovery uses independent
draws from that law. Candidate probabilities are recomputed deterministically by
Gauss--Legendre quadrature. If p10 is the fitted probability of the tenth candidate,
at most 1/p10 haplotypes can have probability >= p10; hence the probability that an
N-draw search misses any such haplotype is at most exp(-N*p10)/p10.

"Empirical probability" means count(x among X with X0=1) / count(X0=1). It is an
in-sample descriptive frequency, not the unfactored Gaussian rectangle probability.
"""
import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri
from scipy.stats import binom, norm


TOP_L = 10
SEED_BASE = 590731
INITIAL_DRAWS = int(os.environ.get("TOP10_INITIAL_DRAWS", "100000"))
MAX_DRAWS = int(os.environ.get("TOP10_MAX_DRAWS", "2000000"))
BATCH = 10_000
DISCOVERY_KEEP = 50
QUAD_ORDER = 1024
REFINE_ORDER = 2048
DELTA = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def packed_key(x):
    return np.packbits(x, bitorder="little").tobytes()


def unpack_key(key, p):
    return np.unpackbits(np.frombuffer(key, dtype=np.uint8),
                         bitorder="little")[:p].astype(np.uint8)


def quadrature_setup(B, psi, tau, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    f = 8.0 * nodes
    logw = np.log(8.0 * weights) + norm.logpdf(f)
    b = B[:, 0]
    sd = np.sqrt(psi)
    z = (np.outer(b, f) - tau[:, None]) / sd[:, None]
    return logw, log_ndtr(z), log_ndtr(-z)


def score(candidates, logw, logq, log1q, logp0):
    ans = np.empty(len(candidates))
    for i, x in enumerate(candidates):
        xb = x.astype(bool)
        ans[i] = logsumexp(
            logw + logq[xb].sum(axis=0) + log1q[~xb].sum(axis=0)
        ) - logp0
    return ans


def deterministic_candidates(B, tau):
    """Conditional-mode cells plus all one-coordinate flips of their best cell."""
    b = B[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        cuts = np.where(np.abs(b) > 1e-14, tau / b, np.inf)
    cuts = np.unique(cuts[np.isfinite(cuts)])
    if len(cuts):
        reps = np.r_[cuts[0] - 1.0, (cuts[:-1] + cuts[1:]) / 2.0, cuts[-1] + 1.0]
    else:
        reps = np.array([0.0])
    chain = (b[None, :] * reps[:, None] > tau[None, :]).astype(np.uint8)
    chain[:, 0] = 1
    chain = np.unique(chain, axis=0)
    return chain


def sample_keys(rng, B, psi, tau, lead_state, n, counter):
    """Exact draws from X | X0=lead_state for q=1, as packed byte keys."""
    b = B[:, 0]
    p_event = float(ndtr(-tau[0]) if lead_state == 1 else ndtr(tau[0]))
    left = n
    while left:
        m = min(BATCH, left)
        if lead_state == 1:
            z0 = ndtri(1.0 - rng.random(m) * p_event)
        else:
            z0 = ndtri(rng.random(m) * p_event)
        f = b[0] * z0 + math.sqrt(psi[0]) * rng.standard_normal(m)
        q = ndtr((f[:, None] * b[None, :] - tau[None, :]) /
                 np.sqrt(psi)[None, :])
        x = (rng.random(q.shape) < q).astype(np.uint8)
        x[:, 0] = lead_state
        packed = np.packbits(x, axis=1, bitorder="little")
        unique, counts = np.unique(packed, axis=0, return_counts=True)
        for row, count in zip(unique, counts):
            counter[row.tobytes()] += int(count)
        left -= m


def coverage_bound(n, p10):
    if p10 <= 0:
        return math.inf
    log_bound = -n * p10 - math.log(p10)
    return math.exp(log_bound) if log_bound < 700 else math.inf


def screening_bound(n, p10, cutoff_count):
    """Union bound for any haplotype with p >= p10 failing the count screen."""
    if p10 <= 0:
        return math.inf
    log_bound = float(binom.logcdf(cutoff_count, n, p10)) - math.log(p10)
    return math.exp(log_bound) if log_bound < 700 else math.inf


def required_draws(p10):
    if p10 <= 0:
        return MAX_DRAWS + 1
    return int(math.ceil((math.log(1.0 / DELTA) + math.log(1.0 / p10)) / p10))


def analyse_locus(dataset, arm, lead_state, input_path, coef_path):
    inp = np.load(input_path, allow_pickle=False)
    fit = np.load(coef_path, allow_pickle=False)
    X = inp["X"].astype(np.uint8)
    ids = inp["ids"].astype(str)
    fit_ids = fit["ids"].astype(str)
    if not np.array_equal(ids, fit_ids):
        raise ValueError(f"{input_path}: input and coefficient IDs differ")
    B = fit["B"].astype(float)
    psi = fit["psi"].astype(float)
    tau = fit["tau"].astype(float)
    if B.shape != (X.shape[1], 1):
        raise ValueError(f"{coef_path}: expected q=1 coefficients")
    if np.max(np.abs(psi - (1.0 - np.square(B[:, 0])))) > 1e-10:
        raise ValueError(f"{coef_path}: unit-variance identity failed")

    logp0 = float(
        log_ndtr(-tau[0]) if lead_state == 1 else log_ndtr(tau[0])
    )
    logw, logq, log1q = quadrature_setup(B, psi, tau, QUAD_ORDER)
    locus_seed = SEED_BASE + int(hashlib.sha256(
        f"{dataset}|{arm}|lead{lead_state}|{input_path.stem}".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(locus_seed)
    counts = Counter()
    n_draws = 0
    target = INITIAL_DRAWS
    final = None
    target_l = min(TOP_L, 2 ** (X.shape[1] - 1)) if X.shape[1] <= 20 else TOP_L

    while True:
        sample_keys(rng, B, psi, tau, lead_state, target - n_draws, counts)
        n_draws = target
        screened = counts.most_common(DISCOVERY_KEEP)
        frequent = [unpack_key(k, X.shape[1]) for k, _ in screened]
        screen_truncated = len(counts) > DISCOVERY_KEEP
        cutoff_count = int(screened[-1][1]) if screen_truncated else 0
        # The exact-draw coverage certificate is sufficient for top-L discovery. Do not append
        # every breakpoint cell or one-coordinate flip: scoring that O(p) expansion costs O(p^2 G)
        # and is prohibitive at the 2,694-variant Arm B locus.
        candidates = np.unique(np.asarray(frequent), axis=0)
        lp = score(candidates, logw, logq, log1q, logp0)
        order = np.argsort(lp)[::-1]
        final = candidates[order[:target_l]]
        final_lp = lp[order[:target_l]]
        p10 = float(np.exp(final_lp[-1]))
        need = required_draws(p10)
        screen_bound = (screening_bound(n_draws, p10, cutoff_count)
                        if screen_truncated else coverage_bound(n_draws, p10))
        if screen_bound <= DELTA or n_draws >= MAX_DRAWS:
            break
        target = min(MAX_DRAWS, max(n_draws * 2, need))

    logw2, logq2, log1q2 = quadrature_setup(B, psi, tau, REFINE_ORDER)
    final_lp2 = score(final, logw2, logq2, log1q2, logp0)
    order2 = np.argsort(final_lp2)[::-1]
    final, final_lp, final_lp2 = final[order2], final_lp[order2], final_lp2[order2]
    p10 = float(np.exp(final_lp2[-1]))
    bound = coverage_bound(n_draws, p10)
    screen_bound = (screening_bound(n_draws, p10, cutoff_count)
                    if screen_truncated else coverage_bound(n_draws, p10))

    conditioned = X[X[:, 0] == lead_state]
    empirical = Counter(packed_key(row) for row in conditioned)
    top = []
    for rank, (x, lp1, lp2) in enumerate(zip(final, final_lp, final_lp2), 1):
        key = packed_key(x)
        count = int(empirical[key])
        top.append({
            "rank": rank,
            "packed_little_endian_hex": key.hex(),
            "alternate_variant_ids": ids[x.astype(bool)].tolist(),
            "n_alternate": int(x.sum()),
            "q1_log_probability": float(lp2),
            "q1_probability": float(np.exp(lp2)),
            "quadrature_1024_vs_2048_log_difference": float(lp1 - lp2),
            "empirical_count_given_lead_state": count,
            "empirical_probability_given_lead_state": count / len(conditioned),
        })

    return {
        "dataset": dataset,
        "arm": arm,
        "lead_state": lead_state,
        "locus": input_path.stem,
        "n_haplotypes": int(len(X)),
        "n_variants": int(X.shape[1]),
        "n_haplotypes_with_conditioning_state": int(len(conditioned)),
        "conditioning_state_empirical_probability": float(len(conditioned) / len(X)),
        "search": {
            "requested_top_l": TOP_L,
            "returned_top_l": int(target_l),
            "reason_fewer_than_requested": (
                f"fewer than 10 possible haplotypes after fixing the lead to {lead_state}"
                if target_l < TOP_L else None
            ),
            "seed": int(locus_seed),
            "n_exact_conditional_draws": n_draws,
            "n_unique_discovered": len(counts),
            "candidate_pool_size": int(len(candidates)),
            "coverage_delta_target": DELTA,
            "coverage_union_bound_at_tenth_probability": bound,
            "screening_cutoff_count": cutoff_count,
            "screen_was_truncated": screen_truncated,
            "screening_union_bound_at_tenth_probability": screen_bound,
            "certified_top10_at_delta": bool(screen_bound <= DELTA),
            "required_draws_at_final_tenth_probability": required_draws(p10),
            "max_draws": MAX_DRAWS,
        },
        "top10": top,
        "input_sha256": sha256(input_path),
        "coefficient_sha256": sha256(coef_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True, choices=("armA", "armB"))
    ap.add_argument("--lead-state", required=True, type=int, choices=(0, 1))
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--coefficient-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    input_paths = sorted(args.input_dir.glob("*.npz"))
    if not input_paths:
        raise SystemExit(f"No NPZ inputs in {args.input_dir}")
    rows = []
    for i, input_path in enumerate(input_paths, 1):
        coef_path = args.coefficient_dir / input_path.name
        if not coef_path.exists():
            raise FileNotFoundError(coef_path)
        row = analyse_locus(
            args.dataset, args.arm, args.lead_state, input_path, coef_path
        )
        rows.append(row)
        print(f"[{i}/{len(input_paths)}] {input_path.stem}: "
              f"certified={row['search']['certified_top10_at_delta']} "
              f"draws={row['search']['n_exact_conditional_draws']} "
              f"p10={row['top10'][-1]['q1_probability']:.3g}", flush=True)

    payload = {
        "script": Path(__file__).name,
        "definition": {
            "target": f"top 10 under fitted q=1 law conditional on lead state {args.lead_state}",
            "empirical_probability": f"count among observed phased haplotypes with lead state {args.lead_state} divided by number with that state",
            "empirical_scope": "in-sample descriptive check; no smoothing",
            "search_certificate": "exp(-N*p10)/p10 union bound",
        },
        "config": {
            "top_l": TOP_L, "seed_base": SEED_BASE,
            "initial_draws": INITIAL_DRAWS, "maximum_draws": MAX_DRAWS,
            "batch": BATCH, "discovery_keep": DISCOVERY_KEEP,
            "quadrature_order": QUAD_ORDER, "refinement_order": REFINE_ORDER,
            "delta": DELTA,
        },
        "dataset": args.dataset,
        "arm": args.arm,
        "lead_state": args.lead_state,
        "n_loci": len(rows),
        "n_certified": sum(r["search"]["certified_top10_at_delta"] for r in rows),
        "per_locus": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
