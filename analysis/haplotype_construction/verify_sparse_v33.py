#!/usr/bin/env python
"""VERIFY SPARSE v33 -- the top-10 haplotypes under the FPS (sparse PCA) model and under the dense
q=1 model, both scored under the TRUE R.  (PI request, 2026-07-27.)

Two jobs:

  (1) THE PI'S COMPARISON.  Enumerate the top 10 haplotypes under each of the two models and score
      the UNION of the two lists under the true pairwise R -- not under either model's own
      surrogate, which would be circular.  Report whether the two constructions pick the same
      haplotype, and by how much.

  (2) CLOSE THE COVERAGE GAP.  verify_mode_v31.py could only reach the 17 loci with k+1 <= 26,
      because the Genz integrator will not scale; the 1286-variant locus was never checked against
      the true R.  A GHK (Geweke-Hajivassiliou-Keane) simulator costs O(K^2) per draw after ONE
      Cholesky and reaches every locus.

GHK, and the three things that make it trustworthy here rather than merely fast:

  * ONE Cholesky per locus, shared by every candidate, with a FIXED variable ordering that does not
    depend on the candidate.  That makes the uniforms genuine COMMON RANDOM NUMBERS, so the
    difference between two candidates' estimates is far better determined than either level.
  * Everything in LOG space (`log_ndtr`, `ndtri_exp`, `logsumexp`).  A 1286-term product of
    sub-unit factors underflows a double outright; the log-space recursion does not.
  * The estimator is VALIDATED AGAINST GENZ at the 17 loci where both run, before any GHK-only
    number is used.  A new estimator that has not been checked against the trusted one is not
    evidence.

Honest statement of what GHK gives: it is UNBIASED for the probability, but log of it is biased
DOWNWARD by Jensen.  Rankings therefore use common-random-number differences, and the reported
uncertainty is the RELATIVE standard error of the probability, computed from the realised draw
spread rather than assumed.

Stochastic (GHK only) -- seeds are fixed, both are recorded in the output, and every locus is scored
under both so the estimator's own noise is reported rather than asserted.  Candidate generation,
the quadrature scoring and the FPS refit are all deterministic.

NO AlphaGenome calls.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp, ndtri_exp
from scipy.stats import multivariate_normal as mvn, norm
from pfa_floor import pfa_floor

WORK = "/hpc/group/xielab/jx42/CHAP/work"
V24 = f"{WORK}/ld_v24"
LEADS = f"{WORK}/leads_for_ldsens.json"
SPCA = f"{WORK}/sparsepca_v32.json"
OUT = f"{WORK}/verify_sparse_v33.json"

M_TOP = 10                # the PI asked for the top 10 under each model
PSI_FLOOR = 1e-4
MAX_IT = 400
TOL = 1e-9
MAX_EXACT = 26            # Genz ceiling, same as verify_mode_v31.py
SEEDS = (11, 2027)        # same two seeds as v31, so the noise figures are comparable
EIG_FLOORS = (1e-6, 1e-4)
N_GHK_SMALL = 20000       # draws when K <= 200
N_GHK_LARGE = 4000        # draws when K > 200  (cost is O(K^2 N))
K_GHK_SWITCH = 200
CLIP = 37.0               # |a| clip: Phi(-37) ~ 1e-300, so this bounds the recursion without
                          # changing any probability that is representable in double precision

NODES, WTS = np.polynomial.legendre.leggauss(1024)
F_GRID, LOG_W = NODES * 8.0, np.log(WTS * 8.0)


def pfa(S, q):
    """Identical to verify_mode_v31.py:63-81 and diag_sparsepca_v32.py, so the model scored here is
    byte-for-byte the model those scripts fitted."""
    return pfa_floor(S, q, PSI_FLOOR, MAX_IT, TOL)


def log_probs(cands, logq, log1mq):
    """log Pr(X = x, X_0 = 1) under a rank-1 model, by 1024-node Gauss-Legendre quadrature.
    The latent prior is the only thing in `lg`; q_0(f) is supplied by the candidate's own x_0 = 1
    term inside the sum, and must NOT appear twice (that squared it in an earlier script)."""
    lg = norm.logpdf(F_GRID)
    out = np.empty(len(cands))
    for c, x in enumerate(cands):
        xb = x.astype(bool)
        out[c] = logsumexp(LOG_W + lg + logq[xb, :].sum(0) + log1mq[~xb, :].sum(0))
    return out


def top_haplotypes(a, psi, tau, M):
    """Exact top-M under the rank-1 model with loading vector `a`.

    Works unchanged for the SPARSE model: an off-support variant has a_j = 0 and psi_j = 1, so its
    breakpoint tau_j/a_j is infinite and drops out of the cell enumeration, while its conditional
    probability collapses to Phi(-tau_j) = p_j -- the marginal.  The conditional mode then assigns
    it 1{0 > tau_j} = 1{p_j > 1/2}, which is exactly the off-support factorisation lemma, obtained
    here without special-casing anything."""
    K = len(tau)
    sd = np.sqrt(np.clip(psi, PSI_FLOOR, None))
    z = (np.outer(a, F_GRID) - tau[:, None]) / sd[:, None]
    logq, log1mq = log_ndtr(z), log_ndtr(-z)

    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(np.abs(a) > 1e-8, tau / a, np.inf)
    pts = np.unique(b[np.isfinite(b)])
    if len(pts) == 0:
        reps = np.array([0.0])
    else:
        reps = np.concatenate([[pts[0] - 1.0], 0.5 * (pts[:-1] + pts[1:]), [pts[-1] + 1.0]])
    modes = (a[None, :] * reps[:, None] > tau[None, :]).astype(np.uint8)
    modes[:, 0] = 1
    chain, first = np.unique(modes, axis=0, return_index=True)
    chain = chain[np.argsort(first)]
    assert len(chain) <= K + 1, f"{len(chain)} cells exceeds the K+1={K+1} bound"

    lp_chain = log_probs(chain, logq, log1mq)
    mode = chain[int(np.argmax(lp_chain))]
    nb = np.repeat(mode[None, :], K - 1, axis=0)
    nb[np.arange(K - 1), np.arange(1, K)] ^= 1          # never flip the lead
    pool = np.unique(np.vstack([chain, nb]), axis=0)
    lp = log_probs(pool, logq, log1mq)
    order = np.argsort(lp)[::-1]
    return pool[order][:M], lp[order][:M], len(chain)


def to_pd(R, eps):
    """Nearest correlation matrix with all eigenvalues >= eps; also returns the largest entrywise
    change so the perturbation is reported rather than hidden.  Copied from verify_mode_v31.py."""
    w, V = np.linalg.eigh(R)
    Rp = (V * np.clip(w, eps, None)) @ V.T
    d = np.sqrt(np.diag(Rp))
    Rp = Rp / np.outer(d, d)
    np.fill_diagonal(Rp, 1.0)
    return Rp, float(np.abs(Rp - R).max())


def genz_prob(x, tau, R, seed, maxpts_mult=50000):
    """Genz quasi-Monte-Carlo rectangle probability under the true R (verify_mode_v31.py:142-155)."""
    lo = np.where(x == 1, tau, -np.inf)
    hi = np.where(x == 1, np.inf, tau)
    np.random.seed(seed)
    return float(mvn.cdf(hi, mean=np.zeros(len(tau)), cov=R, lower_limit=lo,
                         maxpts=maxpts_mult * len(tau), abseps=1e-6, releps=1e-5))


def ghk(cands, tau, L, logU):
    """GHK estimator of Pr(Z_j > tau_j where x_j = 1, Z_j <= tau_j where x_j = 0) for Z ~ N(0, LL').

    `L` is ONE lower-triangular Cholesky factor, and `logU` ONE array of log-uniforms, both shared
    across every candidate -- that sharing is the common-random-number device.

    Returns, per candidate: log of the estimated probability, and the RELATIVE standard error of
    the probability estimate (not of its log)."""
    K, N = len(tau), logU.shape[0]
    diagL = np.diag(L)
    out_lp, out_rse = np.empty(len(cands)), np.empty(len(cands))
    for c, x in enumerate(cands):
        Mrun = np.zeros((K, N))
        acc = np.zeros(N)
        for j in range(K):
            aj = np.clip((tau[j] - Mrun[j]) / diagL[j], -CLIP, CLIP)
            if x[j] == 1:
                lp = log_ndtr(-aj)                       # Pr(eta_j > a_j)
                eta = -ndtri_exp(lp + logU[:, j])        # draw truncated to (a_j, inf)
            else:
                lp = log_ndtr(aj)                        # Pr(eta_j <= a_j)
                eta = ndtri_exp(lp + logU[:, j])         # draw truncated to (-inf, a_j]
            acc += lp
            eta = np.clip(eta, -CLIP - 1.0, CLIP + 1.0)
            if j + 1 < K:
                Mrun[j + 1:] += np.outer(L[j + 1:, j], eta)
        lmean = logsumexp(acc) - np.log(N)
        lm2 = logsumexp(2.0 * acc) - np.log(N)
        # Var = E[w^2] - E[w]^2, computed in logs; Jensen guarantees lm2 >= 2*lmean
        if lm2 > 2.0 * lmean + 1e-12:
            lvar = lm2 + np.log1p(-np.exp(2.0 * lmean - lm2))
            out_rse[c] = float(np.exp(0.5 * (lvar - np.log(N)) - lmean))
        else:
            out_rse[c] = 0.0
        out_lp[c] = float(lmean)
    return out_lp, out_rse


spca = {r["locus"]: r for r in json.load(open(SPCA))["per_locus"]}
leads = {r["locus"]: r for r in json.load(open(LEADS))}
rows = []
t_all = time.time()

for loc, rec in sorted(leads.items()):
    vc, af = f"{V24}/{loc}.vcor", f"{V24}/{loc}.afreq"
    if not (os.path.exists(vc) and os.path.exists(af)):
        continue
    lead = rec["lead"]
    p_ = pd.read_csv(vc, sep="\t")
    p_.columns = [c.lstrip("#") for c in p_.columns]
    if not len(p_):
        continue
    A_ = pd.read_csv(af, sep="\t")
    A_.columns = [c.lstrip("#") for c in A_.columns]
    fcol = "ALT_FREQS" if "ALT_FREQS" in A_.columns else "ALT_FREQ"
    freq = dict(zip(A_.ID.astype(str), A_[fcol].astype(float)))

    fa = np.where(p_.MAJ_A.values == p_.ID_A.str.split(":").str[3].values, 1, -1)
    fb = np.where(p_.MAJ_B.values == p_.ID_B.str.split(":").str[3].values, 1, -1)
    r_alt = p_.PHASED_R.values * fa * fb

    ids = [lead] + sorted(v for v in set(p_.ID_A) | set(p_.ID_B) if v != lead)
    ids = [v for v in ids if v in freq and 0.0 < freq[v] < 1.0]
    if len(ids) < 2 or ids[0] != lead:
        continue
    pos = {v: i for i, v in enumerate(ids)}
    K = len(ids)

    R = np.full((K, K), np.nan)
    np.fill_diagonal(R, 1.0)
    ia, ib = p_.ID_A.map(pos).values, p_.ID_B.map(pos).values
    keep = pd.notna(ia) & pd.notna(ib) & np.isfinite(r_alt)
    ia, ib = ia[keep].astype(int), ib[keep].astype(int)
    R[ia, ib] = r_alt[keep]
    R[ib, ia] = r_alt[keep]
    R[np.isnan(R)] = 0.0
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    tau = norm.ppf(1.0 - np.array([freq[v] for v in ids]))
    t0 = time.time()

    # ---- model A: dense q = 1, the construction Note 1 currently uses --------------------------
    Ld, psid = pfa(R, 1)
    a_dense = Ld[:, 0] * (1.0 if Ld[0, 0] >= 0 else -1.0)
    top_d, lp_d, nchain_d = top_haplotypes(a_dense, psid, tau, M_TOP)

    # ---- model B: the FPS support from job 50782139, refit exactly as v32 refits it ------------
    sp = spca[loc]["sparse"].get("1")
    if sp is None:
        act, R0 = np.arange(K), K
        support_source = "no lambda satisfied the selection rule; fell back to the full set"
    elif sp["support"] is not None:
        act, R0 = np.array(sp["support"], dtype=int), sp["R0"]
        support_source = "support list read from sparsepca_v32.json"
    else:
        # v32 stores the support only when R0 <= 64. Everywhere else it recorded R0, so the full
        # set is the only consistent reading -- ASSERT that rather than assume it.
        assert sp["R0"] == K, f"{loc}: R0={sp['R0']} < K={K} but v32 stored no support list"
        act, R0 = np.arange(K), K
        support_source = "R0 == K asserted against sparsepca_v32.json; support is the full set"
    Ls, psis = pfa(R[np.ix_(act, act)], 1)
    a_sparse = np.zeros(K)
    a_sparse[act] = Ls[:, 0]
    psi_sparse = np.ones(K)
    psi_sparse[act] = psis
    if a_sparse[0] < 0:
        a_sparse = -a_sparse
    top_s, lp_s, nchain_s = top_haplotypes(a_sparse, psi_sparse, tau, M_TOP)

    models_identical = bool(R0 == K and np.allclose(a_sparse, a_dense, atol=1e-10)
                            and np.allclose(psi_sparse, psid, atol=1e-10))

    # ---- union of the two top-10 lists, scored under the TRUE R --------------------------------
    union = np.unique(np.vstack([top_d, top_s]), axis=0)
    rank_of = lambda arr, y: next((i for i, z in enumerate(arr) if np.array_equal(z, y)), None)

    perm = np.argsort(-np.abs(tau))          # GHK ordering: most extreme thresholds first
    n_ghk = N_GHK_SMALL if K <= K_GHK_SWITCH else N_GHK_LARGE
    ghk_res, genz_res, pert = {}, {}, {}
    for eps in EIG_FLOORS:
        Rp, dmax = to_pd(R, eps)
        pert[str(eps)] = dmax
        Lchol = np.linalg.cholesky(Rp[np.ix_(perm, perm)])
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            logU = np.log(rng.random((n_ghk, K)))
            lp, rse = ghk([x[perm] for x in union], tau[perm], Lchol, logU)
            ghk_res[f"{eps}|{seed}"] = dict(logp=lp.tolist(), rel_se=rse.tolist())
        if K <= MAX_EXACT:
            for seed in SEEDS:
                genz_res[f"{eps}|{seed}"] = [genz_prob(x, tau, Rp, seed) for x in union]

    key = f"{EIG_FLOORS[0]}|{SEEDS[0]}"
    lp_true = np.array(ghk_res[key]["logp"])
    best = int(np.argmax(lp_true))
    ghk_argmax = union[best]

    # cross-estimator validation, on the loci where both run
    genz_check = None
    if genz_res:
        g = np.array(genz_res[key])
        with np.errstate(divide="ignore"):
            lg = np.log(np.clip(g, 1e-300, None))
        genz_check = dict(
            max_abs_log_diff=float(np.abs(lg - lp_true).max()),
            max_rel_diff=float(np.abs(np.exp(lp_true) - g).max() / max(g.max(), 1e-300)),
            same_argmax=bool(int(np.argmax(g)) == best),
            genz_top_prob=float(g.max()), ghk_top_prob=float(np.exp(lp_true.max())),
        )

    # seed-to-seed and floor-to-floor stability of the VERDICT, not just of the numbers
    verdicts = {k: int(np.argmax(v["logp"])) for k, v in ghk_res.items()}
    ordr = np.argsort(lp_true)[::-1]
    gaps = lp_true[ordr]
    # Is the winner ahead by more than the estimator's own noise?  Same logic v31 used for Genz.
    # SE(log P) ~ rel_se by the delta method; combining the top two as if INDEPENDENT overstates
    # the noise, because common random numbers correlate them positively -- so this is the
    # conservative direction and a large ratio is a safe verdict.
    rse_arr = np.array(ghk_res[key]["rel_se"])
    se_gap = float(np.hypot(rse_arr[ordr[0]], rse_arr[ordr[1]])) if len(gaps) > 1 else float("nan")
    gap_over_se = float((gaps[0] - gaps[1]) / se_gap) if len(gaps) > 1 and se_gap > 0 else None
    # realised seed-to-seed spread of the winner's log-probability: noise MEASURED, not assumed
    top_by_seed = [np.array(v["logp"])[ordr[0]] for v in ghk_res.values()]
    seed_spread = float(np.ptp(top_by_seed))

    rows.append(dict(
        locus=loc, K=K, n_partners=K - 1, R0=int(R0), support_source=support_source,
        models_identical=models_identical,
        n_chain_dense=int(nchain_d), n_chain_sparse=int(nchain_s),
        n_union=int(len(union)), n_ghk_draws=int(n_ghk),
        top1_dense=" ".join(map(str, top_d[0].tolist())) if K <= 64 else None,
        top1_sparse=" ".join(map(str, top_s[0].tolist())) if K <= 64 else None,
        top1_agree=bool(np.array_equal(top_d[0], top_s[0])),
        top1_hamming=int(np.abs(top_d[0].astype(int) - top_s[0].astype(int)).sum()),
        top10_jaccard=float(len({tuple(v) for v in top_d} & {tuple(v) for v in top_s})
                            / len({tuple(v) for v in top_d} | {tuple(v) for v in top_s})),
        model_logp_dense=lp_d.tolist(), model_logp_sparse=lp_s.tolist(),
        rank_of_sparse_top1_in_dense=rank_of(top_d, top_s[0]),
        rank_of_dense_top1_in_sparse=rank_of(top_s, top_d[0]),
        true_R_logp=lp_true.tolist(),
        true_R_rel_se=ghk_res[key]["rel_se"],
        true_R_argmax_is_dense_top1=bool(np.array_equal(ghk_argmax, top_d[0])),
        true_R_argmax_is_sparse_top1=bool(np.array_equal(ghk_argmax, top_s[0])),
        true_R_p_top=float(np.exp(gaps[0])),
        true_R_logp_gap=float(gaps[0] - gaps[1]) if len(gaps) > 1 else None,
        true_R_gap_se=se_gap, true_R_gap_over_se=gap_over_se,
        true_R_top_seed_spread=seed_spread,
        verdict_stable_across_seeds_and_floors=bool(len(set(verdicts.values())) == 1),
        verdicts=verdicts, matrix_perturbation=pert,
        genz_cross_check=genz_check,
        ghk_all=ghk_res if K <= 64 else None,
        seconds=round(time.time() - t0, 1),
    ))
    r = rows[-1]
    gz = "--" if genz_check is None else \
        f"agree={genz_check['same_argmax']} dlog={genz_check['max_abs_log_diff']:.1e}"
    print(f"{loc:24s} K={K:5d} R0={R0:5d} same_model={models_identical!s:5s} "
          f"top1_agree={r['top1_agree']!s:5s} ham={r['top1_hamming']:3d} "
          f"P_true={r['true_R_p_top']:.4f} relSE={max(r['true_R_rel_se']):.2e} "
          f"genz[{gz}] [{r['seconds']}s]", flush=True)

json.dump(dict(config=dict(
    M_top=M_TOP, seeds=list(SEEDS), eig_floors=list(EIG_FLOORS),
    n_ghk_small=N_GHK_SMALL, n_ghk_large=N_GHK_LARGE, k_ghk_switch=K_GHK_SWITCH,
    ghk_clip=CLIP, max_exact_genz=MAX_EXACT, psi_floor=PSI_FLOOR,
    ghk_ordering="variables permuted by descending |tau_j| (most extreme thresholds first); one "
                 "Cholesky per locus shared across all candidates, so the uniforms are common "
                 "random numbers and the RANKING is better determined than either level",
    ghk_bias="GHK is unbiased for the PROBABILITY; log of it is biased DOWNWARD by Jensen. The "
             "reported rel_se is the relative standard error of the probability estimate.",
    scope="argmax over the union of the two models' top-10 lists (chain cells + one-variant flips "
          "of each model's mode) -- local optimality plus chain dominance, NOT global optimality "
          "over 2^k haplotypes.",
    sparse_model="FPS active set from sparsepca_v32.json (job 50782139), loadings refit by the "
                 "same PFA call v32 uses, so the model scored here is the model v32 selected.",
), per_locus=rows), open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}   {len(rows)} loci   {time.time()-t_all:.1f}s")
