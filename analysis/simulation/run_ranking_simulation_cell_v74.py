#!/usr/bin/env python3
"""Known-truth finite-panel validation of conditional haplotype ranking.

One invocation runs all replicates for a (scenario, n, partner-count) cell.
The production q=1 fixed-margin estimator is used in every cell.  The
lead_ld_q2 scenario is deliberately misspecified and requires only q=2 data
generation, not q=2 fitting.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr


Q1_PATH = Path("/hpc/group/xielab/jx42/CHAP/work/rosmap_factor_v51/code")
if Q1_PATH.exists():
    sys.path.insert(0, str(Q1_PATH))
import probit_fixed_adaptive_v51 as q1_estimator


MASTER_SEED = 20260804
PSI_MIN = 0.01
N_REPLICATES = 30
SCENARIOS = {"lead_ld_q1": 0, "balanced_q1": 1, "lead_ld_q2": 2}


def seed_rng(scenario: str, partners: int, replicate: int, branch: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [MASTER_SEED, SCENARIOS[scenario], partners, replicate, branch]
        )
    )


def make_truth(
    rng: np.random.Generator, p: int, scenario: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scenario == "balanced_q1":
        probability = rng.uniform(0.10, 0.90, size=p)
        communality = rng.uniform(0.45, 0.75, size=p)
        sign = rng.choice((-1.0, 1.0), size=p)
        sign[0] = 1.0
        b = sign[:, None] * np.sqrt(communality)[:, None]
    elif scenario in {"lead_ld_q1", "lead_ld_q2"}:
        lead_probability = rng.uniform(0.20, 0.40)
        sign = rng.choice((-1.0, 1.0), size=p)
        sign[0] = 1.0
        center = np.where(sign > 0, lead_probability, 1.0 - lead_probability)
        probability = np.clip(center + rng.normal(0.0, 0.02, size=p), 0.08, 0.92)
        probability[0] = lead_probability
        communality = rng.uniform(0.90, 0.96, size=p)
        if scenario == "lead_ld_q1":
            b = sign[:, None] * np.sqrt(communality)[:, None]
        else:
            base = np.where(sign > 0.0, 0.0, math.pi)
            offset = np.zeros(p)
            if p > 1:
                offset[1:] = 0.30 * np.where(np.arange(1, p) % 2 == 0, 1.0, -1.0)
                offset[1:] += rng.normal(0.0, 0.04, size=p - 1)
            angle = base + offset
            b = np.sqrt(communality)[:, None] * np.column_stack(
                [np.cos(angle), np.sin(angle)]
            )
    else:
        raise ValueError(scenario)
    tau = ndtri(1.0 - probability)
    return b, tau, probability


def simulate(
    rng: np.random.Generator, n: int, b: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    psi = 1.0 - np.sum(b * b, axis=1)
    factor = rng.normal(size=(n, b.shape[1]))
    error = rng.normal(size=(n, len(b)))
    return (factor @ b.T + error * np.sqrt(psi)[None, :] > tau).astype(np.uint8)


def spectral_start_q1(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0)
    standard_deviation = centered.std(axis=0, ddof=1)
    if np.any(standard_deviation <= 0):
        raise ValueError("simulation produced a monomorphic fitted column")
    standardized = centered / standard_deviation
    v0 = np.linspace(-1.0, 1.0, min(standardized.shape))
    v0 /= np.linalg.norm(v0)
    _, singular_value, vt = svds(
        standardized, k=1, which="LM", return_singular_vectors=True, v0=v0
    )
    eigenvector = vt[0]
    if eigenvector[0] < 0:
        eigenvector = -eigenvector
    eigenvalue = float(singular_value[0] ** 2 / (len(x) - 1.0))
    common = eigenvector * math.sqrt(max(eigenvalue - 1.0, 1e-4))
    target = np.clip(np.abs(common), 0.20, math.sqrt(0.80))
    b = np.sign(common)
    b[b == 0] = 1
    b *= target
    a = b / np.sqrt(1.0 - b * b)
    radius = math.sqrt(1.0 / PSI_MIN - 1.0)
    return np.clip(a, -0.95 * radius, 0.95 * radius)


def fit_q1(x: np.ndarray) -> dict:
    initial = spectral_start_q1(x.astype(float))
    first = q1_estimator.fit_fixed_margin_q1(
        x.astype(float),
        init=initial,
        psi_min=PSI_MIN,
        initial_order=64,
        max_order=1024,
        eps_ll_per_haplotype=1e-3,
        eps_theta=1e-3,
        kkt_tol=1e-5,
        relative_tol=1e-10,
        max_iter_per_order=500,
    )
    if first["reportable"]:
        return first
    # A small fraction of otherwise stable q=1 cells stop immediately above
    # the projected-KKT gate.  Retry deterministically from the refined first
    # fit with tighter optimizer tolerances; no result is reported unless the
    # unchanged production gate is then passed.
    radius = math.sqrt(1.0 / PSI_MIN - 1.0)
    candidates = [first]
    refined = np.asarray(first["a"], dtype=float)[:, 0]
    for scale in (1.0, 0.98, 1.02):
        retry = q1_estimator.fit_fixed_margin_q1(
            x.astype(float),
            init=np.clip(scale * refined, -0.999 * radius, 0.999 * radius),
            psi_min=PSI_MIN,
            initial_order=64,
            max_order=2048,
            eps_ll_per_haplotype=1e-3,
            eps_theta=1e-3,
            kkt_tol=1e-5,
            relative_tol=1e-12,
            max_iter_per_order=2000,
        )
        candidates.append(retry)
        if retry["reportable"]:
            return retry
    return max(candidates, key=lambda fit: fit["loglik"])


def all_partner_configurations(partners: int) -> np.ndarray:
    values = np.arange(2**partners, dtype=np.uint64)[:, None]
    shifts = np.arange(partners, dtype=np.uint64)[None, :]
    return ((values >> shifts) & 1).astype(np.uint8)


def normal_quadrature(q: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes = np.sqrt(2.0) * nodes
    weights = weights / np.sqrt(np.pi)
    if q == 1:
        return nodes[:, None], weights
    mesh = np.meshgrid(*([nodes] * q), indexing="ij")
    points = np.column_stack([part.ravel() for part in mesh])
    weight_mesh = np.meshgrid(*([weights] * q), indexing="ij")
    combined = np.prod(np.stack(weight_mesh, axis=-1), axis=-1).ravel()
    return points, combined


def conditional_distribution(
    partner_configurations: np.ndarray,
    state: int,
    b: np.ndarray,
    tau: np.ndarray,
    probability: np.ndarray,
    order: int,
) -> np.ndarray:
    factors, weights = normal_quadrature(b.shape[1], order)
    eta = (factors @ b.T - tau[None, :]) / np.sqrt(
        1.0 - np.sum(b * b, axis=1)
    )[None, :]
    log_one = log_ndtr(eta)
    log_zero = log_ndtr(-eta)
    lead_log = log_one[:, 0] if state == 1 else log_zero[:, 0]
    log_weight = np.log(weights) + lead_log
    result = np.empty(len(partner_configurations), dtype=float)
    for start in range(0, len(partner_configurations), 256):
        stop = min(start + 256, len(partner_configurations))
        config = partner_configurations[start:stop].astype(float)
        log_partner = config @ log_one[:, 1:].T + (1.0 - config) @ log_zero[:, 1:].T
        denominator = probability[0] if state == 1 else 1.0 - probability[0]
        result[start:stop] = np.exp(logsumexp(log_partner + log_weight, axis=1) - np.log(denominator))
    result /= result.sum()
    return result


def empirical_mode(x: np.ndarray, state: int) -> tuple[np.ndarray, int]:
    conditional = x[x[:, 0] == state, 1:]
    unique, counts = np.unique(conditional, axis=0, return_counts=True)
    maximum = counts.max()
    tied = np.flatnonzero(counts == maximum)
    return unique[tied[0]], int(len(tied))


def ld_sign(x: np.ndarray, state: int) -> np.ndarray:
    centered = x.astype(float) - x.mean(axis=0)
    lead = centered[:, 0]
    numerator = lead @ centered[:, 1:]
    denominator = np.sqrt((lead @ lead) * np.sum(centered[:, 1:] ** 2, axis=0))
    correlation = numerator / denominator
    output = np.where(correlation >= 0, state, 1 - state).astype(np.uint8)
    return output


def fitted_binary_correlation(
    b: np.ndarray, psi: np.ndarray, tau: np.ndarray, order: int = 128
) -> np.ndarray:
    factors, weights = normal_quadrature(1, order)
    eta = (factors @ b.T - tau[None, :]) / np.sqrt(psi)[None, :]
    q = np.exp(log_ndtr(eta))
    margin = weights @ q
    joint = (q * weights[:, None]).T @ q
    variance = margin * (1.0 - margin)
    denominator = np.sqrt(variance[:, None] * variance[None, :])
    correlation = (joint - margin[:, None] * margin[None, :]) / denominator
    np.fill_diagonal(correlation, 1.0)
    return correlation


def choose_metrics(
    name: str,
    selected: np.ndarray,
    true_probability: np.ndarray,
    configurations: np.ndarray,
    true_order: np.ndarray,
) -> dict:
    matches = np.flatnonzero(np.all(configurations == selected, axis=1))
    if len(matches) != 1:
        raise ValueError(f"{name} did not match exactly one enumerated configuration")
    index = int(matches[0])
    best_probability = float(true_probability[true_order[0]])
    selected_probability = float(true_probability[index])
    true_mode = configurations[true_order[0]]
    return {
        "strategy": name,
        "selected_index": index,
        "exact_true_mode": bool(index == int(true_order[0])),
        "hamming_to_true_mode": int(np.sum(selected != true_mode)),
        "true_probability": selected_probability,
        "true_probability_regret": best_probability - selected_probability,
        "true_rank": int(np.flatnonzero(true_order == index)[0] + 1),
    }


def run_replicate(scenario: str, n: int, partners: int, replicate: int) -> dict:
    started = time.perf_counter()
    # Truth and the 2,000-row master panel do not depend on requested sample
    # size.  The n=500 and n=1,000 datasets are nested prefixes of n=2,000,
    # making sample-size contrasts paired on both parameters and haplotypes.
    truth_rng = seed_rng(scenario, partners, replicate, 0)
    b_true, tau_true, probability_true = make_truth(truth_rng, partners + 1, scenario)
    data_rng = seed_rng(scenario, partners, replicate, 1)
    x = simulate(data_rng, 2000, b_true, tau_true)[:n]
    fit = fit_q1(x)
    if not fit["reportable"]:
        raise RuntimeError(f"q1 fit did not pass reporting gates: kkt={fit['kkt']}")

    configurations = all_partner_configurations(partners)
    probability_fit = np.asarray(fit["p_tilde"], dtype=float)
    per_state = []
    for state in (0, 1):
        true_distribution = conditional_distribution(
            configurations, state, b_true, tau_true, probability_true, order=31
        )
        fitted_distribution = conditional_distribution(
            configurations,
            state,
            np.asarray(fit["B"], dtype=float),
            np.asarray(fit["tau"], dtype=float),
            probability_fit,
            order=61,
        )
        true_order = np.lexsort((np.arange(len(configurations)), -true_distribution))
        fitted_order = np.lexsort((np.arange(len(configurations)), -fitted_distribution))
        empirical, tied_modes = empirical_mode(x, state)
        sign_background = ld_sign(x, state)
        strategy_rows = [
            choose_metrics(
                "fitted", configurations[fitted_order[0]], true_distribution, configurations, true_order
            ),
            choose_metrics("empirical_mode", empirical, true_distribution, configurations, true_order),
            choose_metrics("ld_sign", sign_background, true_distribution, configurations, true_order),
        ]
        top_l = min(10, len(configurations))
        overlap = len(set(true_order[:top_l]).intersection(fitted_order[:top_l]))
        per_state.append(
            {
                "lead_state": state,
                "n_conditioning_haplotypes": int(np.sum(x[:, 0] == state)),
                "n_tied_empirical_modes": tied_modes,
                "true_mode_probability": float(true_distribution[true_order[0]]),
                "fitted_mode_probability_under_fit": float(
                    fitted_distribution[fitted_order[0]]
                ),
                "true_top1_in_fitted_top10": bool(int(true_order[0]) in fitted_order[:top_l]),
                "top10_overlap": overlap,
                "top10_overlap_fraction": float(overlap / top_l),
                "probability_rmse_all_configurations": float(
                    np.sqrt(np.mean((fitted_distribution - true_distribution) ** 2))
                ),
                "probability_spearman_all_configurations": float(
                    spearmanr(fitted_distribution, true_distribution).statistic
                ),
                "strategies": strategy_rows,
            }
        )

    empirical_correlation = np.corrcoef(x.T)
    fitted_correlation = fitted_binary_correlation(
        np.asarray(fit["B"], dtype=float),
        np.asarray(fit["psi"], dtype=float),
        np.asarray(fit["tau"], dtype=float),
    )
    off_diagonal = ~np.eye(partners + 1, dtype=bool)
    residual = empirical_correlation[off_diagonal] - fitted_correlation[off_diagonal]
    return {
        "status": "complete",
        "scenario": scenario,
        "n_haplotypes": n,
        "n_partners": partners,
        "replicate": replicate,
        "seed_master": MASTER_SEED,
        "nested_sample_design": True,
        "q_true": int(b_true.shape[1]),
        "q_fit": 1,
        "fit": {
            "kkt": float(fit["kkt"]),
            "quadrature_order": int(fit["quadrature_order"]),
            "quadrature_refine_dev_per_haplotype": float(
                fit["quadrature_refine_dev_per_haplotype"]
            ),
            "parameter_refit_discrepancy": float(fit["parameter_refit_discrepancy"]),
            "pva": float(np.mean(np.asarray(fit["B"]) ** 2)),
            "binary_correlation_residual_rmse": float(np.sqrt(np.mean(residual**2))),
        },
        "truth": {
            "mean_communality": float(np.mean(np.sum(b_true**2, axis=1))),
            "minimum_lead_partner_r2": float(
                np.min(np.corrcoef(x.T)[0, 1:] ** 2)
            ),
            "median_lead_partner_r2": float(
                np.median(np.corrcoef(x.T)[0, 1:] ** 2)
            ),
        },
        "per_state": per_state,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    parser.add_argument("--n-haplotypes", type=int, choices=(500, 1000, 2000), required=True)
    parser.add_argument("--partners", type=int, choices=(4, 8, 12), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=N_REPLICATES)
    args = parser.parse_args()
    cell = args.output_root / args.scenario / f"n{args.n_haplotypes}_k{args.partners}"
    cell.mkdir(parents=True, exist_ok=True)
    complete = 0
    for replicate in range(args.replicates):
        path = cell / f"replicate_{replicate:03d}.json"
        if path.exists() and json.loads(path.read_text()).get("status") == "complete":
            complete += 1
            continue
        try:
            result = run_replicate(
                args.scenario, args.n_haplotypes, args.partners, replicate
            )
        except Exception as error:
            result = {
                "status": "failed",
                "scenario": args.scenario,
                "n_haplotypes": args.n_haplotypes,
                "n_partners": args.partners,
                "replicate": replicate,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(path)
        print(json.dumps({"path": str(path), "status": result["status"]}), flush=True)
        if result["status"] != "complete":
            raise RuntimeError(f"replicate failed: {path}")
        complete += 1
    print(json.dumps({"cell": str(cell), "complete": complete}, indent=2))


if __name__ == "__main__":
    main()
