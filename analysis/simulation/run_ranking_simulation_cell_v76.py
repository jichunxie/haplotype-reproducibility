#!/usr/bin/env python3
"""Scalable known-truth ranking simulation for k = 4, 32, and 256 partners.

The v74 design enumerated all 2^k configurations and therefore could not be
extended beyond small k.  This revision represents each conditional law as the
finite product-Bernoulli mixture induced by fixed Gaussian quadrature and uses
a branch-and-bound MAP search.  The search has a valid remaining-coordinate
upper bound, returns the top ten configurations, and records whether the list
was certified before the node limit.  At k=4 its result is checked against full
enumeration in every replicate and lead state.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp

import run_ranking_simulation_cell_v74 as v74


PARTNER_COUNTS = (2**2, 2**5, 2**8)
TOP_L = 10
NODE_LIMIT = 2_000_000


def conditional_mixture(
    state: int,
    b: np.ndarray,
    tau: np.ndarray,
    probability: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log mixture weights and partner log Bernoulli probabilities."""
    factors, weights = v74.normal_quadrature(b.shape[1], order)
    psi = 1.0 - np.sum(b * b, axis=1)
    eta = (factors @ b.T - tau[None, :]) / np.sqrt(psi)[None, :]
    log_one = log_ndtr(eta)
    log_zero = log_ndtr(-eta)
    lead_log = log_one[:, 0] if state == 1 else log_zero[:, 0]
    # Normalize with the same fixed quadrature used in the numerator.  This
    # leaves the configuration ranking unchanged and makes the finite mixture
    # sum to one even when quadrature does not reproduce the analytic lead
    # margin exactly in a high-communality setting.
    unnormalized = np.log(weights) + lead_log
    log_base = unnormalized - logsumexp(unnormalized)
    return log_base, log_zero[:, 1:], log_one[:, 1:]


def configuration_log_probabilities(
    configurations: np.ndarray,
    mixture: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int = 64,
) -> np.ndarray:
    log_base, log_zero, log_one = mixture
    configurations = np.asarray(configurations, dtype=np.uint8)
    result = np.empty(len(configurations), dtype=float)
    for start in range(0, len(configurations), batch_size):
        stop = min(start + batch_size, len(configurations))
        config = configurations[start:stop].astype(float)
        component = config @ log_one.T + (1.0 - config) @ log_zero.T
        result[start:stop] = logsumexp(component + log_base[None, :], axis=1)
    return result


def unique_rows(rows: list[np.ndarray] | np.ndarray) -> np.ndarray:
    array = np.asarray(rows, dtype=np.uint8)
    if array.ndim == 1:
        array = array[None, :]
    return np.unique(array, axis=0)


def initial_candidates(
    mixture: tuple[np.ndarray, np.ndarray, np.ndarray],
    population_ld_sign: np.ndarray,
    top_seed_count: int = 20,
) -> np.ndarray:
    """Construct strong incumbents without using a finite simulated panel."""
    _, log_zero, log_one = mixture
    component_modes = unique_rows((log_one > log_zero).astype(np.uint8))
    seeds = unique_rows(np.vstack([component_modes, population_ld_sign[None, :]]))
    seed_logp = configuration_log_probabilities(seeds, mixture)
    leading = seeds[np.argsort(-seed_logp)[: min(top_seed_count, len(seeds))]]
    neighbors = [seeds]
    for candidate in leading:
        block = np.repeat(candidate[None, :], len(candidate), axis=0)
        block[np.arange(len(candidate)), np.arange(len(candidate))] ^= 1
        neighbors.append(block)
    return unique_rows(np.vstack(neighbors))


def local_improve(
    seeds: np.ndarray,
    mixture: tuple[np.ndarray, np.ndarray, np.ndarray],
    maximum_starts: int = 12,
) -> np.ndarray:
    """Greedily add one-flip local modes to strengthen branch-and-bound bounds."""
    seed_logp = configuration_log_probabilities(seeds, mixture)
    starts = seeds[np.argsort(-seed_logp)[: min(maximum_starts, len(seeds))]]
    completed = []
    for start in starts:
        current = start.copy()
        current_logp = float(configuration_log_probabilities(current[None, :], mixture)[0])
        for _ in range(len(current) + 1):
            neighbors = np.repeat(current[None, :], len(current), axis=0)
            neighbors[np.arange(len(current)), np.arange(len(current))] ^= 1
            logp = configuration_log_probabilities(neighbors, mixture)
            best = int(np.argmax(logp))
            if float(logp[best]) <= current_logp + 1e-13:
                break
            current = neighbors[best]
            current_logp = float(logp[best])
        completed.append(current)
    return unique_rows(np.vstack([seeds, np.asarray(completed, dtype=np.uint8)]))


def branch_order(
    mixture: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    log_base, log_zero, log_one = mixture
    weight = np.exp(log_base - logsumexp(log_base))
    modal_one = log_one > log_zero
    split = weight @ modal_one
    uncertainty = np.minimum(split, 1.0 - split)
    contrast = np.sqrt(weight @ ((log_one - log_zero) ** 2))
    return np.lexsort((-contrast, -uncertainty)).astype(int)


def branch_and_bound_top_l(
    mixture: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidates: np.ndarray,
    top_l: int = TOP_L,
    node_limit: int = NODE_LIMIT,
) -> dict:
    """Return a certified top-L list under the fixed-quadrature mixture."""
    log_base, log_zero, log_one = mixture
    k = log_zero.shape[1]
    order = branch_order(mixture)
    zero = log_zero[:, order]
    one = log_one[:, order]
    best_remaining = np.maximum(zero, one)
    future = np.zeros((len(log_base), k + 1), dtype=float)
    future[:, :k] = np.cumsum(best_remaining[:, ::-1], axis=1)[:, ::-1]

    score_by_key: dict[bytes, tuple[float, np.ndarray]] = {}

    def retain(config: np.ndarray, score: float) -> None:
        key = config.tobytes()
        previous = score_by_key.get(key)
        if previous is None or score > previous[0]:
            score_by_key[key] = (float(score), config.copy())

    candidate_scores = configuration_log_probabilities(candidates, mixture)
    for config, score in zip(candidates, candidate_scores):
        retain(config, float(score))

    def cutoff() -> float:
        scores = sorted((value[0] for value in score_by_key.values()), reverse=True)
        if len(scores) < top_l:
            return -math.inf
        return float(scores[top_l - 1])

    assignment = np.zeros(k, dtype=np.uint8)
    root_upper = float(logsumexp(log_base + future[:, 0]))
    stack: list[tuple[int, np.ndarray, np.ndarray, float]] = [
        (0, log_base.copy(), assignment, root_upper)
    ]
    nodes = 0
    maximum_pruned_upper = -math.inf
    while stack and nodes < node_limit:
        depth, partial, assigned, upper = stack.pop()
        threshold = cutoff()
        if upper <= threshold + 1e-13:
            maximum_pruned_upper = max(maximum_pruned_upper, upper)
            continue
        nodes += 1
        if depth == k:
            original = np.empty(k, dtype=np.uint8)
            original[order] = assigned
            retain(original, float(logsumexp(partial)))
            continue

        children = []
        for value, contribution in ((0, zero[:, depth]), (1, one[:, depth])):
            child_partial = partial + contribution
            child_upper = float(logsumexp(child_partial + future[:, depth + 1]))
            if child_upper > cutoff() + 1e-13:
                child_assignment = assigned.copy()
                child_assignment[depth] = value
                children.append((depth + 1, child_partial, child_assignment, child_upper))
            else:
                maximum_pruned_upper = max(maximum_pruned_upper, child_upper)
        children.sort(key=lambda item: item[3])
        stack.extend(children)

    ranked = sorted(score_by_key.values(), key=lambda item: (-item[0], item[1].tobytes()))
    ranked = ranked[:top_l]
    remaining_upper = max((item[3] for item in stack), default=-math.inf)
    return {
        "configurations": np.asarray([item[1] for item in ranked], dtype=np.uint8),
        "log_probabilities": np.asarray([item[0] for item in ranked], dtype=float),
        "certified": not stack,
        "nodes": nodes,
        "node_limit": node_limit,
        "largest_unresolved_upper": remaining_upper,
        "largest_pruned_upper": maximum_pruned_upper,
        "log_gap_best_to_unresolved": (
            math.inf if not stack else float(ranked[0][0] - remaining_upper)
        ),
        "branch_order": order.tolist(),
    }


def exhaustive_check(
    partners: int,
    mixture: tuple[np.ndarray, np.ndarray, np.ndarray],
    oracle: dict,
) -> None:
    if partners != 4:
        return
    configurations = v74.all_partner_configurations(partners)
    logp = configuration_log_probabilities(configurations, mixture)
    exact_order = np.lexsort((np.arange(len(configurations)), -logp))[:TOP_L]
    exact = configurations[exact_order]
    if not np.array_equal(exact, oracle["configurations"][: len(exact)]):
        raise AssertionError("branch-and-bound list disagrees with exhaustive k=4 enumeration")


def population_ld_background(b: np.ndarray, state: int) -> np.ndarray:
    signed = b[1:] @ b[0]
    return np.where(signed >= 0.0, state, 1 - state).astype(np.uint8)


def strategy_metrics(
    name: str,
    selected: np.ndarray,
    truth_oracle: dict,
    truth_mixture: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict:
    true_modes = truth_oracle["configurations"]
    true_logp = truth_oracle["log_probabilities"]
    selected_logp = float(configuration_log_probabilities(selected[None, :], truth_mixture)[0])
    matches = np.flatnonzero(np.all(true_modes == selected, axis=1))
    return {
        "strategy": name,
        "exact_true_mode": bool(np.array_equal(selected, true_modes[0])),
        "hamming_to_true_mode": int(np.sum(selected != true_modes[0])),
        "true_log_probability": selected_logp,
        "true_probability": float(np.exp(selected_logp)),
        "true_probability_regret": float(np.exp(true_logp[0]) - np.exp(selected_logp)),
        "true_rank_if_top10": None if not len(matches) else int(matches[0] + 1),
    }


def run_replicate(scenario: str, n: int, partners: int, replicate: int) -> dict:
    started = time.perf_counter()
    truth_rng = v74.seed_rng(scenario, partners, replicate, 0)
    b_true, tau_true, probability_true = v74.make_truth(
        truth_rng, partners + 1, scenario
    )
    data_rng = v74.seed_rng(scenario, partners, replicate, 1)
    x = v74.simulate(data_rng, 2000, b_true, tau_true)[:n]
    fit = v74.fit_q1(x)
    if not fit["reportable"]:
        raise RuntimeError(f"q1 fit did not pass reporting gates: kkt={fit['kkt']}")

    probability_fit = np.asarray(fit["p_tilde"], dtype=float)
    per_state = []
    for state in (0, 1):
        truth_order = 31
        fitted_order = 61
        truth_mixture = conditional_mixture(
            state, b_true, tau_true, probability_true, order=truth_order
        )
        fit_mixture = conditional_mixture(
            state,
            np.asarray(fit["B"], dtype=float),
            np.asarray(fit["tau"], dtype=float),
            probability_fit,
            order=fitted_order,
        )
        truth_candidates = initial_candidates(
            truth_mixture, population_ld_background(b_true, state)
        )
        truth_candidates = local_improve(truth_candidates, truth_mixture)
        truth_oracle = branch_and_bound_top_l(truth_mixture, truth_candidates)
        exhaustive_check(partners, truth_mixture, truth_oracle)
        if not truth_oracle["certified"]:
            raise RuntimeError(
                "truth MAP search did not certify: "
                f"gap={truth_oracle['log_gap_best_to_unresolved']}"
            )

        fitted_candidates = initial_candidates(
            fit_mixture,
            population_ld_background(np.asarray(fit["B"], dtype=float), state),
        )
        fitted_candidates = local_improve(fitted_candidates, fit_mixture)
        fitted_oracle = branch_and_bound_top_l(fit_mixture, fitted_candidates)
        exhaustive_check(partners, fit_mixture, fitted_oracle)
        if not fitted_oracle["certified"]:
            raise RuntimeError(
                "fitted MAP search did not certify: "
                f"gap={fitted_oracle['log_gap_best_to_unresolved']}"
            )

        empirical, tied_modes = v74.empirical_mode(x, state)
        sign_background = v74.ld_sign(x, state)
        fitted_mode = fitted_oracle["configurations"][0]
        strategy_rows = [
            strategy_metrics("fitted", fitted_mode, truth_oracle, truth_mixture),
            strategy_metrics("empirical_mode", empirical, truth_oracle, truth_mixture),
            strategy_metrics("ld_sign", sign_background, truth_oracle, truth_mixture),
        ]
        true_keys = {row.tobytes() for row in truth_oracle["configurations"]}
        fitted_keys = {row.tobytes() for row in fitted_oracle["configurations"]}
        overlap = len(true_keys.intersection(fitted_keys))
        per_state.append(
            {
                "lead_state": state,
                "n_conditioning_haplotypes": int(np.sum(x[:, 0] == state)),
                "n_tied_empirical_modes": tied_modes,
                "true_mode_probability": float(np.exp(truth_oracle["log_probabilities"][0])),
                "true_mode_log_probability": float(truth_oracle["log_probabilities"][0]),
                "fitted_mode_probability_under_fit": float(
                    np.exp(fitted_oracle["log_probabilities"][0])
                ),
                "true_top1_in_fitted_top10": bool(
                    truth_oracle["configurations"][0].tobytes() in fitted_keys
                ),
                "top10_overlap": overlap,
                "top10_overlap_fraction": float(overlap / TOP_L),
                "strategies": strategy_rows,
                "truth_oracle": {
                    key: value
                    for key, value in truth_oracle.items()
                    if key not in {"configurations", "log_probabilities", "branch_order"}
                },
                "fitted_oracle": {
                    key: value
                    for key, value in fitted_oracle.items()
                    if key not in {"configurations", "log_probabilities", "branch_order"}
                },
            }
        )

    empirical_correlation = np.corrcoef(x.T)
    fitted_correlation = v74.fitted_binary_correlation(
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
        "seed_master": v74.MASTER_SEED,
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
            "minimum_lead_partner_r2": float(np.min(np.corrcoef(x.T)[0, 1:] ** 2)),
            "median_lead_partner_r2": float(np.median(np.corrcoef(x.T)[0, 1:] ** 2)),
        },
        "per_state": per_state,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=tuple(v74.SCENARIOS), required=True)
    parser.add_argument(
        "--n-haplotypes", type=int, choices=(500, 1000, 2000), required=True
    )
    parser.add_argument("--partners", type=int, choices=PARTNER_COUNTS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=v74.N_REPLICATES)
    parser.add_argument("--replicate-start", type=int, default=0)
    args = parser.parse_args()
    if args.replicates < 1 or args.replicate_start < 0:
        parser.error("replicates must be positive and replicate-start must be nonnegative")
    replicate_stop = args.replicate_start + args.replicates
    if replicate_stop > v74.N_REPLICATES:
        parser.error(f"requested replicate range ends after {v74.N_REPLICATES}")
    cell = args.output_root / args.scenario / f"n{args.n_haplotypes}_k{args.partners}"
    cell.mkdir(parents=True, exist_ok=True)
    complete = 0
    for replicate in range(args.replicate_start, replicate_stop):
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
