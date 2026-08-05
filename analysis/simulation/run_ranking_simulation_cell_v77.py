#!/usr/bin/env python3
"""Frequency-stratified known-truth haplotype-ranking simulation.

This revision adds rare and common lead-variant strata and assigns exactly half
of the partners to rare and common ALT-frequency strata.  It preserves the v76
population-oracle and branch-and-bound machinery, but records when a finite
panel cannot construct an empirical-mode or LD-sign background.  In
particular, an ALT-conditioned empirical mode is unavailable when the panel
contains no ALT copy of the lead variant; no arbitrary fallback is substituted.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri
from scipy.sparse.linalg import svds

import run_ranking_simulation_cell_v74 as v74
import run_ranking_simulation_cell_v76 as v76


MASTER_SEED = 20260805
LEAD_FREQUENCY_CLASSES = {"rare": 0, "common": 1}
SCENARIOS = tuple(v74.SCENARIOS)
PARTNER_COUNTS = v76.PARTNER_COUNTS


def json_safe(value):
    """Convert NumPy scalars and non-finite search sentinels to strict JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def seed_rng(
    lead_frequency_class: str,
    scenario: str,
    partners: int,
    replicate: int,
    branch: int,
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                MASTER_SEED,
                LEAD_FREQUENCY_CLASSES[lead_frequency_class],
                v74.SCENARIOS[scenario],
                partners,
                replicate,
                branch,
            ]
        )
    )


def draw_alt_frequency(
    rng: np.random.Generator, frequency_class: str, size: int | None = None
) -> np.ndarray | float:
    if frequency_class == "rare":
        value = np.exp(rng.uniform(np.log(0.005), np.log(0.05), size=size))
    elif frequency_class == "common":
        value = rng.uniform(0.05, 0.40, size=size)
    else:
        raise ValueError(frequency_class)
    return float(value) if size is None else np.asarray(value, dtype=float)


def make_truth(
    rng: np.random.Generator,
    partners: int,
    scenario: str,
    lead_frequency_class: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if partners % 2:
        raise ValueError("the 50/50 partner-frequency design requires an even count")
    p = partners + 1
    probability = np.empty(p, dtype=float)
    probability[0] = draw_alt_frequency(rng, lead_frequency_class)

    partner_class = np.asarray(
        ["rare"] * (partners // 2) + ["common"] * (partners // 2), dtype="U6"
    )
    rng.shuffle(partner_class)
    rare = partner_class == "rare"
    probability[1:][rare] = draw_alt_frequency(rng, "rare", int(np.sum(rare)))
    probability[1:][~rare] = draw_alt_frequency(rng, "common", int(np.sum(~rare)))

    sign = rng.choice((-1.0, 1.0), size=p)
    sign[0] = 1.0
    if scenario == "balanced_q1":
        communality = rng.uniform(0.45, 0.75, size=p)
        b = sign[:, None] * np.sqrt(communality)[:, None]
    elif scenario in {"lead_ld_q1", "lead_ld_q2"}:
        communality = rng.uniform(0.90, 0.96, size=p)
        if scenario == "lead_ld_q1":
            b = sign[:, None] * np.sqrt(communality)[:, None]
        else:
            base = np.where(sign > 0.0, 0.0, math.pi)
            offset = np.zeros(p)
            offset[1:] = 0.30 * np.where(
                np.arange(1, p) % 2 == 0, 1.0, -1.0
            )
            offset[1:] += rng.normal(0.0, 0.04, size=p - 1)
            angle = base + offset
            b = np.sqrt(communality)[:, None] * np.column_stack(
                [np.cos(angle), np.sin(angle)]
            )
    else:
        raise ValueError(scenario)
    tau = ndtri(1.0 - probability)
    variant_class = np.concatenate(
        [np.asarray([lead_frequency_class], dtype="U6"), partner_class]
    )
    return b, tau, probability, variant_class


def lead_posterior_location_scale(
    loading_norm: float, threshold: float, psi: float, state: int
) -> tuple[float, float]:
    """Laplace proposal for the factor projection given the lead state."""
    y = 1.0 if state == 1 else -1.0
    root_psi = math.sqrt(psi)
    g = 0.0
    hessian = -1.0
    for _ in range(100):
        t = y * (loading_norm * g - threshold) / root_psi
        log_mills = -0.5 * t * t - 0.5 * math.log(2.0 * math.pi) - log_ndtr(t)
        mills = math.exp(log_mills)
        gradient = -g + y * loading_norm * mills / root_psi
        hessian = -1.0 - (loading_norm**2 / psi) * mills * (t + mills)
        step = gradient / hessian
        g_new = g - step
        if abs(g_new - g) <= 1e-12 * (1.0 + abs(g)):
            g = g_new
            break
        g = g_new
    return float(g), float(1.0 / math.sqrt(-hessian))


def conditional_mixture(
    state: int,
    b: np.ndarray,
    tau: np.ndarray,
    probability: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive finite mixture for the partner law conditional on the lead.

    The lead event changes only the factor projection parallel to its loading.
    Gaussian-Hermite nodes for that projection are centered and scaled at its
    conditional posterior mode; orthogonal factor directions retain their
    standard-normal quadrature.  Importance weights recover the original
    conditional law and are normalized on the finite grid.
    """
    del probability  # The finite-grid weights are normalized directly.
    b = np.asarray(b, dtype=float)
    tau = np.asarray(tau, dtype=float)
    q = b.shape[1]
    if q not in (1, 2):
        raise ValueError("v77 adaptive conditional quadrature supports q=1 or q=2")
    psi = 1.0 - np.sum(b * b, axis=1)
    loading_norm = float(np.linalg.norm(b[0]))
    if loading_norm <= 1e-12:
        factors, weights = v74.normal_quadrature(q, order)
        log_base = np.log(weights) - logsumexp(np.log(weights))
        eta = (factors @ b[1:].T - tau[None, 1:]) / np.sqrt(psi[1:])[None, :]
        return log_base, log_ndtr(-eta), log_ndtr(eta)
    direction = b[0] / loading_norm
    location, scale = lead_posterior_location_scale(
        loading_norm, float(tau[0]), float(psi[0]), state
    )

    standard_node, standard_weight = v74.normal_quadrature(1, order)
    standard_node = standard_node[:, 0]
    parallel = location + scale * standard_node
    y = 1.0 if state == 1 else -1.0
    lead_eta = y * (loading_norm * parallel - tau[0]) / math.sqrt(psi[0])
    log_importance = (
        np.log(standard_weight)
        - 0.5 * parallel**2
        + 0.5 * standard_node**2
        + math.log(scale)
        + log_ndtr(lead_eta)
    )

    if q == 1:
        factors = parallel[:, None] * direction[None, :]
        log_weight = log_importance
    else:
        perpendicular = np.asarray([-direction[1], direction[0]])
        orthogonal, orthogonal_weight = v74.normal_quadrature(1, order)
        orthogonal = orthogonal[:, 0]
        parallel_grid, orthogonal_grid = np.meshgrid(
            parallel, orthogonal, indexing="ij"
        )
        factors = (
            parallel_grid.ravel()[:, None] * direction[None, :]
            + orthogonal_grid.ravel()[:, None] * perpendicular[None, :]
        )
        log_weight = (
            log_importance[:, None] + np.log(orthogonal_weight)[None, :]
        ).ravel()
    log_base = log_weight - logsumexp(log_weight)
    eta = (factors @ b[1:].T - tau[None, 1:]) / np.sqrt(psi[1:])[None, :]
    return log_base, log_ndtr(-eta), log_ndtr(eta)


def robust_spectral_start_q1(x: np.ndarray) -> np.ndarray:
    """Return a finite start even when rare variants are monomorphic."""
    centered = x - x.mean(axis=0)
    standard_deviation = centered.std(axis=0, ddof=1)
    active = standard_deviation > 0.0
    common = np.zeros(x.shape[1], dtype=float)
    if int(np.sum(active)) >= 2:
        standardized = centered[:, active] / standard_deviation[active]
        v0 = np.linspace(-1.0, 1.0, min(standardized.shape))
        v0 /= np.linalg.norm(v0)
        _, singular_value, vt = svds(
            standardized, k=1, which="LM", return_singular_vectors=True, v0=v0
        )
        eigenvector = vt[0]
        orientation_index = 0 if active[0] else int(np.flatnonzero(active)[0])
        active_indices = np.flatnonzero(active)
        active_orientation = int(np.flatnonzero(active_indices == orientation_index)[0])
        if eigenvector[active_orientation] < 0:
            eigenvector = -eigenvector
        eigenvalue = float(singular_value[0] ** 2 / (len(x) - 1.0))
        common[active] = eigenvector * math.sqrt(max(eigenvalue - 1.0, 1e-4))
    elif np.any(active):
        common[active] = 0.20

    sign = np.sign(common)
    sign[sign == 0.0] = 1.0
    target = np.clip(np.abs(common), 0.05, math.sqrt(0.80))
    b = sign * target
    a = b / np.sqrt(1.0 - b * b)
    radius = math.sqrt(1.0 / v74.PSI_MIN - 1.0)
    return np.clip(a, -0.95 * radius, 0.95 * radius)


def fit_q1(x: np.ndarray) -> dict:
    original = v74.spectral_start_q1
    v74.spectral_start_q1 = robust_spectral_start_q1
    try:
        return v74.fit_q1(x)
    finally:
        v74.spectral_start_q1 = original


def empirical_mode(
    x: np.ndarray, state: int, true_mode: np.ndarray | None = None
) -> tuple[np.ndarray | None, int, int, bool | None]:
    conditional = x[x[:, 0] == state, 1:]
    if len(conditional) == 0:
        return None, 0, 0, None
    unique, counts = np.unique(conditional, axis=0, return_counts=True)
    maximum = int(counts.max())
    tied = np.flatnonzero(counts == maximum)
    true_in_set = (
        None
        if true_mode is None
        else bool(np.any(np.all(unique[tied] == true_mode[None, :], axis=1)))
    )
    return unique[tied[0]], int(len(tied)), maximum, true_in_set


def ld_sign(x: np.ndarray, state: int) -> tuple[np.ndarray | None, int]:
    centered = x.astype(float) - x.mean(axis=0)
    sum_squares = np.sum(centered * centered, axis=0)
    if sum_squares[0] <= 0.0:
        return None, x.shape[1] - 1
    undefined = sum_squares[1:] <= 0.0
    if np.any(undefined):
        return None, int(np.sum(undefined))
    numerator = centered[:, 0] @ centered[:, 1:]
    denominator = np.sqrt(sum_squares[0] * sum_squares[1:])
    correlation = numerator / denominator
    output = np.where(correlation >= 0.0, state, 1 - state).astype(np.uint8)
    return output, 0


def strategy_metrics(
    name: str,
    selected: np.ndarray | None,
    truth_oracle: dict,
    truth_mixture: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict:
    if selected is None:
        return {
            "strategy": name,
            "available": False,
            "exact_true_mode": None,
            "hamming_to_true_mode": None,
            "true_log_probability": None,
            "true_probability": None,
            "true_probability_regret": None,
            "true_rank_if_top10": None,
        }
    return {"available": True} | v76.strategy_metrics(
        name, selected, truth_oracle, truth_mixture
    )


def exhaustive_check(
    partners: int,
    mixture: tuple[np.ndarray, np.ndarray, np.ndarray],
    oracle: dict,
) -> dict:
    """Validate k=4 by score, allowing non-unique ordering at a tied cutoff."""
    if partners != 4:
        return {"checked": False, "valid": None, "unique_top_ten": None}
    configurations = v74.all_partner_configurations(partners)
    log_probability = v76.configuration_log_probabilities(configurations, mixture)
    returned = v76.configuration_log_probabilities(
        oracle["configurations"], mixture
    )
    if not np.allclose(returned, oracle["log_probabilities"], atol=2e-12, rtol=0.0):
        raise AssertionError("branch-and-bound returned inconsistent scores")
    cutoff = float(np.sort(log_probability)[-v76.TOP_L])
    tolerance = 2e-12
    if np.any(returned < cutoff - tolerance):
        raise AssertionError("branch-and-bound returned a configuration below the exact cutoff")
    strictly_above = int(np.sum(log_probability > cutoff + tolerance))
    tied_at_cutoff = int(np.sum(np.abs(log_probability - cutoff) <= tolerance))
    unique = strictly_above + tied_at_cutoff == v76.TOP_L
    return {
        "checked": True,
        "valid": True,
        "unique_top_ten": bool(unique),
        "strictly_above_cutoff": strictly_above,
        "tied_at_cutoff": tied_at_cutoff,
        "log_probability_cutoff": cutoff,
    }


def finite_correlation(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = x.std(axis=0, ddof=1) > 0.0
    correlation = np.full((x.shape[1], x.shape[1]), np.nan, dtype=float)
    if int(np.sum(active)) >= 2:
        correlation[np.ix_(active, active)] = np.corrcoef(x[:, active].T)
    elif int(np.sum(active)) == 1:
        index = int(np.flatnonzero(active)[0])
        correlation[index, index] = 1.0
    return correlation, active


def run_replicate(
    lead_frequency_class: str, scenario: str, n: int, partners: int, replicate: int
) -> dict:
    started = time.perf_counter()
    truth_rng = seed_rng(lead_frequency_class, scenario, partners, replicate, 0)
    b_true, tau_true, probability_true, variant_class = make_truth(
        truth_rng, partners, scenario, lead_frequency_class
    )
    data_rng = seed_rng(lead_frequency_class, scenario, partners, replicate, 1)
    x = v74.simulate(data_rng, 2000, b_true, tau_true)[:n]

    fit = None
    fit_error = None
    try:
        candidate_fit = fit_q1(x)
        if candidate_fit["reportable"]:
            fit = candidate_fit
        else:
            fit_error = f"fit did not pass reporting gates: kkt={candidate_fit['kkt']}"
    except Exception as error:
        fit_error = f"{type(error).__name__}: {error}"

    per_state = []
    for state in (0, 1):
        truth_mixture = conditional_mixture(
            state, b_true, tau_true, probability_true, order=31
        )
        truth_candidates = v76.initial_candidates(
            truth_mixture, v76.population_ld_background(b_true, state)
        )
        truth_candidates = v76.local_improve(truth_candidates, truth_mixture)
        truth_oracle = v76.branch_and_bound_top_l(truth_mixture, truth_candidates)
        truth_oracle["exhaustive_check"] = exhaustive_check(
            partners, truth_mixture, truth_oracle
        )
        if not truth_oracle["certified"]:
            raise RuntimeError("population top-ten search did not certify")

        fitted_oracle = None
        fitted_mode = None
        if fit is not None:
            probability_fit = np.asarray(fit["p_tilde"], dtype=float)
            fitted_mixture = conditional_mixture(
                state,
                np.asarray(fit["B"], dtype=float),
                np.asarray(fit["tau"], dtype=float),
                probability_fit,
                order=61,
            )
            fitted_candidates = v76.initial_candidates(
                fitted_mixture,
                v76.population_ld_background(np.asarray(fit["B"]), state),
            )
            fitted_candidates = v76.local_improve(fitted_candidates, fitted_mixture)
            fitted_oracle = v76.branch_and_bound_top_l(
                fitted_mixture, fitted_candidates
            )
            fitted_oracle["exhaustive_check"] = exhaustive_check(
                partners, fitted_mixture, fitted_oracle
            )
            if fitted_oracle["certified"]:
                fitted_mode = fitted_oracle["configurations"][0]
            else:
                fitted_oracle = None

        empirical, tied_modes, empirical_mode_count, true_in_empirical_mode_set = (
            empirical_mode(x, state, truth_oracle["configurations"][0])
        )
        sign_background, undefined_ld_sign_partners = ld_sign(x, state)
        strategy_rows = [
            strategy_metrics("fitted", fitted_mode, truth_oracle, truth_mixture),
            strategy_metrics("empirical_mode", empirical, truth_oracle, truth_mixture),
            strategy_metrics("ld_sign", sign_background, truth_oracle, truth_mixture),
        ]
        fitted_keys = (
            set()
            if fitted_oracle is None
            else {row.tobytes() for row in fitted_oracle["configurations"]}
        )
        true_keys = {row.tobytes() for row in truth_oracle["configurations"]}
        per_state.append(
            {
                "lead_state": state,
                "n_conditioning_haplotypes": int(np.sum(x[:, 0] == state)),
                "empirical_mode_available": empirical is not None,
                "empirical_mode_maximum_count": empirical_mode_count,
                "n_tied_empirical_modes": tied_modes,
                "true_mode_in_empirical_mode_set": true_in_empirical_mode_set,
                "ld_sign_available": sign_background is not None,
                "n_undefined_ld_sign_partners": undefined_ld_sign_partners,
                "true_mode_probability": float(
                    np.exp(truth_oracle["log_probabilities"][0])
                ),
                "true_mode_log_probability": float(
                    truth_oracle["log_probabilities"][0]
                ),
                "true_top1_in_fitted_top10": (
                    None
                    if fitted_oracle is None
                    else truth_oracle["configurations"][0].tobytes() in fitted_keys
                ),
                "top10_overlap_fraction": (
                    None
                    if fitted_oracle is None
                    else len(true_keys.intersection(fitted_keys)) / v76.TOP_L
                ),
                "strategies": strategy_rows,
                "truth_oracle": {
                    key: value
                    for key, value in truth_oracle.items()
                    if key not in {"configurations", "log_probabilities", "branch_order"}
                },
                "fitted_oracle": (
                    None
                    if fitted_oracle is None
                    else {
                        key: value
                        for key, value in fitted_oracle.items()
                        if key
                        not in {"configurations", "log_probabilities", "branch_order"}
                    }
                ),
            }
        )

    empirical_correlation, active = finite_correlation(x)
    fit_summary = {"available": fit is not None, "error": fit_error}
    if fit is not None:
        fitted_correlation = v74.fitted_binary_correlation(
            np.asarray(fit["B"], dtype=float),
            np.asarray(fit["psi"], dtype=float),
            np.asarray(fit["tau"], dtype=float),
        )
        off_diagonal = ~np.eye(partners + 1, dtype=bool)
        usable = off_diagonal & np.isfinite(empirical_correlation)
        residual_rmse = (
            float(
                np.sqrt(
                    np.mean(
                        (empirical_correlation[usable] - fitted_correlation[usable]) ** 2
                    )
                )
            )
            if np.any(usable)
            else None
        )
        fit_summary |= {
            "kkt": float(fit["kkt"]),
            "quadrature_order": int(fit["quadrature_order"]),
            "quadrature_refine_dev_per_haplotype": float(
                fit["quadrature_refine_dev_per_haplotype"]
            ),
            "parameter_refit_discrepancy": float(
                fit["parameter_refit_discrepancy"]
            ),
            "pva": float(np.mean(np.asarray(fit["B"]) ** 2)),
            "binary_correlation_residual_rmse": residual_rmse,
        }

    lead_r2 = empirical_correlation[0, 1:] ** 2
    finite_lead_r2 = lead_r2[np.isfinite(lead_r2)]
    return {
        "status": "complete",
        "lead_frequency_class": lead_frequency_class,
        "scenario": scenario,
        "n_haplotypes": n,
        "n_partners": partners,
        "replicate": replicate,
        "seed_master": MASTER_SEED,
        "nested_sample_design": True,
        "q_true": int(b_true.shape[1]),
        "q_fit": 1,
        "frequency_design": {
            "lead_alt_frequency": float(probability_true[0]),
            "n_rare_partners": int(np.sum(variant_class[1:] == "rare")),
            "n_common_partners": int(np.sum(variant_class[1:] == "common")),
            "minimum_partner_alt_frequency": float(np.min(probability_true[1:])),
            "median_partner_alt_frequency": float(np.median(probability_true[1:])),
            "maximum_partner_alt_frequency": float(np.max(probability_true[1:])),
            "n_monomorphic_panel_variants": int(np.sum(~active)),
        },
        "fit": fit_summary,
        "truth": {
            "mean_communality": float(np.mean(np.sum(b_true**2, axis=1))),
            "minimum_observed_lead_partner_r2": (
                None if not len(finite_lead_r2) else float(np.min(finite_lead_r2))
            ),
            "median_observed_lead_partner_r2": (
                None if not len(finite_lead_r2) else float(np.median(finite_lead_r2))
            ),
        },
        "per_state": per_state,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lead-frequency-class",
        choices=tuple(LEAD_FREQUENCY_CLASSES),
        required=True,
    )
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument(
        "--n-haplotypes", type=int, choices=(500, 1000, 2000), required=True
    )
    parser.add_argument("--partners", type=int, choices=PARTNER_COUNTS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=v74.N_REPLICATES)
    parser.add_argument("--replicate-start", type=int, default=0)
    args = parser.parse_args()
    replicate_stop = args.replicate_start + args.replicates
    if args.replicates < 1 or args.replicate_start < 0:
        parser.error("replicates must be positive and replicate-start nonnegative")
    if replicate_stop > v74.N_REPLICATES:
        parser.error(f"requested replicate range ends after {v74.N_REPLICATES}")

    cell = (
        args.output_root
        / args.lead_frequency_class
        / args.scenario
        / f"n{args.n_haplotypes}_k{args.partners}"
    )
    cell.mkdir(parents=True, exist_ok=True)
    complete = 0
    for replicate in range(args.replicate_start, replicate_stop):
        path = cell / f"replicate_{replicate:03d}.json"
        if path.exists() and json.loads(path.read_text()).get("status") == "complete":
            complete += 1
            continue
        try:
            result = run_replicate(
                args.lead_frequency_class,
                args.scenario,
                args.n_haplotypes,
                args.partners,
                replicate,
            )
        except Exception as error:
            result = {
                "status": "failed",
                "lead_frequency_class": args.lead_frequency_class,
                "scenario": args.scenario,
                "n_haplotypes": args.n_haplotypes,
                "n_partners": args.partners,
                "replicate": replicate,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        temporary = path.with_suffix(".json.tmp")
        result = json_safe(result)
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary.replace(path)
        if result["status"] == "complete":
            complete += 1
    print(
        json.dumps(
            {
                "lead_frequency_class": args.lead_frequency_class,
                "scenario": args.scenario,
                "n_haplotypes": args.n_haplotypes,
                "n_partners": args.partners,
                "complete": complete,
                "requested": args.replicates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
