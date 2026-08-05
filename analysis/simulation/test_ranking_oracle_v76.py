#!/usr/bin/env python3
"""Independent checks for the scalable top-L oracle used in revision v76."""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np


# The production estimator is installed on the DCC, but the oracle tests do
# not fit a model.  A stub keeps this local test independent of that install.
sys.modules.setdefault("probit_fixed_adaptive_v51", types.ModuleType("probit_fixed_adaptive_v51"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_ranking_simulation_cell_v74 as v74  # noqa: E402
import run_ranking_simulation_cell_v76 as v76  # noqa: E402


def exact_top_l(
    mixture: tuple[np.ndarray, np.ndarray, np.ndarray], partners: int
) -> tuple[np.ndarray, np.ndarray]:
    configurations = v74.all_partner_configurations(partners)
    logp = v76.configuration_log_probabilities(configurations, mixture)
    order = np.lexsort((np.arange(len(configurations)), -logp))[: v76.TOP_L]
    return configurations[order], logp[order]


def assert_oracle_exact(
    mixture: tuple[np.ndarray, np.ndarray, np.ndarray],
    partners: int,
    population_ld_sign: np.ndarray,
) -> dict:
    candidates = v76.initial_candidates(mixture, population_ld_sign)
    candidates = v76.local_improve(candidates, mixture)
    oracle = v76.branch_and_bound_top_l(mixture, candidates)
    expected_configurations, expected_logp = exact_top_l(mixture, partners)
    if not oracle["certified"]:
        raise AssertionError(f"k={partners}: oracle did not certify")
    if not np.array_equal(oracle["configurations"], expected_configurations):
        raise AssertionError(f"k={partners}: top-{v76.TOP_L} configurations differ")
    np.testing.assert_allclose(oracle["log_probabilities"], expected_logp, atol=2e-12)
    return oracle


def test_conditional_probability_identity() -> None:
    for scenario in v74.SCENARIOS:
        rng = v74.seed_rng(scenario, 4, 0, 0)
        b, tau, probability = v74.make_truth(rng, 5, scenario)
        configurations = v74.all_partner_configurations(4)
        for state in (0, 1):
            mixture = v76.conditional_mixture(state, b, tau, probability, order=31)
            new = np.exp(v76.configuration_log_probabilities(configurations, mixture))
            old = v74.conditional_distribution(
                configurations, state, b, tau, probability, order=31
            )
            np.testing.assert_allclose(new.sum(), 1.0, atol=5e-10)
            np.testing.assert_allclose(new, old, atol=5e-10)


def test_exact_top_l_small_dimensions() -> None:
    for scenario in v74.SCENARIOS:
        for partners in (4, 8, 12):
            rng = v74.seed_rng(scenario, partners, 2, 0)
            b, tau, probability = v74.make_truth(rng, partners + 1, scenario)
            for state in (0, 1):
                mixture = v76.conditional_mixture(
                    state, b, tau, probability, order=31
                )
                oracle = assert_oracle_exact(
                    mixture, partners, v76.population_ld_background(b, state)
                )
                if oracle["largest_pruned_upper"] > oracle["log_probabilities"][-1] + 1e-12:
                    raise AssertionError("pruned branch exceeded the certified top-L cutoff")


def test_k32_certification() -> None:
    timings = []
    for scenario in v74.SCENARIOS:
        rng = v74.seed_rng(scenario, 32, 1, 0)
        b, tau, probability = v74.make_truth(rng, 33, scenario)
        for state in (0, 1):
            mixture = v76.conditional_mixture(state, b, tau, probability, order=31)
            candidates = v76.initial_candidates(
                mixture, v76.population_ld_background(b, state)
            )
            candidates = v76.local_improve(candidates, mixture)
            started = time.perf_counter()
            oracle = v76.branch_and_bound_top_l(mixture, candidates)
            timings.append(time.perf_counter() - started)
            if not oracle["certified"]:
                raise AssertionError(
                    f"k=32 {scenario} state={state}: oracle did not certify"
                )
    print(
        "k=32 oracle certification seconds: "
        f"median={np.median(timings):.3f}, max={np.max(timings):.3f}"
    )


def test_k256_certification() -> None:
    timings = []
    for scenario in v74.SCENARIOS:
        rng = v74.seed_rng(scenario, 256, 1, 0)
        b, tau, probability = v74.make_truth(rng, 257, scenario)
        for state in (0, 1):
            mixture = v76.conditional_mixture(state, b, tau, probability, order=31)
            candidates = v76.initial_candidates(
                mixture, v76.population_ld_background(b, state)
            )
            candidates = v76.local_improve(candidates, mixture)
            started = time.perf_counter()
            oracle = v76.branch_and_bound_top_l(mixture, candidates)
            timings.append(time.perf_counter() - started)
            if not oracle["certified"]:
                raise AssertionError(
                    f"k=256 {scenario} state={state}: oracle did not certify"
                )
    print(
        "k=256 oracle certification seconds: "
        f"median={np.median(timings):.3f}, max={np.max(timings):.3f}"
    )


def main() -> None:
    test_conditional_probability_identity()
    test_exact_top_l_small_dimensions()
    test_k32_certification()
    test_k256_certification()
    print("ranking oracle v76 checks passed")


if __name__ == "__main__":
    main()
