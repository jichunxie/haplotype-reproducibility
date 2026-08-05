#!/usr/bin/env python3
"""Deterministic checks for the v77 rare/common frequency design."""
from __future__ import annotations

import numpy as np

import run_ranking_simulation_cell_v77 as v77
import run_ranking_simulation_cell_v76 as v76
import run_ranking_simulation_cell_v74 as v74


def exhaustive_top_ten(mixture) -> np.ndarray:
    configurations = v74.all_partner_configurations(4)
    log_probability = v76.configuration_log_probabilities(configurations, mixture)
    order = np.lexsort((np.arange(len(configurations)), -log_probability))
    return configurations[order[:10]]


def test_frequency_bounds_and_balance() -> None:
    for lead_class in v77.LEAD_FREQUENCY_CLASSES:
        for scenario in v77.SCENARIOS:
            for partners in v77.PARTNER_COUNTS:
                rng = v77.seed_rng(lead_class, scenario, partners, 0, 0)
                _, _, probability, variant_class = v77.make_truth(
                    rng, partners, scenario, lead_class
                )
                if lead_class == "rare":
                    assert 0.005 <= probability[0] < 0.05
                else:
                    assert 0.05 <= probability[0] <= 0.40
                partner_class = variant_class[1:]
                assert np.sum(partner_class == "rare") == partners // 2
                assert np.sum(partner_class == "common") == partners // 2
                rare = probability[1:][partner_class == "rare"]
                common = probability[1:][partner_class == "common"]
                assert np.all((rare >= 0.005) & (rare < 0.05))
                assert np.all((common >= 0.05) & (common <= 0.40))


def test_empty_empirical_mode_is_unavailable() -> None:
    x = np.zeros((20, 5), dtype=np.uint8)
    mode, ties, maximum, true_in_set = v77.empirical_mode(x, 1)
    assert mode is None
    assert ties == 0
    assert maximum == 0
    assert true_in_set is None


def test_truth_ranking_quadrature_stability() -> None:
    for lead_class in v77.LEAD_FREQUENCY_CLASSES:
        for scenario in v77.SCENARIOS:
            for replicate in range(5):
                rng = v77.seed_rng(lead_class, scenario, 4, replicate, 0)
                b, tau, probability, _ = v77.make_truth(
                    rng, 4, scenario, lead_class
                )
                for state in (0, 1):
                    top31 = exhaustive_top_ten(
                        v77.conditional_mixture(state, b, tau, probability, order=31)
                    )
                    top61 = exhaustive_top_ten(
                        v77.conditional_mixture(state, b, tau, probability, order=61)
                    )
                    if not np.array_equal(top31, top61):
                        raise AssertionError(
                            f"quadrature ranking changed: {lead_class}, {scenario}, "
                            f"replicate={replicate}, state={state}"
                        )


def test_exhaustive_check_accepts_tied_cutoff() -> None:
    log_base = np.asarray([0.0])
    log_zero = np.full((1, 4), np.log(0.5))
    log_one = np.full((1, 4), np.log(0.5))
    mixture = (log_base, log_zero, log_one)
    configurations = v74.all_partner_configurations(4)
    oracle = {
        "configurations": configurations[:10],
        "log_probabilities": v76.configuration_log_probabilities(
            configurations[:10], mixture
        ),
    }
    result = v77.exhaustive_check(4, mixture, oracle)
    assert result["valid"]
    assert not result["unique_top_ten"]
    assert result["tied_at_cutoff"] == 16


if __name__ == "__main__":
    test_frequency_bounds_and_balance()
    test_empty_empirical_mode_is_unavailable()
    test_truth_ranking_quadrature_stability()
    test_exhaustive_check_accepts_tied_cutoff()
    print("v77 frequency-design and quadrature checks passed")
