#!/usr/bin/env python3
"""Small deterministic tests for build_public_baseline_haplotypes_v73.py."""
import numpy as np

from build_public_baseline_haplotypes_v73 import (
    empirical_conditional_mode,
    ld_sign_background,
    packed_hex,
    signed_correlations,
)


def test_empirical_mode_and_tie_rule() -> None:
    x = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    mode0, audit0 = empirical_conditional_mode(x, 0, 0)
    assert mode0.tolist() == [0, 0, 1]
    assert audit0["n_tied_modes"] == 2
    mode1, audit1 = empirical_conditional_mode(x, 0, 1)
    assert mode1.tolist() == [1, 1, 0]
    assert audit1["mode_count"] == 2


def test_ld_sign() -> None:
    x = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )
    correlations = signed_correlations(x, 0)
    assert correlations[1] > 0 and correlations[2] < 0
    background0, audit0 = ld_sign_background(x, 0, 0)
    background1, audit1 = ld_sign_background(x, 0, 1)
    assert background0.tolist() == [0, 0, 1]
    assert background1.tolist() == [1, 1, 0]
    assert audit0["n_zero_signed_r_fallbacks"] == 0
    assert audit1["n_negative_signed_r"] == 1


def test_pack_roundtrip_shape() -> None:
    vector = np.array([1, 0, 1, 1, 0, 0, 0, 1, 1], dtype=np.uint8)
    raw = bytes.fromhex(packed_hex(vector))
    recovered = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[: len(vector)]
    assert np.array_equal(vector, recovered)


if __name__ == "__main__":
    test_empirical_mode_and_tie_rule()
    test_ld_sign()
    test_pack_roundtrip_shape()
    print("public baseline helper tests passed")
