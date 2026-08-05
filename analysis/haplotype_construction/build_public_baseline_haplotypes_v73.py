#!/usr/bin/env python3
"""Build prespecified public-panel haplotype baselines and compare with fitted top one.

The script uses the lossless 1000 Genomes phased matrices that supplied the
fixed-margin fits.  It does not refit the factor model or alter partner sets.
For each arm, locus and lead state it constructs:

* the empirical conditional mode (lexicographically first in a count tie);
* the LD-sign background (same allele as the lead for positive signed LD and
  the opposite allele for negative signed LD); and
* the existing fitted rank-one haplotype.

All configurations are evaluated on the fitted variant coordinates.  The
output is public-panel data and contains no ROS/MAP donor information.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("armA", "armB")
LEAD_STATES = (0, 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locus_to_variant_id(locus: str) -> str:
    chrom, pos, ref, alt = locus.split("_", 3)
    return f"{chrom}:{pos}:{ref}:{alt}"


def vector_from_alternate_ids(ids: np.ndarray, alternate_ids: list[str]) -> np.ndarray:
    alternate = set(alternate_ids)
    unknown = alternate.difference(ids.tolist())
    if unknown:
        raise ValueError(f"alternate IDs absent from fitted coordinates: {sorted(unknown)[:3]}")
    return np.fromiter((variant in alternate for variant in ids), dtype=np.uint8, count=len(ids))


def packed_hex(vector: np.ndarray) -> str:
    return np.packbits(vector.astype(np.uint8), bitorder="little").tobytes().hex()


def empirical_count(x: np.ndarray, vector: np.ndarray, lead_index: int, state: int) -> tuple[int, int]:
    conditional = x[x[:, lead_index] == state]
    return int(np.all(conditional == vector, axis=1).sum()), int(len(conditional))


def empirical_conditional_mode(
    x: np.ndarray, lead_index: int, state: int
) -> tuple[np.ndarray, dict]:
    conditional = x[x[:, lead_index] == state]
    if conditional.size == 0:
        raise ValueError(f"no haplotypes carry lead state {state}")
    partner_indices = np.delete(np.arange(x.shape[1]), lead_index)
    partners = conditional[:, partner_indices]
    unique, counts = np.unique(partners, axis=0, return_counts=True)
    maximum = int(counts.max())
    tied = np.flatnonzero(counts == maximum)
    # np.unique returns lexicographically sorted rows, so tied[0] implements
    # the prespecified coordinate-order tie rule without using fitted scores.
    selected = unique[tied[0]]
    vector = np.empty(x.shape[1], dtype=np.uint8)
    vector[lead_index] = state
    vector[partner_indices] = selected
    ordered_counts = np.sort(counts)[::-1]
    return vector, {
        "conditional_n": int(len(conditional)),
        "n_distinct_observed": int(len(unique)),
        "mode_count": maximum,
        "mode_probability": float(maximum / len(conditional)),
        "n_tied_modes": int(len(tied)),
        "second_count": int(ordered_counts[1]) if len(ordered_counts) > 1 else None,
    }


def signed_correlations(x: np.ndarray, lead_index: int) -> np.ndarray:
    lead = x[:, lead_index].astype(float)
    centered_lead = lead - lead.mean()
    centered = x.astype(float) - x.mean(axis=0)
    denominator = np.sqrt(np.sum(centered_lead**2) * np.sum(centered**2, axis=0))
    numerator = centered_lead @ centered
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = numerator / denominator
    correlation[lead_index] = 1.0
    return correlation


def ld_sign_background(
    x: np.ndarray, lead_index: int, state: int
) -> tuple[np.ndarray, dict]:
    correlation = signed_correlations(x, lead_index)
    vector = np.empty(x.shape[1], dtype=np.uint8)
    vector[lead_index] = state
    zero_fallbacks = 0
    for index in range(x.shape[1]):
        if index == lead_index:
            continue
        if correlation[index] > 0:
            vector[index] = state
        elif correlation[index] < 0:
            vector[index] = 1 - state
        else:
            conditional = x[x[:, lead_index] == state, index]
            ones = int(conditional.sum())
            vector[index] = int(ones > len(conditional) - ones)
            zero_fallbacks += 1
    return vector, {
        "minimum_absolute_signed_r": float(np.nanmin(np.abs(np.delete(correlation, lead_index))))
        if x.shape[1] > 1
        else None,
        "n_negative_signed_r": int(np.sum(np.delete(correlation, lead_index) < 0)),
        "n_zero_signed_r_fallbacks": zero_fallbacks,
    }


def conditional_log_probability(
    vector: np.ndarray,
    b: np.ndarray,
    psi: np.ndarray,
    tau: np.ndarray,
    p_tilde: np.ndarray,
    lead_index: int,
    order: int,
) -> float:
    from scipy.special import log_ndtr, logsumexp, roots_legendre

    nodes, weights = roots_legendre(order)
    factors = 8.0 * nodes
    log_weights = np.log(weights) + np.log(8.0)
    log_integrand = log_weights - 0.5 * factors**2 - 0.5 * np.log(2.0 * np.pi)
    scale = np.sqrt(psi)
    # Work in coordinate chunks to avoid a p-by-order temporary at the largest locus.
    chunk = 256
    for start in range(0, len(vector), chunk):
        stop = min(start + chunk, len(vector))
        eta = (b[start:stop, None] * factors[None, :] - tau[start:stop, None]) / scale[
            start:stop, None
        ]
        states = vector[start:stop, None]
        log_integrand += np.sum(np.where(states == 1, log_ndtr(eta), log_ndtr(-eta)), axis=0)
    state = int(vector[lead_index])
    denominator = p_tilde[lead_index] if state == 1 else 1.0 - p_tilde[lead_index]
    if not 0.0 < denominator < 1.0:
        raise ValueError("invalid fitted lead margin")
    return float(logsumexp(log_integrand) - np.log(denominator))


def align_matrix(input_path: Path, coefficient_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = np.load(input_path, allow_pickle=False)
    fitted = np.load(coefficient_path, allow_pickle=False)
    raw_ids = raw["ids"].astype(str)
    ids = fitted["ids"].astype(str)
    raw_lookup = {variant: index for index, variant in enumerate(raw_ids)}
    missing = [variant for variant in ids if variant not in raw_lookup]
    if missing:
        raise ValueError(f"fitted IDs absent from phased matrix: {missing[:3]}")
    columns = np.array([raw_lookup[variant] for variant in ids], dtype=int)
    x = raw["X"][:, columns].astype(np.uint8, copy=False)
    parameters = {
        "b": fitted["B"][:, 0].astype(float),
        "psi": fitted["psi"].astype(float),
        "tau": fitted["tau"].astype(float),
        "p_tilde": fitted["p_tilde"].astype(float),
    }
    return x, ids, parameters


def top10_lookup(path: Path) -> tuple[dict[str, dict], dict]:
    document = json.loads(path.read_text())
    return {row["locus"]: row for row in document["per_locus"]}, document


def strategy_record(
    strategy: str,
    vector: np.ndarray,
    fitted: np.ndarray,
    x: np.ndarray,
    ids: np.ndarray,
    lead_index: int,
    state: int,
    parameters: dict,
    top10: list[dict],
    quadrature_order: int,
    details: dict,
) -> dict:
    count, conditional_n = empirical_count(x, vector, lead_index, state)
    partner_mask = np.arange(len(ids)) != lead_index
    distance = int(np.sum(vector[partner_mask] != fitted[partner_mask]))
    alt_ids = ids[vector.astype(bool)].tolist()
    existing_rank = None
    for row in top10:
        if set(row["alternate_variant_ids"]) == set(alt_ids):
            existing_rank = int(row["rank"])
            break
    log_probability = conditional_log_probability(
        vector,
        parameters["b"],
        parameters["psi"],
        parameters["tau"],
        parameters["p_tilde"],
        lead_index,
        quadrature_order,
    )
    return {
        "strategy": strategy,
        "packed_little_endian_hex": packed_hex(vector),
        "alternate_variant_ids": alt_ids,
        "n_alternate": int(vector.sum()),
        "exactly_fitted_top1": bool(distance == 0),
        "partner_hamming_to_fitted_top1": distance,
        "normalized_partner_hamming_to_fitted_top1": float(distance / max(1, partner_mask.sum())),
        "empirical_count_given_lead_state": count,
        "empirical_probability_given_lead_state": float(count / conditional_n),
        "observed_in_panel": bool(count > 0),
        "fitted_top10_rank": existing_rank,
        "q1_log_probability": log_probability,
        "q1_probability": float(np.exp(log_probability)),
        **details,
    }


def summarize(rows: list[dict]) -> dict:
    output: dict[str, dict] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        output[arm] = {}
        for strategy in ("empirical_mode", "ld_sign"):
            selected = [row for row in arm_rows if row["strategy"] == strategy]
            distances = np.array([row["partner_hamming_to_fitted_top1"] for row in selected])
            output[arm][strategy] = {
                "n_locus_state": len(selected),
                "n_exactly_fitted_top1": int(np.sum(distances == 0)),
                "fraction_exactly_fitted_top1": float(np.mean(distances == 0)),
                "median_partner_hamming": float(np.median(distances)),
                "p95_partner_hamming": float(np.percentile(distances, 95)),
                "maximum_partner_hamming": int(np.max(distances)),
                "n_observed_in_panel": int(sum(row["observed_in_panel"] for row in selected)),
                "n_in_fitted_top10": int(sum(row["fitted_top10_rank"] is not None for row in selected)),
            }
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "arm",
        "locus",
        "lead_state",
        "n_variants",
        "n_partners",
        "strategy",
        "exactly_fitted_top1",
        "partner_hamming_to_fitted_top1",
        "normalized_partner_hamming_to_fitted_top1",
        "empirical_count_given_lead_state",
        "empirical_probability_given_lead_state",
        "observed_in_panel",
        "fitted_top10_rank",
        "q1_probability",
        "n_alternate",
        "packed_little_endian_hex",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--top10-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quadrature-order", type=int, default=2048)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict] = []
    source_files: list[dict] = []
    per_locus: list[dict] = []
    for arm in ARMS:
        lookups = {}
        top_documents = {}
        for state in LEAD_STATES:
            top_path = args.top10_root / f"top10_contrast_1kg_{arm}_lead{state}_v60.json"
            lookups[state], top_documents[state] = top10_lookup(top_path)
            source_files.append({"path": str(top_path), "sha256": sha256(top_path)})
        if set(lookups[0]) != set(lookups[1]):
            raise ValueError(f"lead-state locus mismatch in {arm}")

        for locus in sorted(lookups[0]):
            coefficient_path = args.fit_root / arm / f"{locus}.npz"
            input_path = args.input_root / arm / f"{locus}.npz"
            x, ids, parameters = align_matrix(input_path, coefficient_path)
            lead_id = locus_to_variant_id(locus)
            matches = np.flatnonzero(ids == lead_id)
            if len(matches) != 1:
                raise ValueError(f"lead coordinate not unique for {locus}: {len(matches)}")
            lead_index = int(matches[0])
            locus_rows = []
            for state in LEAD_STATES:
                source = lookups[state][locus]
                fitted = vector_from_alternate_ids(ids, source["top10"][0]["alternate_variant_ids"])
                if fitted[lead_index] != state:
                    raise ValueError(f"fitted top-one lead state mismatch for {arm} {locus} {state}")
                empirical, empirical_details = empirical_conditional_mode(x, lead_index, state)
                ld_sign, ld_details = ld_sign_background(x, lead_index, state)
                for strategy, vector, details in (
                    ("empirical_mode", empirical, empirical_details),
                    ("ld_sign", ld_sign, ld_details),
                ):
                    record = strategy_record(
                        strategy,
                        vector,
                        fitted,
                        x,
                        ids,
                        lead_index,
                        state,
                        parameters,
                        source["top10"],
                        args.quadrature_order,
                        details,
                    )
                    record.update(
                        {
                            "dataset": "1000 Genomes 30x, 503 unrelated EUR",
                            "arm": arm,
                            "locus": locus,
                            "lead_state": state,
                            "n_haplotypes": int(len(x)),
                            "n_variants": int(len(ids)),
                            "n_partners": int(len(ids) - 1),
                            "input_sha256": sha256(input_path),
                            "coefficient_sha256": sha256(coefficient_path),
                        }
                    )
                    locus_rows.append(record)
                    output_rows.append(record)
            per_locus.append(
                {
                    "arm": arm,
                    "locus": locus,
                    "n_variants": int(len(ids)),
                    "rows": locus_rows,
                }
            )

    document = {
        "script": Path(__file__).name,
        "definition": {
            "empirical_mode": "most frequent fitted-coordinate partner vector conditional on lead state; lexicographically first coordinate-order vector in a count tie",
            "ld_sign": "partner equals lead state for positive signed phased ALT-indicator correlation and is opposite for negative correlation",
            "comparison": "no refit; all distances exclude the lead and use the fitted public-panel coordinates",
        },
        "config": {"quadrature_order": args.quadrature_order, "quadrature_interval": [-8, 8]},
        "sources": source_files,
        "summary": summarize(output_rows),
        "per_locus": per_locus,
    }
    json_path = args.output_dir / "public_haplotype_baselines_v73.json"
    csv_path = args.output_dir / "public_haplotype_baselines_v73.csv"
    json_path.write_text(json.dumps(document, indent=2))
    write_csv(csv_path, output_rows)
    print(json.dumps(document["summary"], indent=2))


if __name__ == "__main__":
    main()
