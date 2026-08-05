#!/usr/bin/env python3
"""Stratified comparison of 1000 Genomes and ROS/MAP fitted haplotypes.

No model is refitted.  Loci are separated according to whether the two fitted
models contain the same partner coordinates.  When ROS/MAP contains fewer
partners, haplotypes are projected onto the coordinates shared by the two
models before identity and Hamming distance are computed.

The output is disclosure-safe: it contains locus-level partner counts and
aggregate identity/distance summaries, but no protected ROS/MAP haplotype or
allele identities.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lead_id(locus):
    return locus.replace("_", ":")


def partner_alt_set(haplotype, locus):
    return frozenset(haplotype["alternate_variant_ids"]) - {lead_id(locus)}


def quantiles(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    return {
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def load_rows(path, arm, state):
    payload = json.load(open(path))
    assert payload["arm"] == arm
    assert payload["lead_state"] == state
    rows = {row["locus"]: row for row in payload["per_locus"]}
    assert len(rows) == payload["n_loci"]
    return rows


def load_partner_set(model_dir, arm, locus, row0, row1):
    """Read and authenticate the coordinate set used in a fitted model."""
    assert row0["locus"] == row1["locus"] == locus
    assert row0["n_variants"] == row1["n_variants"]
    assert row0["coefficient_sha256"] == row1["coefficient_sha256"]
    path = model_dir / arm / f"{locus}.npz"
    assert sha256(path) == row0["coefficient_sha256"], path
    with np.load(path, allow_pickle=False) as fitted:
        ids = [str(value) for value in fitted["ids"]]
    assert len(ids) == row0["n_variants"]
    assert ids[0] == lead_id(locus)
    assert len(ids) == len(set(ids))
    return frozenset(ids[1:])


def recover_rosmap_partner_set(row0, row1):
    """Recover ROS/MAP fitted coordinates from its complementary modal pair."""
    assert row0["locus"] == row1["locus"]
    assert row0["n_variants"] == row1["n_variants"]
    locus = row0["locus"]
    alt0 = partner_alt_set(row0["top10"][0], locus)
    alt1 = partner_alt_set(row1["top10"][0], locus)
    partners = alt0 ^ alt1
    expected = row0["n_variants"] - 1
    assert len(partners) == expected, (locus, len(partners), expected)
    return partners


def distance_record(one_hap, ros_hap, locus, compared_coordinates, rank, state):
    one_alt = partner_alt_set(one_hap, locus) & compared_coordinates
    ros_alt = partner_alt_set(ros_hap, locus) & compared_coordinates
    hamming = len(one_alt ^ ros_alt)
    n_compared = len(compared_coordinates)
    return {
        "lead_state": state,
        "rank": rank,
        "hamming": hamming,
        "normalized_hamming": hamming / n_compared if n_compared else 0.0,
        "exact": hamming == 0,
    }


def summarize_distances(records):
    return {
        "n_rank_matched_haplotype_pairs": len(records),
        "n_exact": int(sum(row["exact"] for row in records)),
        "fraction_exact": float(np.mean([row["exact"] for row in records])),
        "hamming": quantiles([row["hamming"] for row in records]),
        "normalized_hamming": quantiles([
            row["normalized_hamming"] for row in records
        ]),
    }


def compare_top_lists(one_row, ros_row, locus, compared_coordinates, limit):
    one = {
        partner_alt_set(haplotype, locus) & compared_coordinates
        for haplotype in one_row["top10"][:limit]
    }
    ros = {
        partner_alt_set(haplotype, locus) & compared_coordinates
        for haplotype in ros_row["top10"][:limit]
    }
    shared = one & ros
    denominator = min(len(one), len(ros))
    n_compared = len(compared_coordinates)
    nearest = []
    for source, target in ((one, ros), (ros, one)):
        for haplotype in sorted(source, key=lambda value: tuple(sorted(value))):
            distance = min(len(haplotype ^ candidate) for candidate in target)
            nearest.append({
                "hamming": distance,
                "normalized_hamming": (
                    distance / n_compared if n_compared else 0.0
                ),
            })
    return {
        "n_1000g_unique_after_projection": len(one),
        "n_rosmap_unique_after_projection": len(ros),
        "n_shared_exact": len(shared),
        "shared_fraction_of_smaller_list": (
            len(shared) / denominator if denominator else 1.0
        ),
        "lists_identical": one == ros,
        "nearest": nearest,
    }


def summarize_list_comparisons(records):
    nearest = [row for record in records for row in record["nearest"]]
    return {
        "n_locus_state_list_comparisons": len(records),
        "n_identical_lists": int(sum(row["lists_identical"] for row in records)),
        "fraction_identical_lists": float(np.mean([
            row["lists_identical"] for row in records
        ])),
        "shared_fraction_of_smaller_list": quantiles([
            row["shared_fraction_of_smaller_list"] for row in records
        ]),
        "nearest_hamming": quantiles([row["hamming"] for row in nearest]),
        "nearest_normalized_hamming": quantiles([
            row["normalized_hamming"] for row in nearest
        ]),
    }


def summarize_loci(loci, partner_status):
    selected = [row for row in loci if row["partner_status"] == partner_status]
    all_distances = [distance for row in selected for distance in row["distances"]]
    top1 = [distance for distance in all_distances if distance["rank"] == 1]
    return {
        "partner_status": partner_status,
        "n_loci": len(selected),
        "n_partners_1000g": quantiles([
            row["n_partners_1000g"] for row in selected
        ]),
        "n_partners_rosmap": quantiles([
            row["n_partners_rosmap"] for row in selected
        ]),
        "n_partners_missing_from_rosmap": quantiles([
            row["n_partners_missing_from_rosmap"] for row in selected
        ]),
        "top1": summarize_distances(top1),
        "ranks_1_to_10": summarize_distances(all_distances),
        "top5_lists": summarize_list_comparisons([
            record for row in selected for record in row["top5_lists"]
        ]),
        "top10_lists": summarize_list_comparisons([
            record for row in selected for record in row["top10_lists"]
        ]),
    }


def compare_arm(
    one_rows_by_state,
    ros_rows_by_state,
    arm,
    onekg_model_dir,
):
    assert one_rows_by_state[0].keys() == one_rows_by_state[1].keys()
    assert ros_rows_by_state[0].keys() == ros_rows_by_state[1].keys()
    assert one_rows_by_state[0].keys() == ros_rows_by_state[0].keys()

    loci = []
    for locus in sorted(one_rows_by_state[0]):
        one_partners = load_partner_set(
            onekg_model_dir,
            arm,
            locus,
            one_rows_by_state[0][locus],
            one_rows_by_state[1][locus],
        )
        ros_partners = recover_rosmap_partner_set(
            ros_rows_by_state[0][locus],
            ros_rows_by_state[1][locus],
        )
        assert ros_partners <= one_partners, locus
        shared = one_partners & ros_partners
        same = one_partners == ros_partners
        compared = one_partners if same else shared

        distances = []
        list_comparisons = {5: [], 10: []}
        for state in (0, 1):
            one_row = one_rows_by_state[state][locus]
            ros_row = ros_rows_by_state[state][locus]
            assert all(
                partner_alt_set(haplotype, locus) <= one_partners
                for haplotype in one_row["top10"]
            ), (arm, locus, state, "1000 Genomes alternate outside fitted set")
            assert all(
                partner_alt_set(haplotype, locus) <= ros_partners
                for haplotype in ros_row["top10"]
            ), (arm, locus, state, "ROS/MAP alternate outside fitted set")
            limit = min(len(one_row["top10"]), len(ros_row["top10"]), 10)
            for index in range(limit):
                distances.append(distance_record(
                    one_row["top10"][index],
                    ros_row["top10"][index],
                    locus,
                    compared,
                    rank=index + 1,
                    state=state,
                ))
            for limit in (5, 10):
                list_comparisons[limit].append(compare_top_lists(
                    one_row,
                    ros_row,
                    locus,
                    compared,
                    limit,
                ))

        top1_distances = [row for row in distances if row["rank"] == 1]
        loci.append({
            "locus": locus,
            "arm": arm,
            "partner_status": "same" if same else "different",
            "n_partners_1000g": len(one_partners),
            "n_partners_rosmap": len(ros_partners),
            "n_partners_shared": len(shared),
            "n_partners_missing_from_rosmap": len(one_partners - ros_partners),
            "n_rank_matched_haplotype_pairs": len(distances),
            "n_exact_on_compared_coordinates": int(sum(
                distance["exact"] for distance in distances
            )),
            "top1": summarize_distances(top1_distances),
            "ranks_1_to_10": summarize_distances(distances),
            "top5_lists_summary": summarize_list_comparisons(
                list_comparisons[5]
            ),
            "top10_lists_summary": summarize_list_comparisons(
                list_comparisons[10]
            ),
            "top5_lists": list_comparisons[5],
            "top10_lists": list_comparisons[10],
            "distances": distances,
        })

    return {
        "arm": arm,
        "n_loci": len(loci),
        "partner_availability": {
            "same_partner_set": int(sum(
                row["partner_status"] == "same" for row in loci
            )),
            "different_partner_set": int(sum(
                row["partner_status"] == "different" for row in loci
            )),
        },
        "same_partner_set": summarize_loci(loci, "same"),
        "different_partner_set_shared_coordinates_only": summarize_loci(
            loci, "different"
        ),
        "per_locus": [{key: value for key, value in row.items()
                       if key not in {"distances", "top5_lists", "top10_lists"}}
                      for row in loci],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onekg-dir", type=Path, required=True)
    parser.add_argument("--rosmap-dir", type=Path, required=True)
    parser.add_argument("--onekg-model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    inputs = []
    results = []
    for arm in ("armA", "armB"):
        one_rows_by_state = {}
        ros_rows_by_state = {}
        for state in (0, 1):
            one_path = args.onekg_dir / (
                f"top10_contrast_1kg_{arm}_lead{state}_v60.json"
            )
            ros_path = args.rosmap_dir / (
                f"top10_contrast_rosmap_{arm}_lead{state}_v60.json"
            )
            inputs.extend([one_path, ros_path])
            one_rows_by_state[state] = load_rows(one_path, arm, state)
            ros_rows_by_state[state] = load_rows(ros_path, arm, state)
        results.append(compare_arm(
            one_rows_by_state,
            ros_rows_by_state,
            arm,
            args.onekg_model_dir,
        ))

    payload = {
        "script": Path(__file__).name,
        "definition": (
            "No model is refitted. Loci are stratified by whether the fitted "
            "1000 Genomes and ROS/MAP models contain identical partner "
            "coordinates. Corresponding fitted ranks 1--10 are compared. At "
            "loci with different partner sets, identity and Hamming distance "
            "are computed after projecting both haplotypes onto the partner "
            "coordinates shared by the two fitted models."
        ),
        "partner_set_source": (
            "1000 Genomes partner coordinates are read directly from the ids "
            "array in its public fitted coefficient file, authenticated against "
            "the coefficient_sha256 stored in both lead-state artifacts. The "
            "ROS/MAP coordinates are recovered as the symmetric difference of "
            "the partner ALT sets in its X0=0 and X0=1 rank-one haplotypes; the "
            "script asserts at every locus that the recovered size equals "
            "n_variants-1. This identity holds at all 64 ROS/MAP fits."
        ),
        "privacy": (
            "Disclosure-safe locus-level partner counts and aggregate identity/"
            "distance summaries only; no protected ROS/MAP haplotype or allele "
            "identities and no empirical donor counts."
        ),
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs],
        "by_arm": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
