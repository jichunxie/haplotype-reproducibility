#!/usr/bin/env python3
"""Fast binary-scale pairwise residual diagnostic for fitted q=1 models.

This is a descriptive goodness-of-fit diagnostic, not a factor-rank test.
For each locus it compares the empirical alternate-allele indicator
correlation with the correlation implied by the fitted one-factor probit
model.  The model-implied bivariate probabilities are evaluated as a
one-dimensional Gaussian-factor integral and all pairs are computed at once
by matrix multiplication.

The script supports the public 1000 Genomes inputs and the protected ROS/MAP
PLINK haplotype exports.  ROS/MAP output contains aggregate locus-level
diagnostics only; no individual-level or variant-level protected data leave
RCC.  The optional heat-map artifact is therefore enabled only for 1000
Genomes.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr, roots_hermitenorm
from scipy.sparse.linalg import eigsh


FINAL_ORDER = 2048
CHECK_ORDER = 1024
HEATMAP_BINS = 120
HEATMAP_MIN_P = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("onekg", "rosmap"), required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--extract-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heatmap-output", type=Path)
    return parser.parse_args()


def read_haps(path: Path) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip().split()
            if len(fields) < 7:
                raise RuntimeError(f"malformed .haps row in {path}")
            values = np.asarray(fields[5:], dtype=np.uint8)
            if not np.isin(values, (0, 1)).all():
                raise RuntimeError(f"{fields[1]}: nonbinary phased allele")
            rows[fields[1]] = values
    if not rows:
        raise RuntimeError(f"no variants in {path}")
    return rows


def load_onekg_input(
    input_root: Path,
    arm: str,
    locus: str,
    fitted_ids: list[str],
) -> np.ndarray:
    with np.load(input_root / arm / f"{locus}.npz", allow_pickle=False) as source:
        X = source["X"].astype(np.float64)
        ids = [str(value) for value in source["ids"].tolist()]
    if ids != fitted_ids:
        raise AssertionError(f"{arm}/{locus}: input and fitted IDs disagree")
    return X


def load_rosmap_input(
    extract_root: Path,
    arm: str,
    locus: str,
    fitted_ids: list[str],
    metadata: dict,
) -> np.ndarray:
    rows = read_haps(extract_root / arm / locus / "rosmap.haps")
    coding = {row["variant"]: row["alt_coding_rule"]
              for row in metadata["allele_coding"]}
    columns = []
    for variant in fitted_ids:
        raw = rows[variant]
        rule = coding[variant]
        if rule == "A2_indicator":
            values = raw
        elif rule == "1-A2_indicator":
            values = 1 - raw
        else:
            raise RuntimeError(f"{variant}: unknown ALT coding rule {rule}")
        columns.append(values)
    return np.stack(columns, axis=1).astype(np.float64)


def empirical_pairwise(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_hat = X.mean(axis=0)
    variance = p_hat * (1.0 - p_hat)
    if np.any(variance <= 0.0):
        raise ValueError("pairwise diagnostic received a monomorphic variant")
    joint = (X.T @ X) / X.shape[0]
    corr = (
        joint - np.outer(p_hat, p_hat)
    ) / np.sqrt(np.outer(variance, variance))
    np.fill_diagonal(corr, 1.0)
    return p_hat, joint, corr


def fitted_pairwise(
    B: np.ndarray,
    psi: np.ndarray,
    tau: np.ndarray,
    p_tilde: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_hermitenorm(order)
    weights = weights / math.sqrt(2.0 * math.pi)
    conditional = ndtr(
        (
            B[:, None] * nodes[None, :]
            - tau[:, None]
        ) / np.sqrt(psi)[:, None]
    )
    joint = (conditional * weights[None, :]) @ conditional.T
    variance = p_tilde * (1.0 - p_tilde)
    corr = (
        joint - np.outer(p_tilde, p_tilde)
    ) / np.sqrt(np.outer(variance, variance))
    np.fill_diagonal(joint, p_tilde)
    np.fill_diagonal(corr, 1.0)
    return joint, corr


def upper_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def residual_summary(
    empirical_joint: np.ndarray,
    empirical_corr: np.ndarray,
    fitted_joint: np.ndarray,
    fitted_corr: np.ndarray,
) -> dict:
    p = empirical_corr.shape[0]
    corr_residual = empirical_corr - fitted_corr
    joint_residual = empirical_joint - fitted_joint
    np.fill_diagonal(corr_residual, 0.0)
    np.fill_diagonal(joint_residual, 0.0)
    corr_upper = upper_values(corr_residual)
    joint_upper = upper_values(joint_residual)
    partner_upper = (
        upper_values(corr_residual[1:, 1:])
        if p > 2 else np.asarray([], dtype=np.float64)
    )
    observed_upper = upper_values(empirical_corr)
    largest_positive = float(eigsh(
        corr_residual,
        k=1,
        which="LA",
        v0=np.ones(p, dtype=np.float64),
        tol=1e-8,
        return_eigenvectors=False,
    )[0])
    operator_norm = float(abs(eigsh(
        corr_residual,
        k=1,
        which="LM",
        v0=np.ones(p, dtype=np.float64),
        tol=1e-8,
        return_eigenvectors=False,
    )[0]))
    return {
        "correlation_residual_rmse": float(np.sqrt(np.mean(corr_upper ** 2))),
        "correlation_residual_mae": float(np.mean(np.abs(corr_upper))),
        "correlation_residual_p95_abs": float(
            np.quantile(np.abs(corr_upper), 0.95)
        ),
        "correlation_residual_max_abs": float(np.max(np.abs(corr_upper))),
        "lead_correlation_residual_rmse": float(np.sqrt(
            np.mean(corr_residual[0, 1:] ** 2)
        )),
        "partner_correlation_residual_rmse": (
            float(np.sqrt(np.mean(partner_upper ** 2)))
            if partner_upper.size else None
        ),
        "joint_probability_residual_rmse": float(
            np.sqrt(np.mean(joint_upper ** 2))
        ),
        "joint_probability_residual_p95_abs": float(
            np.quantile(np.abs(joint_upper), 0.95)
        ),
        "offdiagonal_squared_fit_fraction": float(
            1.0
            - np.sum(corr_upper ** 2)
            / max(np.sum(observed_upper ** 2), np.finfo(float).tiny)
        ),
        "residual_largest_positive_eigenvalue": largest_positive,
        "residual_operator_norm": operator_norm,
        "residual_largest_positive_eigenvalue_per_variant": (
            largest_positive / p
        ),
    }


def variant_position(variant: str) -> int:
    fields = variant.replace("_", ":").split(":")
    return int(fields[1])


def block_average(matrix: np.ndarray, bins: list[np.ndarray]) -> np.ndarray:
    output = np.empty((len(bins), len(bins)), dtype=np.float64)
    for i, rows in enumerate(bins):
        for j, columns in enumerate(bins):
            block = matrix[np.ix_(rows, columns)]
            if i == j and len(rows) > 1:
                values = block[np.triu_indices(len(rows), k=1)]
                output[i, j] = float(values.mean())
            else:
                output[i, j] = float(block.mean())
    return output


def heatmap_artifact(
    ids: list[str],
    empirical_corr: np.ndarray,
    fitted_corr: np.ndarray,
) -> dict[str, np.ndarray]:
    positions = np.asarray([variant_position(variant) for variant in ids])
    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    empirical = empirical_corr[np.ix_(order, order)]
    fitted = fitted_corr[np.ix_(order, order)]
    residual = empirical - fitted
    np.fill_diagonal(residual, 0.0)
    n_bins = min(HEATMAP_BINS, len(ids))
    bins = [
        np.asarray(chunk, dtype=int)
        for chunk in np.array_split(np.arange(len(ids)), n_bins)
    ]
    lead_sorted_index = int(np.flatnonzero(order == 0)[0])
    lead_bin = next(i for i, values in enumerate(bins)
                    if lead_sorted_index in values)
    return {
        "positions": positions,
        "bin_start": np.asarray(
            [positions[values[0]] for values in bins], dtype=np.int64
        ),
        "bin_end": np.asarray(
            [positions[values[-1]] for values in bins], dtype=np.int64
        ),
        "bin_size": np.asarray([len(values) for values in bins], dtype=np.int64),
        "lead_bin": np.asarray(lead_bin, dtype=np.int64),
        "observed": block_average(empirical, bins),
        "fitted": block_average(fitted, bins),
        "residual": block_average(residual, bins),
    }


def group_summary(rows: list[dict], cohort: str, arm: str) -> dict:
    selected = [row for row in rows if row["arm"] == arm]
    values = np.asarray([
        row["correlation_residual_rmse"] for row in selected
    ])
    return {
        "cohort": cohort,
        "arm": arm,
        "n_loci": len(selected),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    if args.cohort == "onekg" and args.input_root is None:
        raise ValueError("--input-root is required for 1000 Genomes")
    if args.cohort == "rosmap" and args.extract_root is None:
        raise ValueError("--extract-root is required for ROS/MAP")
    if args.cohort == "rosmap" and args.heatmap_output is not None:
        raise ValueError("protected ROS/MAP matrices cannot be exported")

    started = time.time()
    cohort_label = (
        "1000 Genomes unrelated EUR"
        if args.cohort == "onekg" else "ROS/MAP"
    )
    rows: list[dict] = []
    heatmap_candidates: list[tuple[float, str, str]] = []
    for arm in ("armA", "armB"):
        for metadata_file in sorted((args.result_root / arm).glob("*.json")):
            metadata = json.loads(metadata_file.read_text())
            if not metadata.get("complete", False):
                continue
            locus = metadata["locus"]
            coefficient_file = args.result_root / arm / f"{locus}.npz"
            with np.load(coefficient_file, allow_pickle=False) as fit:
                ids = [str(value) for value in fit["ids"].tolist()]
                B = fit["B"][:, 0].astype(np.float64)
                psi = fit["psi"].astype(np.float64)
                tau = fit["tau"].astype(np.float64)
                p_tilde = fit["p_tilde"].astype(np.float64)
            if args.cohort == "onekg":
                X = load_onekg_input(
                    args.input_root, arm, locus, ids
                )
            else:
                X = load_rosmap_input(
                    args.extract_root, arm, locus, ids, metadata
                )
            if X.shape != (metadata.get(
                "n_haplotypes",
                metadata.get("n_rosmap_haplotypes"),
            ), len(ids)):
                raise AssertionError(f"{arm}/{locus}: unexpected X shape")

            empirical_p, empirical_joint, empirical_corr = empirical_pairwise(X)
            fitted_joint_check, fitted_corr_check = fitted_pairwise(
                B, psi, tau, p_tilde, CHECK_ORDER
            )
            fitted_joint, fitted_corr = fitted_pairwise(
                B, psi, tau, p_tilde, FINAL_ORDER
            )
            summary = residual_summary(
                empirical_joint,
                empirical_corr,
                fitted_joint,
                fitted_corr,
            )
            corr_refinement = upper_values(fitted_corr - fitted_corr_check)
            joint_refinement = upper_values(fitted_joint - fitted_joint_check)
            summary.update({
                "cohort": cohort_label,
                "arm": arm,
                "locus": locus,
                "n_haplotypes": int(X.shape[0]),
                "n_fitted_variants": len(ids),
                "empirical_frequency_min": float(empirical_p.min()),
                "empirical_frequency_max": float(empirical_p.max()),
                "quadrature_check_order": CHECK_ORDER,
                "quadrature_final_order": FINAL_ORDER,
                "correlation_refinement_rmse": float(np.sqrt(
                    np.mean(corr_refinement ** 2)
                )),
                "correlation_refinement_max_abs": float(
                    np.max(np.abs(corr_refinement))
                ),
                "joint_probability_refinement_rmse": float(np.sqrt(
                    np.mean(joint_refinement ** 2)
                )),
            })
            rows.append(summary)
            if (
                args.cohort == "onekg"
                and len(ids) >= HEATMAP_MIN_P
                and summary["partner_correlation_residual_rmse"] is not None
            ):
                heatmap_candidates.append((
                    summary["partner_correlation_residual_rmse"],
                    arm,
                    locus,
                ))
            print(
                cohort_label,
                arm,
                locus,
                f"p={len(ids)}",
                f"RMSE={summary['correlation_residual_rmse']:.6f}",
                f"refine={summary['correlation_refinement_rmse']:.2e}",
                flush=True,
            )

    expected = {"armA": 27, "armB": 37}
    for arm, count in expected.items():
        observed = sum(row["arm"] == arm for row in rows)
        if observed != count:
            raise AssertionError(
                f"{cohort_label} {arm}: expected {count} loci, got {observed}"
            )

    heatmap_selection = None
    if args.heatmap_output is not None:
        _, arm, locus = max(heatmap_candidates)
        metadata = json.loads(
            (args.result_root / arm / f"{locus}.json").read_text()
        )
        with np.load(
            args.result_root / arm / f"{locus}.npz",
            allow_pickle=False,
        ) as fit:
            ids = [str(value) for value in fit["ids"].tolist()]
            B = fit["B"][:, 0].astype(np.float64)
            psi = fit["psi"].astype(np.float64)
            tau = fit["tau"].astype(np.float64)
            p_tilde = fit["p_tilde"].astype(np.float64)
        X = load_onekg_input(args.input_root, arm, locus, ids)
        _, _, empirical_corr = empirical_pairwise(X)
        _, fitted_corr = fitted_pairwise(
            B, psi, tau, p_tilde, FINAL_ORDER
        )
        heatmap = heatmap_artifact(ids, empirical_corr, fitted_corr)
        args.heatmap_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.heatmap_output,
            cohort=np.asarray(cohort_label),
            arm=np.asarray(arm),
            locus=np.asarray(locus),
            n_fitted_variants=np.asarray(len(ids), dtype=np.int64),
            selection_rule=np.asarray(
                "largest partner-partner correlation-residual RMSE "
                f"among public 1KGP loci with p >= {HEATMAP_MIN_P}"
            ),
            **heatmap,
        )
        heatmap_selection = {
            "cohort": cohort_label,
            "arm": arm,
            "locus": locus,
            "n_fitted_variants": len(ids),
            "selection_rule": (
                "largest partner-partner correlation-residual RMSE "
                f"among public 1KGP loci with p >= {HEATMAP_MIN_P}"
            ),
            "artifact": str(args.heatmap_output),
        }

    artifact = {
        "script": "build_q1_pairwise_residuals_v54.py",
        "complete": True,
        "cohort": cohort_label,
        "q": 1,
        "scale": (
            "Pearson correlation of observed alternate-allele indicators; "
            "model-implied correlation from fitted bivariate probit "
            "probabilities"
        ),
        "interpretation": (
            "descriptive off-diagonal goodness-of-fit diagnostic, not a "
            "formal factor-rank test"
        ),
        "quadrature": {
            "rule": (
                "standard-normal Gauss--Hermite factor integration; all "
                "pairs evaluated by matrix multiplication"
            ),
            "check_order": CHECK_ORDER,
            "final_order": FINAL_ORDER,
        },
        "heatmap_selection": heatmap_selection,
        "groups": [
            group_summary(rows, cohort_label, arm)
            for arm in ("armA", "armB")
        ],
        "per_locus": rows,
        "seconds_total": round(time.time() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n"
    )
    print("wrote", args.output)
    if args.heatmap_output is not None:
        print("wrote", args.heatmap_output)


if __name__ == "__main__":
    main()
