#!/usr/bin/env python
"""Fit one fixed-margin q=1 model to 503 unrelated 1000 Genomes EUR samples.

The same 1,006 alternate-coded phased haplotypes supply the Jeffreys-corrected
thresholds, likelihood, and deterministic signed-correlation spectral starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


ROOT = Path(os.environ.get(
    "CHAP_1KG_ROOT",
    "/hpc/group/xielab/jx42/CHAP/work/onekg_factor_v52",
))
INPUT_ROOT = Path(os.environ.get(
    "CHAP_1KG_INPUT_ROOT",
    "/hpc/group/xielab/jx42/CHAP/work/rosmap_factor_v51/public_inputs",
))
PSI_MIN = 0.01
INITIAL_ORDER = 64
MAX_ORDER = int(os.environ.get("CHAP_MAX_ORDER", "1024"))
EPS_LL_PER_HAP = 1e-3
EPS_THETA = 1e-3
KKT_TOL = 1e-5

sys.path.insert(0, str(ROOT / "code"))
import probit_fixed_adaptive_v51 as factor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("armA", "armB"), required=True)
    parser.add_argument(
        "--index", type=int, required=True, help="1-based manifest row"
    )
    return parser.parse_args()


def manifest_row(arm: str, index: int) -> tuple[str, str, int]:
    rows = [
        line.rstrip().split("\t")
        for line in (
            INPUT_ROOT / f"manifest_{arm}.tsv"
        ).read_text().splitlines()
        if line.strip()
    ]
    if not 1 <= index <= len(rows):
        raise IndexError(f"{arm} index {index} is outside 1..{len(rows)}")
    locus, chrom, n_variant = rows[index - 1]
    return locus, chrom, int(n_variant)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def leading_working_eigenvector(
    X: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Leading eigenvector of the full signed phased allelic correlation."""
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    if np.any(sd <= 0.0):
        raise ValueError("working-correlation input contains a monomorphic column")
    Z = (X - mean) / sd

    def matvec(v: np.ndarray) -> np.ndarray:
        return (Z.T @ (Z @ v)) / n

    operator = LinearOperator((p, p), matvec=matvec, dtype=np.float64)
    eigenvalue, eigenvector = eigsh(
        operator,
        k=1,
        which="LA",
        v0=np.ones(p, dtype=np.float64),
        tol=1e-10,
        maxiter=max(1000, 10 * p),
    )
    value = float(eigenvalue[0])
    vector = np.asarray(eigenvector[:, 0])
    if vector[0] < 0.0:
        vector = -vector
    residual = float(
        np.linalg.norm(matvec(vector) - value * vector)
        / max(1.0, abs(value))
    )
    if residual > 1e-7:
        raise RuntimeError(
            f"working-correlation eigensolver residual {residual:.3e}"
        )
    if abs(vector[0]) < 1e-10:
        raise RuntimeError(
            "leading working eigenvector has an unresolved lead sign"
        )
    return vector, value, residual


def warm_starts(vector: np.ndarray) -> list[tuple[str, np.ndarray]]:
    direction = np.sign(vector)
    direction[np.abs(vector) < 1e-12] = 0.0
    magnitude = vector / np.max(np.abs(vector))
    starts = [
        ("1kg-self-eigen-row-r0.50", 0.50 * direction),
        ("1kg-self-eigen-row-r0.15", 0.15 * direction),
        ("1kg-self-eigen-max-r0.50", 0.50 * magnitude),
    ]
    if any(start[0] <= 0.0 for _, start in starts):
        raise AssertionError("the lead loading must be positive in every start")
    return starts


def result_summary(result: dict) -> dict:
    keep = (
        "loglik",
        "n_haplotypes",
        "n_patterns",
        "q",
        "psi_min",
        "n_at_bound",
        "kkt",
        "unitvar_max_dev",
        "margin_max_dev",
        "quadrature_order",
        "quadrature_refine_dev_per_haplotype",
        "parameter_refit_discrepancy",
        "quadrature_converged",
        "converged",
        "reportable",
        "order_history",
        "integration",
        "optimizer",
        "fixed_tau",
        "free_threshold_fit",
    )
    return {key: result[key] for key in keep}


def json_safe(value):
    """Convert NumPy values and non-finite failed-run diagnostics to JSON."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    started = time.time()
    locus, chrom, expected_variants = manifest_row(args.arm, args.index)
    input_file = INPUT_ROOT / args.arm / f"{locus}.npz"
    with np.load(input_file, allow_pickle=False) as source:
        X = source["X"].astype(np.float64)
        ids = [str(value) for value in source["ids"].tolist()]

    if X.shape != (1006, expected_variants):
        raise AssertionError(
            f"{locus}: expected shape {(1006, expected_variants)}, got {X.shape}"
        )
    if len(ids) != expected_variants:
        raise AssertionError("manifest, IDs, and haplotype matrix disagree")
    if ids[0].replace(":", "_") != locus:
        raise AssertionError("the lead variant is not the first input column")
    if not np.isin(X, (0.0, 1.0)).all():
        raise ValueError("haplotypes must be complete and binary")
    if np.any(X.std(axis=0) <= 0.0):
        raise ValueError("input contains a monomorphic 1000 Genomes column")

    tau, p_tilde = factor.jeffreys_margins(X)
    eigenvector, eigenvalue, eigen_residual = (
        leading_working_eigenvector(X)
    )
    runs: list[dict] = []
    fitted: list[tuple[str, dict]] = []
    for label, init in warm_starts(eigenvector):
        run_started = time.time()
        try:
            fit = factor.fit_fixed_margin_q1(
                X,
                init=init,
                tau=tau,
                psi_min=PSI_MIN,
                initial_order=INITIAL_ORDER,
                max_order=MAX_ORDER,
                eps_ll_per_haplotype=EPS_LL_PER_HAP,
                eps_theta=EPS_THETA,
                kkt_tol=KKT_TOL,
            )
            runs.append({
                "start": label,
                "seconds": round(time.time() - run_started, 3),
                "status": (
                    "reportable" if fit["reportable"] else "failed_checks"
                ),
                "fit": result_summary(fit),
            })
            if fit["reportable"]:
                fitted.append((label, fit))
        except Exception as exc:
            runs.append({
                "start": label,
                "seconds": round(time.time() - run_started, 3),
                "status": "exception",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            })

    output_dir = ROOT / "results" / args.arm
    output_dir.mkdir(parents=True, exist_ok=True)
    if not fitted:
        failure = {
            "script": "fit_1kg_locus_v52.py",
            "module": "probit_fixed_adaptive_v51.py",
            "arm": args.arm,
            "locus": locus,
            "complete": False,
            "runs": runs,
        }
        (output_dir / f"{locus}.failed.json").write_text(
            json.dumps(json_safe(failure), indent=2, allow_nan=False)
        )
        raise RuntimeError(f"{locus}: no reportable warm-start fit")

    selected_label, selected = max(
        fitted, key=lambda item: item[1]["loglik"]
    )
    coefficient_file = output_dir / f"{locus}.npz"
    np.savez_compressed(
        coefficient_file,
        ids=np.asarray(ids, dtype="U"),
        B=selected["B"],
        psi=selected["psi"],
        tau=selected["tau"],
        p_tilde=selected["p_tilde"],
        a=selected["a"],
        warm_eigenvector=eigenvector,
    )
    metadata = {
        "script": "fit_1kg_locus_v52.py",
        "module": "probit_fixed_adaptive_v51.py",
        "arm": args.arm,
        "locus": locus,
        "chrom": chrom,
        "complete": True,
        "selected_start": selected_label,
        "selected_loglik": float(selected["loglik"]),
        "q": 1,
        "dataset": "1000 Genomes 30x, 503 unrelated EUR",
        "n_individuals": 503,
        "n_haplotypes": 1006,
        "threshold_source": (
            "1000 Genomes unrelated-EUR phased haplotypes; "
            "Jeffreys-corrected ALT frequencies (m+1/2)/(n+1)"
        ),
        "likelihood_source": (
            "1000 Genomes unrelated-EUR alternate-coded phased haplotypes"
        ),
        "warm_start_source": (
            "the same 1000 Genomes haplotypes; full signed phased "
            "alternate-allele working correlation"
        ),
        "warm_start_eigenvalue": eigenvalue,
        "warm_start_eigen_residual": eigen_residual,
        "n_requested_variants": expected_variants,
        "n_fitted_variants": len(ids),
        "n_monomorphic": 0,
        "input_file": str(input_file),
        "input_sha256": sha256(input_file),
        "psi_min": PSI_MIN,
        "initial_quadrature_order": INITIAL_ORDER,
        "maximum_quadrature_order": MAX_ORDER,
        "epsilon_ll_per_haplotype": EPS_LL_PER_HAP,
        "epsilon_theta": EPS_THETA,
        "kkt_tolerance": KKT_TOL,
        "free_threshold_fit": False,
        "randomness": "none",
        "runs": runs,
        "seconds_total": round(time.time() - started, 3),
        "coefficient_file": coefficient_file.name,
    }
    (output_dir / f"{locus}.json").write_text(
        json.dumps(json_safe(metadata), indent=2, allow_nan=False)
    )
    print(json.dumps({
        "arm": args.arm,
        "locus": locus,
        "selected_start": selected_label,
        "loglik": selected["loglik"],
        "n_fitted_variants": len(ids),
        "quadrature_order": selected["quadrature_order"],
        "kkt": selected["kkt"],
        "pva": float(np.sum(selected["B"][:, 0] ** 2) / len(ids)),
        "seconds": metadata["seconds_total"],
    }, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
