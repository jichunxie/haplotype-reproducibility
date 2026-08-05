#!/usr/bin/env python
"""Fit one ROS/MAP q=1 factor model for one Arm A or Arm B locus.

ROS/MAP phased haplotypes supply the Jeffreys-corrected thresholds and the
likelihood. Public 1000 Genomes phased haplotypes supply only deterministic
spectral warm starts. Individual-level ROS/MAP data never leave RCC.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


ROOT = Path(os.environ.get(
    "CHAP_ROSMAP_ROOT",
    "/data/irb/biostatisticsbioinformatics/pro00103365/jx42/"
    "chap_factor_v51",
))
PFILES = Path(os.environ.get(
    "CHAP_ROSMAP_PFILES",
    "/data/irb/biostatisticsbioinformatics/pro00103365/Data/"
    "AD-Knowledge/phased_wgs_pfiles",
))
PLINK2 = Path(os.environ.get("CHAP_PLINK2", str(Path.home() / "bin/plink2")))
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
    parser.add_argument("--index", type=int, required=True, help="1-based manifest row")
    return parser.parse_args()


def manifest_row(arm: str, index: int) -> tuple[str, str, int]:
    rows = [
        line.rstrip().split("\t")
        for line in (ROOT / "public_inputs" / f"manifest_{arm}.tsv")
        .read_text().splitlines()
        if line.strip()
    ]
    if not 1 <= index <= len(rows):
        raise IndexError(f"{arm} index {index} is outside 1..{len(rows)}")
    locus, chrom, n_variant = rows[index - 1]
    return locus, chrom, int(n_variant)


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(command)
            + "\n"
            + result.stdout[-2000:]
            + "\n"
            + result.stderr[-4000:]
        )


def read_haps(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, str]], int]:
    rows: dict[str, np.ndarray] = {}
    alleles: dict[str, tuple[str, str]] = {}
    n_haplotypes: int | None = None
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip().split()
            if len(fields) < 7:
                raise RuntimeError(f"malformed .haps row in {path}")
            variant = fields[1]
            values = np.asarray(fields[5:], dtype=np.uint8)
            if not np.isin(values, (0, 1)).all():
                raise RuntimeError(f"{variant}: nonbinary or missing phased allele")
            if n_haplotypes is None:
                n_haplotypes = len(values)
            elif len(values) != n_haplotypes:
                raise RuntimeError("inconsistent haplotype counts in .haps")
            rows[variant] = values
            alleles[variant] = (fields[3], fields[4])
    if n_haplotypes is None:
        raise RuntimeError("PLINK exported no variants")
    return rows, alleles, n_haplotypes


def read_alt_frequencies(path: Path) -> dict[str, float]:
    with path.open() as handle:
        header = handle.readline().lstrip("#").rstrip().split("\t")
        fields = [line.rstrip().split("\t") for line in handle if line.strip()]
    id_col = header.index("ID")
    freq_name = "ALT_FREQS" if "ALT_FREQS" in header else "ALT_FREQ"
    freq_col = header.index(freq_name)
    return {row[id_col]: float(row[freq_col]) for row in fields}


def alternate_coded(
    desired: list[str],
    rows: dict[str, np.ndarray],
    alleles: dict[str, tuple[str, str]],
) -> tuple[np.ndarray, list[str], list[str], list[dict]]:
    present = [variant for variant in desired if variant in rows]
    missing = [variant for variant in desired if variant not in rows]
    matrices: list[np.ndarray] = []
    coding: list[dict] = []
    for variant in present:
        _, _, ref, alt = variant.split(":", 3)
        a1, a2 = alleles[variant]
        raw = rows[variant]
        if (a1, a2) == (alt, ref):
            values = 1 - raw
            rule = "1-A2_indicator"
        elif (a1, a2) == (ref, alt):
            values = raw
            rule = "A2_indicator"
        else:
            raise RuntimeError(
                f"{variant}: PLINK alleles {(a1, a2)} do not match REF/ALT "
                f"{(ref, alt)}"
            )
        matrices.append(values)
        coding.append({
            "variant": variant,
            "plink_a1": a1,
            "plink_a2": a2,
            "alt_coding_rule": rule,
        })
    if not matrices:
        raise RuntimeError("none of the requested variants is present in ROS/MAP")
    return np.stack(matrices, axis=1), present, missing, coding


def leading_working_eigenvector(X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Leading eigenvector of the full signed phased allelic correlation."""
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    polymorphic = sd > 0.0
    Z = np.zeros_like(X)
    Z[:, polymorphic] = (
        X[:, polymorphic] - mean[polymorphic]
    ) / sd[polymorphic]

    def matvec(v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(v)
        if polymorphic.any():
            out[polymorphic] = (
                Z[:, polymorphic].T @ (Z[:, polymorphic] @ v[polymorphic])
            ) / n
        out[~polymorphic] = v[~polymorphic]
        return out

    operator = LinearOperator((p, p), matvec=matvec, dtype=np.float64)
    v0 = np.ones(p, dtype=np.float64)
    eigenvalue, eigenvector = eigsh(
        operator,
        k=1,
        which="LA",
        v0=v0,
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
        raise RuntimeError(f"working-correlation eigensolver residual {residual:.3e}")
    if abs(vector[0]) < 1e-10:
        raise RuntimeError("leading working eigenvector has an unresolved lead sign")
    return vector, value, residual


def warm_starts(vector: np.ndarray) -> list[tuple[str, np.ndarray]]:
    direction = np.sign(vector)
    direction[np.abs(vector) < 1e-12] = 0.0
    magnitude = vector / np.max(np.abs(vector))
    starts = [
        ("1kg-eigen-row-r0.50", 0.50 * direction),
        ("1kg-eigen-row-r0.15", 0.15 * direction),
        ("1kg-eigen-max-r0.50", 0.50 * magnitude),
    ]
    if any(start[0] <= 0.0 for _, start in starts):
        raise AssertionError("the lead loading must be positive in every warm start")
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
    """Convert NumPy scalars and non-finite diagnostics to strict JSON values."""
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
    public_file = ROOT / "public_inputs" / args.arm / f"{locus}.npz"
    public = np.load(public_file)
    X_1kg_all = public["X"].astype(np.float64)
    desired = [str(value) for value in public["ids"].tolist()]
    if len(desired) != expected_variants or desired[0].replace(":", "_") != locus:
        raise AssertionError("public manifest and locus file disagree")

    task_dir = ROOT / "extract" / args.arm / locus
    output_dir = ROOT / "results" / args.arm
    task_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    extract_file = task_dir / "variants.extract"
    extract_file.write_text("\n".join(desired) + "\n")
    prefix = task_dir / "rosmap"
    pfile = PFILES / f"ref_hg38_chr{chrom}"

    run([
        str(PLINK2),
        "--pfile", str(pfile),
        "--extract", str(extract_file),
        "--export", "haps",
        "--out", str(prefix),
        "--threads", str(int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
    ])
    run([
        str(PLINK2),
        "--pfile", str(pfile),
        "--extract", str(extract_file),
        "--freq",
        "--out", str(prefix),
        "--threads", str(int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
    ])
    rows, alleles, n_haplotypes = read_haps(prefix.with_suffix(".haps"))
    if n_haplotypes != 890:
        raise AssertionError(
            f"{locus}: expected 890 ROS/MAP haplotypes, got {n_haplotypes}"
        )
    X_ros_all, present, missing, coding = alternate_coded(
        desired, rows, alleles
    )
    frequencies = read_alt_frequencies(prefix.with_suffix(".afreq"))
    frequency_error = float(max(
        abs(X_ros_all[:, j].mean() - frequencies[variant])
        for j, variant in enumerate(present)
    ))
    if frequency_error > 5e-7:
        raise AssertionError(
            f"{locus}: alternate coding disagrees with PLINK ALT_FREQS by "
            f"{frequency_error:.3e}"
        )
    if desired[0] not in present:
        raise RuntimeError(f"{locus}: lead variant is absent from ROS/MAP")

    public_pos = {variant: j for j, variant in enumerate(desired)}
    X_1kg_present = X_1kg_all[:, [public_pos[v] for v in present]]
    polymorphic_ros = X_ros_all.std(axis=0) > 0.0
    excluded_monomorphic = [
        variant for variant, keep in zip(present, polymorphic_ros) if not keep
    ]
    ids = [variant for variant, keep in zip(present, polymorphic_ros) if keep]
    X_ros = X_ros_all[:, polymorphic_ros]
    X_1kg = X_1kg_present[:, polymorphic_ros]
    coding = [row for row, keep in zip(coding, polymorphic_ros) if keep]
    if not ids or ids[0] != desired[0]:
        raise RuntimeError(f"{locus}: lead is monomorphic in ROS/MAP")
    if X_ros.shape[1] < 2:
        raise RuntimeError(f"{locus}: fewer than two polymorphic ROS/MAP variants")

    tau, p_tilde = factor.jeffreys_margins(X_ros)
    eigenvector, eigenvalue, eigen_residual = leading_working_eigenvector(X_1kg)
    runs: list[dict] = []
    fitted: list[tuple[str, dict]] = []
    for label, init in warm_starts(eigenvector):
        run_started = time.time()
        try:
            fit = factor.fit_fixed_margin_q1(
                X_ros,
                init=init,
                tau=tau,
                psi_min=PSI_MIN,
                initial_order=INITIAL_ORDER,
                max_order=MAX_ORDER,
                eps_ll_per_haplotype=EPS_LL_PER_HAP,
                eps_theta=EPS_THETA,
                kkt_tol=KKT_TOL,
            )
            row = {
                "start": label,
                "seconds": round(time.time() - run_started, 3),
                "status": "reportable" if fit["reportable"] else "failed_checks",
                "fit": result_summary(fit),
            }
            runs.append(row)
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

    if not fitted:
        failure = {
            "script": "fit_rosmap_locus_v51.py",
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

    selected_label, selected = max(fitted, key=lambda item: item[1]["loglik"])
    np.savez_compressed(
        output_dir / f"{locus}.npz",
        ids=np.asarray(ids, dtype="U"),
        B=selected["B"],
        psi=selected["psi"],
        tau=selected["tau"],
        p_tilde=selected["p_tilde"],
        a=selected["a"],
        warm_eigenvector=eigenvector,
    )
    metadata = {
        "script": "fit_rosmap_locus_v51.py",
        "module": "probit_fixed_adaptive_v51.py",
        "arm": args.arm,
        "locus": locus,
        "chrom": chrom,
        "complete": True,
        "selected_start": selected_label,
        "selected_loglik": float(selected["loglik"]),
        "q": 1,
        "n_rosmap_individuals": 445,
        "n_rosmap_haplotypes": int(n_haplotypes),
        "threshold_source": (
            "ROS/MAP phased haplotypes; Jeffreys-corrected ALT frequencies "
            "(m+1/2)/(n+1)"
        ),
        "likelihood_source": "ROS/MAP alternate-coded phased haplotypes",
        "warm_start_source": (
            "1000 Genomes 30x, 503 unrelated EUR, full signed phased "
            "alternate-allele working correlation"
        ),
        "warm_start_eigenvalue": eigenvalue,
        "warm_start_eigen_residual": eigen_residual,
        "n_requested_variants": len(desired),
        "n_present_rosmap": len(present),
        "n_fitted_variants": len(ids),
        "missing_rosmap": missing,
        "monomorphic_rosmap": excluded_monomorphic,
        "n_monomorphic_1kg_among_fitted": int(np.sum(X_1kg.std(axis=0) == 0.0)),
        "alt_frequency_validation_max_abs_error": frequency_error,
        "allele_coding": coding,
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
        "coefficient_file": f"{locus}.npz",
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
        "missing": len(missing),
        "monomorphic": len(excluded_monomorphic),
        "quadrature_order": selected["quadrature_order"],
        "kkt": selected["kkt"],
        "seconds": metadata["seconds_total"],
    }, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
