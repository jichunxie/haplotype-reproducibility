"""One-factor fixed-margin probit factor analysis with adaptive quadrature.

This production module intentionally implements only the estimator used in the
manuscript:

* q = 1;
* thresholds fixed at Jeffreys-corrected cohort allele frequencies;
* posterior-adaptive one-dimensional Gauss--Hermite quadrature;
* bounded L-BFGS optimization of the observed marginal likelihood;
* a genuine refit at every quadrature order; and
* likelihood and parameter-refinement checks before a fit is reportable.

There is no free-threshold likelihood branch.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri, roots_hermitenorm


LOG_2PI = math.log(2.0 * math.pi)


def jeffreys_margins(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return tau and p-tilde, where p-tilde=(m+1/2)/(n+1)."""
    X = _binary_matrix(X)
    n = X.shape[0]
    p_tilde = (X.sum(axis=0) + 0.5) / (n + 1.0)
    tau = ndtri(1.0 - p_tilde)
    if not np.isfinite(tau).all():
        raise FloatingPointError("Jeffreys-corrected thresholds must be finite")
    return tau, p_tilde


def collapse_patterns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse duplicate binary haplotypes and return patterns and multiplicities."""
    X = _binary_matrix(X)
    Xb = np.ascontiguousarray(X, dtype=np.uint8)
    packed = np.packbits(Xb, axis=1)
    key = np.ascontiguousarray(packed).view(
        np.dtype((np.void, packed.shape[1]))
    )
    _, idx, counts = np.unique(
        key.ravel(), return_index=True, return_counts=True
    )
    return Xb[idx].astype(np.float64), counts.astype(np.float64)


def _binary_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < 2:
        raise ValueError("X must be a non-empty n by p matrix with p >= 2")
    if not np.isfinite(X).all() or not np.isin(X, (0.0, 1.0)).all():
        raise ValueError("X must contain only finite zeroes and ones")
    return X


def _nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    if int(order) != order or order < 8:
        raise ValueError("quadrature order must be an integer >= 8")
    u, w = roots_hermitenorm(int(order))
    logw = np.full_like(w, -np.inf)
    positive = w > 0.0
    logw[positive] = np.log(w[positive]) - 0.5 * LOG_2PI
    return u, logw


def _posterior_modes(
    patterns: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    *,
    max_iter: int = 80,
    score_tol: float = 2e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return each pattern's posterior mode and negative Hessian for q=1."""
    signs = 2.0 * patterns - 1.0
    scale = np.sqrt(1.0 + a * a)
    offset = tau * scale
    f = np.zeros(len(patterns), dtype=np.float64)

    for _ in range(max_iter):
        eta = f[:, None] * a[None, :] - offset[None, :]
        z = signs * eta
        mills = np.exp(-0.5 * z * z - 0.5 * LOG_2PI - log_ndtr(z))
        score = -f + (signs * mills) @ a
        weight = np.maximum(mills * (mills + z), 0.0)
        hessian = 1.0 + weight @ (a * a)
        direction = score / hessian

        h0 = -0.5 * f * f + log_ndtr(z).sum(axis=1)
        directional = np.maximum(score * direction, 0.0)
        step = np.ones(len(f), dtype=np.float64)
        accepted = np.zeros(len(f), dtype=bool)
        trial = f.copy()
        for _ in range(40):
            pending = ~accepted
            if not pending.any():
                break
            fp = f[pending] + step[pending] * direction[pending]
            zp = signs[pending] * (
                fp[:, None] * a[None, :] - offset[None, :]
            )
            hp = -0.5 * fp * fp + log_ndtr(zp).sum(axis=1)
            ok = hp >= (
                h0[pending]
                + 1e-4 * step[pending] * directional[pending]
                - 1e-13 * np.maximum(1.0, np.abs(h0[pending]))
            )
            rows = np.flatnonzero(pending)
            if ok.any():
                trial[rows[ok]] = fp[ok]
                accepted[rows[ok]] = True
            step[rows[~ok]] *= 0.5
        if not accepted.all():
            raise RuntimeError(
                f"posterior-mode line search failed for "
                f"{int((~accepted).sum())} pattern(s)"
            )
        moved = float(np.max(np.abs(trial - f)))
        f = trial
        if moved < 1e-10:
            break

    eta = f[:, None] * a[None, :] - offset[None, :]
    z = signs * eta
    mills = np.exp(-0.5 * z * z - 0.5 * LOG_2PI - log_ndtr(z))
    score = -f + (signs * mills) @ a
    weight = np.maximum(mills * (mills + z), 0.0)
    hessian = 1.0 + weight @ (a * a)
    max_score = float(np.max(np.abs(score)))
    if max_score > score_tol:
        raise RuntimeError(
            f"posterior modes are unresolved: max score {max_score:.3e}"
        )
    if not np.isfinite(hessian).all() or np.min(hessian) <= 0.0:
        raise FloatingPointError("posterior Hessian is not positive")
    return f, hessian, max_score


def adaptive_loglik_score(
    patterns: np.ndarray,
    multiplicity: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    order: int,
    *,
    want_score: bool = True,
    max_tensor_entries: int = 2_000_000,
) -> tuple[float, np.ndarray | None, float]:
    """Evaluate the adaptive likelihood and Fisher-identity score for q=1."""
    patterns = np.asarray(patterns, dtype=np.float64)
    multiplicity = np.asarray(multiplicity, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    if patterns.ndim != 2 or multiplicity.shape != (len(patterns),):
        raise ValueError("pattern/multiplicity shapes disagree")
    if a.shape != (patterns.shape[1],) or tau.shape != a.shape:
        raise ValueError("a and tau must have one entry per variant")
    if not np.isfinite(a).all() or not np.isfinite(tau).all():
        raise ValueError("a and tau must be finite")

    u, logw = _nodes(order)
    fhat, hessian, mode_score = _posterior_modes(patterns, a, tau)
    n_nodes = len(u)
    p = len(a)
    scale = np.sqrt(1.0 + a * a)
    offset = tau * scale
    chain = tau * a / scale
    half_u2 = 0.5 * u * u
    chunk = max(
        1, int(max_tensor_entries // max(n_nodes * p, 1))
    )

    total = 0.0
    score = np.zeros_like(a) if want_score else None
    for start in range(0, len(patterns), chunk):
        stop = min(start + chunk, len(patterns))
        Xc = patterns[start:stop]
        mc = multiplicity[start:stop]
        signs = 2.0 * Xc - 1.0
        fn = (
            fhat[start:stop, None]
            + u[None, :] / np.sqrt(hessian[start:stop, None])
        )
        z = signs[:, None, :] * (
            fn[:, :, None] * a[None, None, :]
            - offset[None, None, :]
        )
        h = -0.5 * fn * fn + log_ndtr(z).sum(axis=2)
        exponent = logw[None, :] + h + half_u2[None, :]
        lse = logsumexp(exponent, axis=1)
        log_i = lse - 0.5 * np.log(hessian[start:stop])
        total += float(mc @ log_i)

        if want_score:
            posterior_w = np.exp(exponent - lse[:, None])
            mills = np.exp(
                -0.5 * z * z - 0.5 * LOG_2PI - log_ndtr(z)
            )
            signed_mills = signs[:, None, :] * mills
            weighted = posterior_w[:, :, None] * signed_mills
            score += np.einsum(
                "b,bmp,bm->p", mc, weighted, fn, optimize=True
            )
            score -= chain * np.einsum(
                "b,bmp->p", mc, weighted, optimize=True
            )
    return total, score, mode_score


def _project(a: np.ndarray, amax: float) -> np.ndarray:
    return np.clip(a, -amax, amax)


def _kkt(a: np.ndarray, score: np.ndarray, n: int, amax: float) -> float:
    projected = _project(a + score / n, amax)
    return float(np.max(np.abs(projected - a)))


def _mapped(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    psi = 1.0 / (1.0 + a * a)
    b = a * np.sqrt(psi)
    return b, psi


def _parameter_discrepancy(a_low: np.ndarray, a_high: np.ndarray) -> float:
    """Equation (29) without materializing either p by p outer product."""
    b_low, psi_low = _mapped(a_low)
    b_high, psi_high = _mapped(a_high)
    fro_sq = (
        float(b_low @ b_low) ** 2
        + float(b_high @ b_high) ** 2
        - 2.0 * float(b_low @ b_high) ** 2
    )
    fro = math.sqrt(max(fro_sq, 0.0)) / len(a_low)
    return float(max(fro, np.max(np.abs(psi_high - psi_low))))


def _fit_order(
    patterns: np.ndarray,
    multiplicity: np.ndarray,
    tau: np.ndarray,
    init: np.ndarray,
    *,
    psi_min: float,
    order: int,
    kkt_tol: float,
    relative_tol: float,
    max_iter: int,
) -> dict[str, Any]:
    n = int(multiplicity.sum())
    amax = math.sqrt(1.0 / psi_min - 1.0)
    x0 = _project(np.asarray(init, dtype=np.float64), amax)
    accepted_loglik: list[float] = []
    cache_x: np.ndarray | None = None
    cache_ll: float | None = None
    cache_score: np.ndarray | None = None
    cache_mode_score: float | None = None

    def evaluate(x: np.ndarray) -> tuple[float, np.ndarray, float]:
        nonlocal cache_x, cache_ll, cache_score, cache_mode_score
        if cache_x is None or not np.array_equal(x, cache_x):
            ll, score, mode_score = adaptive_loglik_score(
                patterns, multiplicity, x, tau, order, want_score=True
            )
            if score is None:
                raise AssertionError("score was not computed")
            cache_x = np.array(x, copy=True)
            cache_ll = float(ll)
            cache_score = np.asarray(score)
            cache_mode_score = float(mode_score)
        return cache_ll, cache_score, cache_mode_score

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        ll, score, _ = evaluate(x)
        return -ll / n, -score / n

    ll0, _, _ = evaluate(x0)
    accepted_loglik.append(ll0)

    def callback(xk: np.ndarray) -> None:
        ll, _, _ = evaluate(xk)
        accepted_loglik.append(ll)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(-amax, amax)] * len(x0),
        callback=callback,
        options={
            "maxiter": int(max_iter),
            "ftol": float(relative_tol),
            "gtol": float(kkt_tol),
            "maxls": 50,
            "maxcor": 20,
        },
    )
    a = np.asarray(result.x, dtype=np.float64)
    ll, score, mode_score = evaluate(a)
    if not accepted_loglik or accepted_loglik[-1] != ll:
        accepted_loglik.append(ll)
    ascent_slack = 1e-9 * max(1.0, abs(ll))
    if np.min(np.diff(accepted_loglik), initial=0.0) < -ascent_slack:
        raise AssertionError("accepted L-BFGS iterates decreased the likelihood")
    kkt = _kkt(a, score, n, amax)
    converged = bool(result.success and kkt <= kkt_tol)
    return {
        "a": a,
        "loglik": float(ll),
        "score": score,
        "kkt": kkt,
        "converged": converged,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "n_iter": int(result.nit),
        "n_eval": int(result.nfev),
        "accepted_loglik": accepted_loglik,
        "mode_score": mode_score,
        "order": int(order),
    }


def fit_fixed_margin_q1(
    X: np.ndarray,
    *,
    init: np.ndarray,
    tau: np.ndarray | None = None,
    psi_min: float = 0.01,
    initial_order: int = 64,
    max_order: int = 1024,
    eps_ll_per_haplotype: float = 1e-3,
    eps_theta: float = 1e-3,
    kkt_tol: float = 1e-5,
    relative_tol: float = 1e-10,
    max_iter_per_order: int = 500,
) -> dict[str, Any]:
    """Fit and validate the manuscript's q=1 fixed-margin estimator."""
    X = _binary_matrix(X)
    n, p = X.shape
    if not 0.0 < psi_min < 1.0:
        raise ValueError("psi_min must lie in (0,1)")
    if int(initial_order) != initial_order or initial_order < 8:
        raise ValueError("initial_order must be an integer >= 8")
    if int(max_order) != max_order or max_order < initial_order:
        raise ValueError("max_order must be >= initial_order")
    for value, name in (
        (eps_ll_per_haplotype, "eps_ll_per_haplotype"),
        (eps_theta, "eps_theta"),
        (kkt_tol, "kkt_tol"),
        (relative_tol, "relative_tol"),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")

    if tau is None:
        tau, p_tilde = jeffreys_margins(X)
    else:
        tau = np.asarray(tau, dtype=np.float64)
        if tau.shape != (p,) or not np.isfinite(tau).all():
            raise ValueError("tau must be finite with one entry per variant")
        p_tilde = 1.0 - ndtr(tau)
    init = np.asarray(init, dtype=np.float64)
    if init.shape == (p, 1):
        init = init[:, 0]
    if init.shape != (p,) or not np.isfinite(init).all():
        raise ValueError("init must be a finite p-vector or p by 1 array")

    patterns, multiplicity = collapse_patterns(X)
    fits: list[dict[str, Any]] = []
    current = init.copy()
    order = int(initial_order)
    passed = False
    final_refine_per_hap = math.inf
    final_theta = math.inf

    while True:
        fit = _fit_order(
            patterns,
            multiplicity,
            tau,
            current,
            psi_min=psi_min,
            order=order,
            kkt_tol=kkt_tol,
            relative_tol=relative_tol,
            max_iter=max_iter_per_order,
        )
        fits.append(fit)

        if len(fits) >= 2 and fit["converged"] and fits[-2]["converged"]:
            previous = fits[-2]
            ll_low_at_high, _, _ = adaptive_loglik_score(
                patterns,
                multiplicity,
                fit["a"],
                tau,
                previous["order"],
                want_score=False,
            )
            final_refine_per_hap = abs(
                fit["loglik"] - ll_low_at_high
            ) / n
            final_theta = _parameter_discrepancy(
                previous["a"], fit["a"]
            )
            passed = bool(
                final_refine_per_hap <= eps_ll_per_haplotype
                and final_theta <= eps_theta
            )
            fit["lower_order_at_final_loglik"] = float(ll_low_at_high)
            fit["refine_dev_per_haplotype"] = float(
                final_refine_per_hap
            )
            fit["parameter_discrepancy"] = float(final_theta)
            if passed:
                break

        if order >= max_order:
            break
        # A coarse rule can be too noisy for its score and likelihood stopping
        # conditions to agree. Its last feasible iterate is still a valid warm
        # start, so continue to the next order and require two consecutive
        # converged refits before reporting anything.
        current = fit["a"]
        order = min(2 * order, int(max_order))

    final = fits[-1]
    a = np.asarray(final["a"], dtype=np.float64)
    if a[0] < 0.0:
        a = -a
    b, psi = _mapped(a)
    unitvar = float(np.max(np.abs(b * b + psi - 1.0)))
    margin_error = float(
        np.max(np.abs((1.0 - ndtr(tau)) - p_tilde))
    )
    reportable = bool(
        passed
        and final["converged"]
        and final["kkt"] <= kkt_tol
        and unitvar < 1e-12
        and margin_error < 1e-12
    )
    return {
        "B": b[:, None],
        "psi": psi,
        "tau": tau,
        "a": a[:, None],
        "p_tilde": p_tilde,
        "loglik": float(final["loglik"]),
        "n_haplotypes": int(n),
        "n_patterns": int(len(patterns)),
        "q": 1,
        "psi_min": float(psi_min),
        "n_at_bound": int(
            np.sum(psi <= psi_min * (1.0 + 1e-9))
        ),
        "kkt": float(final["kkt"]),
        "unitvar_max_dev": unitvar,
        "margin_max_dev": margin_error,
        "quadrature_order": int(final["order"]),
        "quadrature_refine_dev_per_haplotype": float(
            final_refine_per_hap
        ),
        "parameter_refit_discrepancy": float(final_theta),
        "quadrature_converged": bool(passed),
        "converged": bool(final["converged"]),
        "reportable": reportable,
        "order_history": [
            {
                k: v
                for k, v in fit.items()
                if k not in {"a", "score", "accepted_loglik"}
            }
            | {
                "accepted_loglik": [
                    float(x) for x in fit["accepted_loglik"]
                ]
            }
            for fit in fits
        ],
        "integration": (
            "posterior-adaptive one-dimensional Gauss--Hermite; "
            "refit at every reported order"
        ),
        "optimizer": "bounded L-BFGS on average observed log likelihood",
        "fixed_tau": True,
        "free_threshold_fit": False,
    }
