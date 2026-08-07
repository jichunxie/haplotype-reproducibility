# Stage 1: fixed-margin model estimation

This directory fits the one-factor latent Gaussian model and builds the diagnostics used to decide
whether the fit is adequate. The lead variant is column zero throughout.

## Reading order

1. `probit_fixed_adaptive_v51.py` — estimator implementation.
2. `fit_1kg_locus_v52.py` — public 1000 Genomes locus driver.
3. `fit_rosmap_locus_v51.py` — protected ROS/MAP locus driver.
4. `build_q1_pva_comparison_v53.py` — cohort-level variance summary.
5. `build_q1_pairwise_residuals_v54.py` — binary-correlation residual diagnostic.
6. `build_q1_loading_density_v55.py` — loading-density diagnostic.

## Files

| Script | Status / environment | Inputs | Output and purpose |
|---|---|---|---|
| `probit_fixed_adaptive_v51.py` | current dependency; DCC/RCC | binary haplotype matrix | Implements the fixed-margin q=1 likelihood, adaptive quadrature, bounded optimization, KKT check, and quadrature-refinement gate. |
| `fit_1kg_locus_v52.py` | current entry point; DCC | public phased matrix and arm manifest | One `.npz` fit and one JSON diagnostic record per locus. |
| `fit_rosmap_locus_v51.py` | protected entry point; RCC | approved ROS/MAP pfiles plus public manifest | Protected per-locus fit; only approved aggregate summaries may be exported. |
| `build_q1_pva_comparison_v53.py` | provenance utility; DCC | completed cohort summaries | Disclosure-safe public-versus-donor proportion-of-variance summary. |
| `build_q1_pairwise_residuals_v54.py` | current diagnostic; DCC/RCC | fits plus phased matrices | Public residual matrices and disclosure-safe protected scalar summaries. |
| `build_q1_loading_density_v55.py` | current diagnostic; DCC | public q=1 coefficient files | Rank-magnitude and cumulative loading-energy summaries. |

## Code chunks in the estimator

`probit_fixed_adaptive_v51.py` is organized as follows:

1. **Input and fixed margins** — validate a binary matrix, compute Jeffreys-corrected marginal
   probabilities, and collapse duplicate haplotypes.
2. **Adaptive integration** — find each distinct haplotype's posterior factor mode and curvature,
   then evaluate the marginal likelihood and score with adaptive Gauss-Hermite quadrature.
3. **Constrained parameterization** — map unconstrained working parameters to loadings and
   uniquenesses while enforcing the uniqueness floor.
4. **Optimization at one quadrature order** — maximize the observed log likelihood and evaluate the
   KKT residual.
5. **Refinement loop** — refit at increasing quadrature orders and report only fits that satisfy the
   likelihood, parameter, fixed-margin, unit-variance, and KKT gates.

The two cohort drivers then add deterministic spectral warm starts, choose the best reportable fit,
and write explicit failure JSON if no start passes.

