# Stage 2: conditional haplotype construction

This directory turns each fitted locus model into ranked partner configurations conditional on
lead state 0 or 1, compares those configurations with simple baselines, and checks agreement across
the public and donor panels.

## Reading order

1. `build_top10_empirical_v59.py` — fitted top-ten discovery, deterministic scoring, and coverage
   certification for both lead states.
2. `build_public_baseline_haplotypes_v73.py` — empirical conditional mode and LD-sign baselines.
3. `test_public_baseline_haplotypes_v73.py` — deterministic unit tests for baseline rules.
4. `compare_cohort_top_haplotypes_v72.py` — 1000 Genomes versus ROS/MAP comparison on valid shared
   coordinates.
5. The remaining scripts produce specific active audit artifacts and are not the preferred entry
   point for understanding the construction.

## Files

| Script | Status / environment | Purpose |
|---|---|---|
| `build_top10_empirical_v59.py` | current entry point; DCC/RCC | Discovers candidates, rescales them with deterministic quadrature, certifies top-list coverage, and records empirical conditional frequencies. |
| `build_public_baseline_haplotypes_v73.py` | current entry point; DCC | Constructs empirical-mode and LD-sign backgrounds, maps them to fitted ranks, and records sequence-level comparisons. |
| `test_public_baseline_haplotypes_v73.py` | current dependency; public | Checks tie handling, LD-sign coding, and packed-vector round trips. |
| `compare_cohort_top_haplotypes_v72.py` | current entry point; DCC with approved ROS/MAP aggregate | Compares panels in full when partner sets match and on shared coordinates when they differ. |
| `count_partners_v36.py` | provenance utility; DCC | Re-thresholds the stored lead-partner LD table and reproduces the Arm A/B partner counts. |
| `cond_probs_v40.py` | provenance utility; DCC | Converts previously stored joint probabilities to lead-state conditional probabilities; rankings are unchanged. |
| `verify_sparse_v33.py` | provenance utility; DCC | Legacy filename for the active unfactored-correlation probability check used by Supplementary Table 1. |

## Code chunks in the top-list workflow

`build_top10_empirical_v59.py` contains six logical blocks:

1. pack/unpack binary haplotypes for stable deduplication;
2. construct the q=1 quadrature representation;
3. generate deterministic and conditional-sampling candidates;
4. rescore every candidate at fixed higher quadrature order;
5. certify that the probability of a missed top-ten configuration is below the stated bound; and
6. count selected configurations among phased chromosomes carrying the requested lead state.

`build_public_baseline_haplotypes_v73.py` then aligns fitted coordinates to the phased matrix,
constructs the empirical and LD-sign rules, evaluates their fitted probabilities, maps them to an
existing fitted rank, and writes JSON plus CSV summaries with source checksums.

The `verify_sparse_v33.py` filename is retained solely because it names the recorded artifact and
production job. Sparse PCA is not a manuscript estimator; see `../../docs/development_history.md`.

