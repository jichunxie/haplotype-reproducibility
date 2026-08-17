# Development history and legacy filenames

This page is for provenance. It is not part of the recommended code-reading path and none of the
items below is an instruction to rerun a superseded result.

## Why historical version numbers remain

Production filenames retain their version suffixes so a script can be tied to exact Slurm jobs and
locked artifacts. Renaming those files would obscure provenance. The current result flow is listed
in `manifests/code_inventory.csv`; version number alone does not determine status.

## JASA technical-check display reorganization

The JASA review version reduced the main paper to four figures. The one-factor adequacy display,
empirical rank-validation display, and known-truth simulation design moved from main Figures 2--4
to Supplementary Figures 9--11. The former main cross-panel table moved to Supplementary Table 3.
The remaining main displays were renumbered: simulation results became Figure 2, the donor-panel
enrichment comparison became Figure 3, and the public-panel comparison became Figure 4.

This was a placement and numbering change only. The locked artifacts, numerical claims, builders,
seeds, and compute provenance did not change. Legacy source names such as
`make_figure7_reference_panel_enrichment_v78.py`, `S2_q1_residual_heatmap.pdf`, and
`S3_q1_loading_density.pdf` remain to preserve their production history; the current display map is
`manifests/result_map.csv`.

## Conditional-probability normalization (`cond_probs_v40.py`)

An earlier stored value was a joint probability of a full haplotype and the alternate lead state,
although it had been described as conditional. Version 40 divides by the lead-state marginal
probability. Because that divisor is constant across candidates at a locus and lead state, rankings,
top-one identity, and log-probability gaps are unchanged. The released `cond_probs_v40.json` records
both forms and the assertions used in the correction.

## Legacy sparse name (`verify_sparse_v33.py`)

This filename arose during a sparse-PCA comparison. The sparse and dense constructions were
identical in the evaluated setting, and sparse PCA was retired from the manuscript. The artifact is
still active for a narrower purpose: candidate haplotypes are evaluated against the unfactored
working correlation matrix, using Genz integration where feasible and common-random-number GHK at
larger loci. The filename and artifact name remain unchanged to preserve the production link.

## Simulation versions 74, 76, and 77

- v74 implemented small-partner exhaustive enumeration and the shared q=1 simulation base.
- v76 added scalable certified branch-and-bound search.
- v77 added the final rare/common frequency design and reporting schema.

The final manuscript result is v77. Its source imports base and search functions from v74/v76, so
those files are current dependencies rather than competing analyses.

## Failed and superseded compute jobs

Failed and superseded job IDs are retained in the manuscript project's provenance taskboard and in
artifact-level records, where they can be audited without interrupting the public code narrative.
Only the jobs named in `manifests/result_map.csv` support active displays. The public source files do
not use informal debugging commentary as instructions.
