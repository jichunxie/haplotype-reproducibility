# Stage 3: known-truth ranking simulation

The final simulation is version 77. It studies rare and common lead variants, three dependence
regimes, panels of 500/1,000/2,000 haplotypes, and 4/32/256 partners.

## What to run

- Local design and oracle tests: `make verify` from the repository root.
- One final simulation cell: `run_ranking_simulation_cell_v77.py`.
- Full Slurm array: `run_ranking_simulation_slurm_portable.sh`.
- Aggregate completed cells: `aggregate_ranking_simulation_v77.py`.
- Audit the complete grid: `audit_ranking_simulation_v77.py`.

## Why v74 and v76 remain

The final v77 script imports stable implementation blocks from earlier versions:

- v74 supplies q=1 truth generation, finite-panel sampling, exhaustive small-k enumeration, and
  shared constants;
- v76 supplies the scalable branch-and-bound top-L search and certification logic;
- v77 adds frequency strata, robust finite-panel handling, and the final reporting schema.

Therefore v74 and v76 are **current dependencies**, not alternative result sets. Manuscript results
come from the audited v77 aggregate only.

## Files

| Script | Status | Purpose |
|---|---|---|
| `run_ranking_simulation_cell_v77.py` | current entry point | Generates truth and nested panels, fits q=1, evaluates three strategies, certifies population/fitted top tens, and writes strict JSON. |
| `run_ranking_simulation_slurm_portable.sh` | current entry point | Portable 1,620-cell Slurm launcher using `REPRO_ROOT`, `PYTHON`, and optional `OUTPUT_ROOT`. |
| `aggregate_ranking_simulation_v77.py` | current entry point | Validates completeness and creates strategy, panel, availability, and JSON summaries. |
| `audit_ranking_simulation_v77.py` | current entry point | Independently checks the final grid, frequency design, searches, and exhaustive k=4 results. |
| `test_ranking_frequency_design_v77.py` | current dependency | Deterministic tests for frequency bounds, availability handling, quadrature stability, and ties. |
| `test_ranking_oracle_v76.py` | current dependency | Independent exact/certification tests for the scalable top-L oracle. |
| `run_ranking_simulation_cell_v76.py` | current dependency | Branch-and-bound search and scalable conditional-mixture implementation imported by v77. |
| `run_ranking_simulation_cell_v74.py` | current dependency | Base truth, panel, estimator, and small-k enumeration functions imported by v77. |
| `run_ranking_simulation_replicate_array_v77.sh` | production provenance | Original DCC launcher with recorded absolute paths; use the portable launcher for a new run. |

## Code chunks in the final cell driver

`run_ranking_simulation_cell_v77.py` proceeds through:

1. deterministic seed derivation and rare/common frequency generation;
2. latent loading construction and population parameter generation;
3. nested finite-panel simulation;
4. production q=1 fitting with reportability gates;
5. empirical-mode and LD-sign construction, including unavailable states;
6. population and fitted top-ten search with certification;
7. exhaustive k=4 cross-checks; and
8. strict JSON serialization of truth, fit, strategies, and diagnostics.

