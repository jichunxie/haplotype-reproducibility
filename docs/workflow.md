# Executable workflow and provenance

For a reader-oriented explanation of the code, start with
[`code_walkthrough.md`](code_walkthrough.md). This page summarizes the execution order and release
gates.

## Public analysis path

1. Obtain the specified 1000 Genomes GRCh38 panel and record source checksums.
2. Harmonize phased ALT coding and select Arm A/Arm B lead-linked partners.
3. Fit the fixed-margin one-factor model with `analysis/estimation/`.
4. Build conditional top-haplotype lists and simple backgrounds with
   `analysis/haplotype_construction/`.
5. Run the final known-truth simulation with `analysis/simulation/run_ranking_simulation_cell_v77.py`.
6. If an AlphaGenome credential is available, run the public scripts in `analysis/alphagenome/`.
7. Run the benchmark comparisons in `analysis/benchmark/`, keeping protected hypothesis rows inside
   the controlled environment.
8. Stage only approved compact aggregates in `paper/figdata/`.
9. Run `make verify figures`.

## Canonical entry points

| Stage | Entry point | Execution boundary |
|---|---|---|
| q=1 public fit | `analysis/estimation/fit_1kg_locus_v52.py` | DCC/public data |
| q=1 donor fit | `analysis/estimation/fit_rosmap_locus_v51.py` | RCC/protected data |
| fitted top ten | `analysis/haplotype_construction/build_top10_empirical_v59.py` | DCC or RCC, depending on cohort |
| simple public baselines | `analysis/haplotype_construction/build_public_baseline_haplotypes_v73.py` | DCC/public data |
| final simulation | `analysis/simulation/run_ranking_simulation_cell_v77.py` | DCC/synthetic data |
| full simulation array | `analysis/simulation/run_ranking_simulation_slurm_portable.sh` | Slurm/synthetic data |
| gene-level AlphaGenome | `analysis/alphagenome/build_total_transcript_alphagenome_v65.py` | DCC or RCC, depending on input |
| matched-count benchmark | `analysis/benchmark/compare_single_lead_enrichment_v64.py` | protected inputs; aggregate export only |
| final strategy comparison | `analysis/benchmark/compare_public_baseline_enrichment_v75.py` | protected inputs; aggregate export only |
| public display build | `scripts/build_paper_outputs.py` | public aggregate artifacts |

The complete script-level inventory is `manifests/code_inventory.csv`.

## Version suffixes and paths

Version suffixes bind production code to recorded artifacts and compute jobs. They do not define the
reading order. Historical absolute path defaults identify the original DCC/RCC execution roots;
portable launchers and current drivers expose input/output roots through arguments or environment
variables where feasible.

Do not move the original versioned work directories. The reader-oriented release mirror and the
production layout are described in [`dcc_rcc_layout.md`](dcc_rcc_layout.md).

## Seeds and numerical gates

- enrichment bootstrap seed: 640731; 10,000 whole-locus draws;
- v77 simulation master seed: 20260805;
- stochastic candidate-discovery scripts record their seeds and draw counts;
- the q=1 estimator is deterministic and has no seed;
- the q=1 estimator must pass refinement, KKT, fixed-margin, and unit-variance gates; and
- branch-and-bound results are reported only when the top list is certified.

Nested finite panels in v77 are prefixes of one 2,000-haplotype master panel, so changing panel size
does not change population truth or consume a different simulation branch.

## Release gates

Before publishing a change:

1. `make verify` passes;
2. `make figures` produces every expected output;
3. no protected file, credential, donor row, haplotype identity, or hypothesis-level scored table is
   present;
4. `manifests/result_map.csv` still maps every active display to its upstream analysis and job;
5. comment-only source changes have updated SHA-256 entries in `manifests/sha256.json`; and
6. the DCC/RCC code mirrors point to the same reviewed Git commit.

Historical corrections that help audit the record, but are not execution instructions, are kept in
[`development_history.md`](development_history.md).
