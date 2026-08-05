# Workflow and provenance

## Public analysis path

1. Download and phase/harmonize the specified 1000 Genomes GRCh38 panel.
2. Select lead-linked partners for Arms A and B using the paper's fixed thresholds.
3. Fit the fixed-margin one-factor model with `analysis/estimation/`.
4. Build conditional top-haplotype lists and simple backgrounds with
   `analysis/haplotype_construction/`.
5. Run the known-truth simulation with `analysis/simulation/`.
6. If authorized and an AlphaGenome credential is available, run `analysis/alphagenome/`.
7. Run benchmark analyses, keeping protected row-level results inside the controlled environment.
8. Stage only approved compact aggregates in `paper/figdata/`.
9. Run `make verify figures`.

The exact production scripts retain version suffixes because those suffixes bind the code to the
recorded artifacts and compute jobs. Historical absolute path defaults identify their original
execution locations; input/output roots are exposed as command-line arguments or environment
variables where applicable. The public one-command build uses only relative paths.

## Seeds and numerical gates

- enrichment bootstrap seed: 640731; 10,000 whole-locus draws;
- v77 simulation master seed: 20260805;
- stochastic candidate discovery scripts record their seeds and draw counts;
- the q=1 estimator is deterministic and has no seed;
- the q=1 estimator must pass refinement, KKT, fixed-margin, and unit-variance gates; and
- branch-and-bound results are reported only when the top list is certified.

## Compute provenance

The result map records the relevant DCC/RCC job IDs. Failed and superseded jobs are retained in code
comments and artifact documentation so they cannot be mistaken for the source of reported results.
