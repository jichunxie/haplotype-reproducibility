# Protected workflow

This repository publishes the exact analysis scripts but not the protected inputs or row-level
outputs. Run protected steps only inside an approved ROS/MAP environment.

## Required protected inputs

- phased ROS/MAP binary haplotype matrices and coordinate manifests;
- full ROS/MAP top-haplotype identities and empirical counts;
- Fujita benchmark files obtained under their applicable terms;
- protected hypothesis-level AlphaGenome prediction summaries; and
- canonical transcript annotation used for gene-level aggregation.

## Stages

1. Fit each locus with `analysis/estimation/fit_rosmap_locus_v51.py`, using
   `analysis/estimation/probit_fixed_adaptive_v51.py` on the approved filesystem.
2. Construct both lead states with
   `analysis/haplotype_construction/build_top10_empirical_v59.py`. Require every reported top-ten
   search to pass its coverage/certification checks.
3. Run AlphaGenome using the scripts in `analysis/alphagenome/`. Supply the API key through the
   environment, record client/output/track metadata, and checkpoint server responses.
4. Build the Fujita benchmark and enrichment analyses with `analysis/benchmark/`.
5. Export only the disclosure-safe aggregate schemas represented in `paper/figdata/`.

## Disclosure review before export

- no donor rows or genotype matrices;
- no ROS/MAP haplotype identities or partner allele vectors;
- no small carrier counts or candidate-level frequencies;
- no protected scored hypothesis rows;
- no filesystem credentials, API keys, tokens, or cookies; and
- every aggregate checked against the approved disclosure rule.

The public figure build consumes already-approved summaries and never reconstructs protected
candidate-level data.
