# HaploPerturb paper reproducibility archive

This repository contains the versioned code, disclosure-safe results, environment records, and
figure/table builders for:

> *HaploPerturb: Low-rank copula construction of haplotype perturbations improves
> sequence-to-function analysis of Alzheimer's disease loci*

The reusable study-independent software is maintained separately at
[`jichunxie/haploperturb`](https://github.com/jichunxie/haploperturb).

## What can be reproduced publicly

The public release has two reproducibility levels:

1. `make verify` checks immutable input checksums and all locked numerical claims.
2. `make figures` rebuilds all seven active main figures, all eight active supplementary figures,
   the main cross-panel table, and both supplementary tables from the released aggregate artifacts.

The public build does not require ROS/MAP donor genotypes or scored hypothesis-level AlphaGenome
predictions. Those inputs are access-controlled and are not redistributed. Authorized researchers
can rerun the protected stages using the exact analysis scripts and instructions in
[`docs/protected_workflow.md`](docs/protected_workflow.md).

## Quick start

Using Conda or Mamba:

```bash
conda env create -f environment/figure-environment.yml
conda activate haploperturb-paper
make verify
make figures
```

Using Docker:

```bash
docker build -t haploperturb-paper .
docker run --rm -v "$PWD:/work" -w /work haploperturb-paper make verify figures
```

Generated outputs are written to:

- `paper/main/figures/`
- `paper/main/table1_cohort_haplotype_agreement.tex`
- `paper/supp/supp-figures/`
- `paper/supp/supp-tables/`

The submitted reference PDFs and tables are in `reference/`. PDF bytes may differ because PDF
metadata and font toolchains vary; numerical assertions and source-artifact checks are the primary
verification gates.

## Repository map

```text
analysis/
  estimation/             locked q=1 estimator and cohort fitting drivers
  haplotype_construction/ public baselines, top-haplotype ranking, panel comparison
  simulation/             v77 known-truth simulation, aggregation, and audit
  alphagenome/            sequence-to-function query and gene-summary code
  benchmark/              Fujita benchmark and enrichment analyses
paper/figdata/             disclosure-safe locked inputs and display builders
reference/                 submitted figure/table outputs
environment/               exact production and portable build environments
manifests/                 checksums and result-to-source map
scripts/                   one-command build and fail-closed audits
docs/                      data access, workflow, protection, and provenance notes
```

## Data boundaries

Publicly included:

- 1000 Genomes-derived manifests, conditional-haplotype summaries, and compact fit diagnostics;
- known-truth simulation summaries;
- disclosure-safe ROS/MAP aggregate summaries;
- aggregate AlphaGenome/eQTL benchmark summaries; and
- public figure and table source artifacts.

Not included:

- individual-level ROS/MAP genotypes;
- protected ROS/MAP haplotype identities, counts, or candidate-level frequencies;
- hypothesis-level protected AlphaGenome predictions or scored benchmark tables; and
- API credentials.

See [`docs/data_access.md`](docs/data_access.md) for public download locations and controlled-access
accessions.

## Verification status

The release audit asserts the values reported in the manuscript, including 38 loci; 128 fitted
cohort/arm/locus models; cross-panel rank-one agreement; all 1,620 v77 simulation cells and 3,240
population plus 3,240 fitted certified searches; the main complete-exon enrichment estimates; the
simple-baseline estimates; and 27/27 unfactored-correlation mode checks. See
[`manifests/result_map.csv`](manifests/result_map.csv) for the complete display-to-code mapping.

## Environment

The production run used Python 3.10.19 on Linux 5.14/glibc 2.34 under Slurm 25.11.1. Key package
versions were NumPy 1.26.4, SciPy 1.11.4, pandas 2.2.3, Matplotlib 3.10.6, PyArrow 12.0.1,
scikit-learn 1.7.2, statsmodels 0.14.5, seaborn 0.13.2, pysam 0.23.3, and AlphaGenome 0.4.0.
The hosted AlphaGenome API did not expose a permanent server-model version identifier, so exact
future re-querying also depends on server-side stability. Full details are in `environment/`.

## License and citation

Code is released under the BSD 3-Clause license. The data-derived artifacts retain the terms of
their source datasets; access-controlled data are not relicensed or redistributed. Please cite the
paper, the 1000 Genomes resource, ROS/MAP, Fujita et al., and AlphaGenome as applicable.
