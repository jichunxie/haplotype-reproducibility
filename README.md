# HaploPerturb paper reproducibility archive

This repository contains the versioned analysis code, disclosure-safe result artifacts,
environment records, and figure/table builders for:

> *HaploPerturb: Low-rank copula construction of haplotype perturbations improves
> sequence-to-function analysis of Alzheimer's disease loci*

The reusable, study-independent software is maintained separately in
[`jichunxie/haploperturb`](https://github.com/jichunxie/haploperturb). This repository records the
paper-specific analysis.

## Start here

Choose the path that matches your goal:

| Goal | Start with | Requires protected data? |
|---|---|---|
| Check every released artifact and manuscript number | `make verify` | No |
| Rebuild every active figure and table | `make figures` | No |
| Understand the analysis from input to manuscript output | [`docs/code_walkthrough.md`](docs/code_walkthrough.md) | No |
| Find the role, status, inputs, and outputs of one script | [`manifests/code_inventory.csv`](manifests/code_inventory.csv) | No |
| Rerun public-panel or synthetic stages | [`docs/workflow.md`](docs/workflow.md) | No |
| Rerun ROS/MAP stages as an approved researcher | [`docs/protected_workflow.md`](docs/protected_workflow.md) | Yes |
| Understand the DCC/RCC production layout | [`docs/dcc_rcc_layout.md`](docs/dcc_rcc_layout.md) | Only for execution |
| Review superseded implementations or corrections | [`docs/development_history.md`](docs/development_history.md) | No |

New readers should follow the current workflow documents above. Version numbers such as `v51` and
`v77` identify the exact production implementation; they are provenance labels, not a suggested
reading order.

## Analysis flow

```text
GWAS leads + phased genotypes
        |
        v
1. Select LD partners and fit the q=1 fixed-margin model
   analysis/estimation/
        |
        v
2. Construct and compare conditional haplotypes
   analysis/haplotype_construction/
        |                         \
        |                          \--> 3. Known-truth simulation
        |                               analysis/simulation/
        v
4. Query AlphaGenome and summarize paired lead-state predictions
   analysis/alphagenome/
        |
        v
5. Evaluate against the Fujita cell-type-specific eQTL benchmark
   analysis/benchmark/
        |
        v
6. Stage disclosure-safe aggregates, verify claims, and rebuild displays
   paper/figdata/ -> scripts/ -> paper/main/ and paper/supp/
```

The first three stages can be understood using public or synthetic data. ROS/MAP fitting and
hypothesis-level benchmark rows remain inside the approved RCC environment. Only disclosure-safe
aggregates cross into this repository.

## Public reproduction

The public release supports two levels of reproduction:

1. `make verify` checks the immutable input checksums, locked numerical claims, and ranking-search
   tests.
2. `make figures` rebuilds all four active main figures, all eleven active supplementary figures,
   and all three supplementary tables from released aggregate artifacts.

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
- `paper/supp/supp-figures/`
- `paper/supp/supp-tables/`

Submitted reference PDFs and tables are in `reference/`. PDF bytes can vary with metadata and font
toolchains; numerical assertions and source-artifact checks are the primary verification gates.

## Repository map

```text
analysis/
  README.md               numbered stage overview and code-reading order
  estimation/             model estimator, cohort fit drivers, fit diagnostics
  haplotype_construction/ top-haplotype search, simple baselines, panel comparison
  simulation/             final v77 simulation and its v74/v76 implementation dependencies
  alphagenome/            sequence construction, API queries, gene-level summaries
  benchmark/              common-universe eQTL ranking and paired-locus inference
paper/
  figdata/                locked aggregate inputs and display builders
reference/                submitted figure/table outputs
environment/              production and portable build environments
manifests/
  code_inventory.csv      one row per analysis script, with status and data boundary
  result_map.csv          manuscript display -> builder -> artifact -> production job
  sha256.json             immutable release-input checksums
scripts/                  one-command build and fail-closed release audits
docs/                     workflow, code, data-access, and provenance guides
```

Each `analysis/` subdirectory has its own README. Those files explain what to read first, how the
larger scripts are divided into logical chunks, and which scripts are current entry points versus
inherited implementation dependencies.

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
[`manifests/result_map.csv`](manifests/result_map.csv) for the display-level provenance map.

The display map matches the current JASA review version: four main figures, eleven supplementary
figures, and three supplementary tables. Model-adequacy, empirical-rank, and simulation-design
displays formerly in the main paper are Supplementary Figures 9--11; the cross-panel agreement
table is Supplementary Table 3. This placement change did not alter an analysis result.

## Environment

The production run used Python 3.10.19 on Linux 5.14/glibc 2.34 under Slurm 25.11.1. Key package
versions were NumPy 1.26.4, SciPy 1.11.4, pandas 2.2.3, Matplotlib 3.10.6, PyArrow 12.0.1,
scikit-learn 1.7.2, statsmodels 0.14.5, seaborn 0.13.2, pysam 0.23.3, and AlphaGenome 0.4.0.
The hosted AlphaGenome API did not expose a permanent server-model version identifier, so exact
future re-querying also depends on server-side stability. Full details are in `environment/`.

## License and citation

Code is released under the MIT license. Data-derived artifacts retain the terms of their source
datasets; access-controlled data are not relicensed or redistributed. Please cite the paper, the
1000 Genomes resource, ROS/MAP, Fujita et al., and AlphaGenome as applicable.
