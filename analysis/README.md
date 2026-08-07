# Analysis code guide

The analysis is organized by scientific stage. Run stages in the order below; do not infer order
from the numeric version suffixes in filenames.

| Stage | Directory | Question answered | Primary current entry points |
|---:|---|---|---|
| 1 | [`estimation/`](estimation/) | What one-factor fixed-margin model is fitted to each phased panel? | `fit_1kg_locus_v52.py`, `fit_rosmap_locus_v51.py` |
| 2 | [`haplotype_construction/`](haplotype_construction/) | Which partner configuration is most probable given each lead state, and how do panels/rules agree? | `build_top10_empirical_v59.py`, `build_public_baseline_haplotypes_v73.py`, `compare_cohort_top_haplotypes_v72.py` |
| 3 | [`simulation/`](simulation/) | When does finite-panel estimation recover the population conditional mode? | `run_ranking_simulation_cell_v77.py` |
| 4 | [`alphagenome/`](alphagenome/) | What gene-expression contrast does each paired sequence construction produce? | `build_total_transcript_alphagenome_v65.py`, `build_onekg_total_transcript_alphagenome_v68.py` |
| 5 | [`benchmark/`](benchmark/) | Do high-scoring locus-gene pairs align with the Fujita eQTL benchmark? | `compare_single_lead_enrichment_v64.py`, `compare_reference_panel_enrichment_v68.py`, `compare_public_baseline_enrichment_v75.py` |

## Script status labels

The per-folder READMEs and `../manifests/code_inventory.csv` use four labels:

- **current entry point**: invoke this script for the final analysis stage;
- **current dependency**: imported by a current entry point or required for an active manuscript
  check, but usually not invoked first;
- **protected entry point**: current, but requires approved ROS/MAP data or protected row-level
  results inside RCC;
- **provenance utility**: reproduces an active compact artifact or correction but is not part of the
  usual end-to-end rerun.

No script in `analysis/` is an informal AI scratch note. Historical context that is useful for
auditing, but not needed to understand the current flow, is isolated in
[`../docs/development_history.md`](../docs/development_history.md).

## Data movement boundary

```text
DCC: 1000 Genomes, synthetic simulation, public-panel AlphaGenome stages
RCC:  ROS/MAP genotypes, protected haplotype identities, protected hypothesis rows
                         |
                         | disclosure review
                         v
Public repository: compact aggregate JSON/CSV/NPZ artifacts only
```

The public display build never reads donor-level data. See
[`../docs/protected_workflow.md`](../docs/protected_workflow.md) before running or exporting anything
on RCC.

