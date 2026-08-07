# Stage 5: eQTL benchmark and enrichment

This directory evaluates AlphaGenome locus-gene scores against Fujita et al.'s published
cell-type-specific two-step-FDR indicator. It does not claim new eQTL discovery or causal-variant
identification.

## Reading order

1. `compare_single_lead_enrichment_v64.py` — defines the common scoring, truth mapping,
   matched-count enrichment, and whole-locus bootstrap.
2. `compare_reference_panel_enrichment_v68.py` — compares donor- and public-panel fitted ranks.
3. `compare_public_baseline_enrichment_v75.py` — compares fitted, empirical-mode, LD-sign, and
   single-lead strategies.
4. `test_rosmap_top5_eqtl_v62.py` — protected internal-null and matched-count audit retained for the
   corresponding aggregate artifact.

## Files

| Script | Status / environment | Purpose |
|---|---|---|
| `compare_single_lead_enrichment_v64.py` | current entry point; DCC with approved inputs | Establishes the complete-exon common universe and compares fitted top one/five/ten with single lead. |
| `compare_reference_panel_enrichment_v68.py` | current entry point; DCC | Paired public-versus-donor comparison on one locus-gene universe. |
| `compare_public_baseline_enrichment_v75.py` | current entry point; DCC | Four-strategy comparison plus locus-gene and leave-one-locus-out audits. |
| `test_rosmap_top5_eqtl_v62.py` | protected provenance utility; RCC | Produces protected scored hypotheses and exports only approved recall/precision aggregates. |

## Code chunks

The three main comparison scripts use the same pattern:

1. collapse technical RNA tracks and, when applicable, haplotype ranks;
2. intersect all strategies on one locus-cell-gene universe;
3. attach the Fujita indicator using a significant variant-gene pair inside the same sequence
   window;
4. compute the absolute log1p lead-state contrast;
5. select exactly K top scores when K Fujita-positive hypotheses occur in that family;
6. calculate enrichment and hypergeometric summaries; and
7. resample whole loci jointly across strategies for paired intervals.

Full scored ROS/MAP hypothesis rows stay protected. Public JSON/CSV files contain only the
disclosure-safe aggregates needed for figures, tables, and the release audit.
