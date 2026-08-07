# Code walkthrough

This guide explains how the paper-specific scripts connect. It is designed for a reader who wants
to understand the implementation before launching a full rerun.

## Three reproduction levels

### Level 1: verify the released scientific record

Run:

```bash
make verify
```

This checks every file in `manifests/sha256.json`, asserts locked manuscript numbers, and exercises
the independent ranking-oracle and frequency-design tests. It makes no network calls and needs no
protected data.

### Level 2: rebuild manuscript displays

Run:

```bash
make figures
```

This reads only `paper/figdata/` and writes the active figures/tables under `paper/main/` and
`paper/supp/`. `scripts/build_paper_outputs.py` is the ordered display driver. The submitted copies
under `reference/` are comparison targets, not build inputs.

### Level 3: rerun upstream analyses

Follow the numbered stages below. Public and synthetic stages run on DCC or another Linux/Slurm
system. ROS/MAP stages require authorized RCC access. A complete upstream rerun is not a single
public command because protected inputs cannot be redistributed.

## Stage 0: inputs and fixed definitions

Inputs are described in `data_access.md`. Before analysis, fix:

- the 38 Wightman lead variants;
- GRCh38 coordinate and REF/ALT conventions;
- Arm A (`r^2 >= 0.8`) and Arm B (`r^2 >= 0.5`) partner rules;
- the 503 unrelated European 1000 Genomes panel used for fitting;
- the ROS/MAP approval and storage boundary;
- the AlphaGenome 1,048,576-bp window and RNA biosample list; and
- the Fujita `significant_by_2step_FDR` benchmark definition.

The public release contains compact manifests and summaries, not a downloader or a replacement for
source-data access agreements.

## Stage 1: estimate the conditional haplotype model

Start with `analysis/estimation/probit_fixed_adaptive_v51.py`. The public API is
`fit_fixed_margin_q1(X, ...)`. It fixes thresholds from Jeffreys-corrected margins, integrates the
latent factor adaptively, optimizes the observed likelihood, and requires numerical refinement and
KKT gates before reporting a fit.

The cohort drivers add data handling:

- `fit_1kg_locus_v52.py`: loads the public phased matrix and fits one arm/locus;
- `fit_rosmap_locus_v51.py`: extracts an approved ROS/MAP locus inside RCC and fits the same model.

After all loci finish, v53--v55 scripts produce compact fit diagnostics. Pairwise ROS/MAP residuals
are reduced inside RCC; no donor or variant-level protected matrix is exported.

## Stage 2: construct lead-state haplotypes

`build_top10_empirical_v59.py` is the fitted construction. For each lead state, it discovers
candidates by conditional sampling, deterministically rescales them by quadrature, and records a
coverage certificate. Empirical frequency is only a descriptive count within the fitted panel.

`build_public_baseline_haplotypes_v73.py` constructs two simpler rules without refitting:

- empirical conditional mode among chromosomes carrying the lead state;
- coordinate-wise allele selected from the sign of phased lead-partner association.

The script maps each simple sequence to its fitted top-ten rank. In the locked analysis all required
sequences were already among existing fitted predictions, so no new AlphaGenome call was needed for
the baseline comparison.

`compare_cohort_top_haplotypes_v72.py` compares public and donor-panel fitted lists. It uses all
coordinates only when partner sets match; otherwise it projects both panels onto their shared
coordinates before calculating identity or Hamming distance.

## Stage 3: known-truth simulation

The final entry point is `run_ranking_simulation_cell_v77.py`. One cell is identified by lead
frequency class, dependence scenario, panel size, partner count, and replicate.

Within each cell the script:

1. creates fixed population truth;
2. draws a 2,000-haplotype master panel and uses nested 500/1,000 prefixes;
3. fits the production q=1 estimator;
4. constructs fitted, empirical-mode, and LD-sign strategies;
5. obtains population and fitted top-ten lists by certified branch-and-bound search;
6. checks every four-partner search against exhaustive enumeration; and
7. writes one strict JSON result.

Run the full grid with `run_ranking_simulation_slurm_portable.sh`, aggregate with
`aggregate_ranking_simulation_v77.py`, and independently audit with
`audit_ranking_simulation_v77.py`.

Versions 74 and 76 remain because v77 imports their stable base and search functions. They are code
dependencies; v77 is the result version.

## Stage 4: query AlphaGenome

The AlphaGenome scripts build paired complete sequences for lead state 0 and state 1, query the
predefined RNA tracks, and summarize predictions over gene exon unions. Sequence hashes and
per-locus checkpoints prevent duplicate API calls and allow interrupted runs to resume.

Use:

- `build_single_lead_alphagenome_v63.py` for the conventional lead-only edit;
- `build_total_transcript_alphagenome_v65.py` for fitted/protected paired haplotypes;
- `build_onekg_total_transcript_alphagenome_v68.py` for public-panel ranks 1--10.

These scripts can be inspected publicly, but actual reruns require the reference cache, annotation,
sequence manifests, and an API credential. Hosted server behavior is outside the repository's
control.

## Stage 5: benchmark locus-gene rankings

The benchmark scripts first create one common locus-cell-gene universe, then attach the independently
defined Fujita indicator. A score is the absolute difference in log1p AlphaGenome expression between
lead states.

The main inference is matched-count enrichment: if K hypotheses in a comparison family carry the
Fujita indicator, exactly K top AlphaGenome scores are called. Whole loci are resampled jointly so
strategy and panel contrasts are paired.

- v64 compares fitted aggregation with a single-lead edit;
- v68 compares the 1000 Genomes and ROS/MAP fitted constructions;
- v75 compares fitted top one, empirical mode, LD sign, and single lead.

These analyses support ranking enrichment, not new eQTL discovery, probability calibration, causal
variant identification, or directional validation.

## Stage 6: release and manuscript outputs

Only approved compact outputs are copied into `paper/figdata/`. Then:

1. `scripts/verify_checksums.py` verifies the frozen release inputs;
2. `scripts/audit_results.py` asserts the manuscript's locked numerical claims;
3. `scripts/build_paper_outputs.py` invokes display builders in manuscript order; and
4. `manifests/result_map.csv` links every display to its builder, inputs, upstream script, compute
   provenance, and access class.

The public code inventory in `manifests/code_inventory.csv` is the fastest way to locate one script
without reading unrelated stages.

