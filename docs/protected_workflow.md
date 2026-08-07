# Protected ROS/MAP workflow

This repository publishes the analysis source but not the protected inputs or row-level outputs.
Run the steps below only as an approved researcher inside the ROS/MAP RCC project.

## Required protected inputs

- phased ROS/MAP binary haplotype matrices or approved pfiles;
- full ROS/MAP top-haplotype identities and empirical counts;
- Fujita benchmark files obtained under their applicable terms;
- protected hypothesis-level AlphaGenome prediction summaries; and
- the canonical transcript annotation used for gene-level aggregation.

## Ordered protected flow

### 1. Fit each ROS/MAP locus

Run `analysis/estimation/fit_rosmap_locus_v51.py`. It extracts only the requested locus, harmonizes
ALT coding, uses public 1000 Genomes information only for deterministic warm starts, and fits the
same fixed-margin q=1 estimator used for the public panel.

Keep per-locus `.npz`, extracted `.haps`, coding records, and failure diagnostics inside RCC. Export
only the approved cohort-level summary.

### 2. Construct both lead-state top lists

Run `analysis/haplotype_construction/build_top10_empirical_v59.py` separately for lead state 0 and
lead state 1. Require every reported top-ten search to pass its coverage/certification check.

Full vectors, empirical counts, and candidate-level frequencies remain protected. The public v60
summary contains only disclosure-safe ranks and distance summaries.

### 3. Query AlphaGenome

Use the scripts in `analysis/alphagenome/`. Supply the API key only through the environment, record
client/output/track metadata, checkpoint each locus/rank, and retain the full prediction rows inside
RCC.

The complete-exon-union summary used by the main benchmark comes from
`build_total_transcript_alphagenome_v65.py`. The terminal-exon-tail summary is a sensitivity output,
not the main estimand.

### 4. Run the benchmark

Use `analysis/benchmark/test_rosmap_top5_eqtl_v62.py` for the protected top-five audit and the v64,
v68, and v75 comparison scripts for the manuscript analyses. Create the common evaluation universe
before attaching the Fujita indicator, and resample whole loci jointly across compared strategies.

Full scored hypothesis tables stay inside RCC.

### 5. Review aggregate exports

An export must contain only a schema already represented by an approved file in `paper/figdata/` or
must receive a new disclosure review. Record the producing script, Git commit, RCC job ID, source
checksums, output checksum, and the exact export schema.

## Disclosure checklist

- no donor rows or genotype matrices;
- no ROS/MAP haplotype identities or partner allele vectors;
- no small carrier counts or candidate-level frequencies;
- no protected scored hypothesis rows;
- no filesystem credentials, API keys, tokens, cookies, or private URLs;
- no logs or exception traces that include protected rows; and
- every aggregate checked against the approved disclosure rule.

The public figure build consumes already-approved summaries and cannot reconstruct protected
candidate-level data.

## Code organization on RCC

Keep protected data and results in their existing approved production roots. Mirror only the reviewed
Git checkout into a separate `code_release/haplotype-reproducibility/` directory. Do not reorganize
or rename completed production directories, because they provide the path/job provenance for the
paper.

See [`dcc_rcc_layout.md`](dcc_rcc_layout.md) for the production-to-release mapping. Before copying a
new code release to RCC, confirm that the Git commit matches the DCC release mirror and that the copy
contains code/documentation only.
