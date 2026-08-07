# Stage 4: AlphaGenome sequence-to-function analysis

These scripts construct paired lead-state sequences, query AlphaGenome RNA predictions, and reduce
the responses to gene-level summaries. API calls require AlphaGenome client version 0.4.0 and an
`ALPHAGENOME_API_KEY` environment variable.

## Reading order

1. `build_rosmap_top5_alphagenome_v61.py` — shared sequence-writing, reference-cache, GTF, and
   rank-pair helpers; protected top-five entry point.
2. `build_total_transcript_alphagenome_v65.py` — complete-exon-union aggregation used by the main
   benchmark.
3. `build_onekg_total_transcript_alphagenome_v68.py` — public-panel top-ten application of the same
   gene summary.
4. `build_single_lead_alphagenome_v63.py` — lead-only reference-background comparator.
5. `combine_top10_predictions_v66.py` — strict rank 1--10 merge utility.

## Files and boundaries

| Script | Environment | Output |
|---|---|---|
| `build_rosmap_top5_alphagenome_v61.py` | protected RCC | Protected paired rank 1--5 hypothesis rows and checkpoints. |
| `build_total_transcript_alphagenome_v65.py` | DCC/RCC | Complete-exon-union gene summaries for fitted or single-lead inputs. |
| `build_onekg_total_transcript_alphagenome_v68.py` | DCC | Public-panel fitted ranks 1--10, with optional terminal-exon sensitivity output. |
| `build_single_lead_alphagenome_v63.py` | DCC | GRCh38 lead-only reference/alternate contrasts. |
| `combine_top10_predictions_v66.py` | protected environment | Verified concatenation of ranks 1--5 and 6--10. |

## Code chunks

The query scripts share the same conceptual blocks:

1. parse locus and haplotype identifiers;
2. fetch and checksum the 1,048,576-bp GRCh38 reference window;
3. write only the requested lead and partner alleles, checking reference bases;
4. select the predefined RNA biosamples from AlphaGenome metadata;
5. query lead-state pairs and checkpoint each locus/rank;
6. merge overlapping exons once and summarize complete gene exon unions; and
7. write metadata sufficient to audit client version, tracks, sequence hashes, and coverage.

The hosted service did not provide a persistent server-model identifier. Re-querying can therefore
verify the client-side workflow but cannot guarantee bitwise equality to the original server output.

