#!/usr/bin/env python3
"""AlphaGenome contrasts for 1000 Genomes-fitted high-probability haplotypes.

The public-panel top-haplotype lists are paired by fitted rank under X0=0 and
X0=1.  Each AlphaGenome response is summarized over complete gene-level exon
unions and, optionally, terminal-exon tails.  Full prediction arrays are not
retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
from alphagenome.models import dna_client

from build_rosmap_top5_alphagenome_v61 import (
    AD_BIOSAMPLES,
    TARGET_LEN,
    apply_alt_variants,
    fetch_reference,
    load_terminal_tails,
    summarize_pair as summarize_tail_pair,
    windows_for_interval,
)
from build_total_transcript_alphagenome_v65 import (
    genes_for_window,
    load_exon_unions,
    predict_pair,
    summarize_total_pair,
)


DATASET = "1000 Genomes unrelated EUR"


def load_onekg_top10(
    root: Path, max_rank: int = 10
) -> tuple[pd.DataFrame, dict[tuple[str, str, int, int], dict]]:
    rows: list[dict] = []
    lookup: dict[tuple[str, str, int, int], dict] = {}
    for arm in ("armA", "armB"):
        for state in (0, 1):
            path = root / f"top10_contrast_1kg_{arm}_lead{state}_v60.json"
            source = json.load(open(path))
            for locus_row in source["per_locus"]:
                locus = locus_row["locus"]
                for haplotype in locus_row["top10"][:max_rank]:
                    rank = int(haplotype["rank"])
                    record = {
                        "dataset": DATASET,
                        "arm": arm,
                        "locus": locus,
                        "lead_state": state,
                        "rank": rank,
                        "q1_probability": float(haplotype["q1_probability"]),
                        "q1_log_probability": float(haplotype["q1_log_probability"]),
                        "empirical_probability": float(
                            haplotype["empirical_probability_given_lead_state"]
                        ),
                        "empirical_count": int(
                            haplotype["empirical_count_given_lead_state"]
                        ),
                        "n_alternate": int(haplotype["n_alternate"]),
                        "packed_little_endian_hex": haplotype[
                            "packed_little_endian_hex"
                        ],
                        "alternate_variant_ids": haplotype["alternate_variant_ids"],
                    }
                    rows.append(record)
                    lookup[(arm, locus, state, rank)] = record
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No 1000 Genomes haplotypes were loaded")
    return manifest, lookup


def relabel(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["dataset"] = DATASET
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--only", help="Optional locus identifier for a smoke test")
    parser.add_argument("--min-rank", type=int, default=1)
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--also-terminal", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_rank <= args.max_rank <= 10:
        raise ValueError("Ranks must satisfy 1 <= min-rank <= max-rank <= 10")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    terminal_checkpoints = args.output_dir / "terminal_checkpoints"
    if args.also_terminal:
        terminal_checkpoints.mkdir(exist_ok=True)
    reference_cache = args.output_dir / "reference_cache"

    manifest, lookup = load_onekg_top10(args.root, args.max_rank)
    manifest.to_parquet(
        args.output_dir / "onekg_top10_haplotype_manifest_v68.parquet", index=False
    )
    manifest.drop(
        columns=["alternate_variant_ids", "packed_little_endian_hex"]
    ).to_csv(
        args.output_dir / "onekg_top10_haplotype_index_v68.tsv",
        sep="\t",
        index=False,
    )

    genes_by_chrom = load_exon_unions(args.gtf)
    tails_by_chrom = load_terminal_tails(args.gtf) if args.also_terminal else None
    api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAGENOME_API_KEY is not set")
    model = dna_client.create(api_key)
    metadata = model.output_metadata(dna_client.Organism.HOMO_SAPIENS).rna_seq
    missing = sorted(
        set(AD_BIOSAMPLES) - set(metadata.biosample_name.astype(str))
    )
    if missing:
        raise RuntimeError(f"Requested AlphaGenome biosamples missing: {missing}")
    ontology_terms = sorted(
        set(
            metadata.loc[
                metadata.biosample_name.astype(str).isin(AD_BIOSAMPLES),
                "ontology_curie",
            ].astype(str)
        )
    )

    loci = sorted({(arm, locus) for arm, locus, _, _ in lookup})
    if args.only:
        loci = [(arm, locus) for arm, locus in loci if locus == args.only]
    if not loci:
        raise ValueError(f"No locus matched --only={args.only!r}")

    for arm, locus in loci:
        output_path = checkpoints / f"{arm}_{locus}.parquet"
        terminal_path = terminal_checkpoints / f"{arm}_{locus}.parquet"
        if output_path.exists() and (
            not args.also_terminal or terminal_path.exists()
        ):
            continue

        chrom, start0, end0, reference = fetch_reference(reference_cache, locus)
        genes = genes_for_window(genes_by_chrom, chrom, start0, end0)
        tails = (
            windows_for_interval(tails_by_chrom, chrom, start0, end0)
            if args.also_terminal
            else None
        )
        ranks0 = {
            rank
            for a, loc, state, rank in lookup
            if a == arm and loc == locus and state == 0
        }
        ranks1 = {
            rank
            for a, loc, state, rank in lookup
            if a == arm and loc == locus and state == 1
        }
        ranks = sorted(
            rank
            for rank in ranks0 & ranks1
            if args.min_rank <= rank <= args.max_rank
        )
        if not ranks:
            print(
                f"skipped {arm} {locus}: no paired ranks in "
                f"{args.min_rank}--{args.max_rank}",
                flush=True,
            )
            continue

        total_frames: list[pd.DataFrame] = []
        terminal_frames: list[pd.DataFrame] = []
        for rank in ranks:
            record0 = lookup[(arm, locus, 0, rank)]
            record1 = lookup[(arm, locus, 1, rank)]
            sequence0 = apply_alt_variants(
                reference, start0, record0["alternate_variant_ids"]
            )
            sequence1 = apply_alt_variants(
                reference, start0, record1["alternate_variant_ids"]
            )
            outputs = predict_pair(model, sequence0, sequence1, ontology_terms)
            hash0 = hashlib.sha256(sequence0.encode()).hexdigest()
            hash1 = hashlib.sha256(sequence1.encode()).hexdigest()
            total_frames.append(
                relabel(
                    summarize_total_pair(
                        outputs[0],
                        outputs[1],
                        genes,
                        arm,
                        locus,
                        rank,
                        hash0,
                        hash1,
                    )
                )
            )
            if args.also_terminal:
                terminal_frames.append(
                    relabel(
                        summarize_tail_pair(
                            outputs[0],
                            outputs[1],
                            tails,
                            arm,
                            locus,
                            rank,
                            hash0,
                            hash1,
                        )
                    )
                )
            time.sleep(1)

        pd.concat(total_frames, ignore_index=True).to_parquet(
            output_path, index=False
        )
        if args.also_terminal:
            pd.concat(terminal_frames, ignore_index=True).to_parquet(
                terminal_path, index=False
            )
        print(
            f"completed {arm} {locus}: {len(ranks)} ranks, {len(genes)} genes",
            flush=True,
        )

    total_paths = sorted(checkpoints.glob("*.parquet"))
    if not total_paths:
        raise RuntimeError("No complete-exon-union checkpoints were produced")
    combined = pd.concat(
        [pd.read_parquet(path) for path in total_paths], ignore_index=True
    )
    combined.to_parquet(
        args.output_dir / "onekg_fitted_total_top10_gene_changes_v68.parquet",
        index=False,
    )
    terminal_combined = None
    if args.also_terminal:
        terminal_paths = sorted(terminal_checkpoints.glob("*.parquet"))
        if len(terminal_paths) != len(total_paths):
            raise RuntimeError("Total and terminal checkpoint counts differ")
        terminal_combined = pd.concat(
            [pd.read_parquet(path) for path in terminal_paths], ignore_index=True
        )
        terminal_combined.to_parquet(
            args.output_dir
            / "onekg_fitted_terminal_top10_gene_changes_v68.parquet",
            index=False,
        )

    audit = {
        "script": Path(__file__).name,
        "dataset": DATASET,
        "target_length": TARGET_LEN,
        "rank_range": [args.min_rank, args.max_rank],
        "summary": (
            "mean RNA-seq signal over complete gene-level exon unions; "
            "introns excluded"
        ),
        "terminal_summary_also_written": bool(args.also_terminal),
        "n_manifest_rows": int(len(manifest)),
        "n_arm_loci": int(
            combined[["arm", "locus"]].drop_duplicates().shape[0]
        ),
        "n_rank_pairs": int(
            combined[["arm", "locus", "rank"]].drop_duplicates().shape[0]
        ),
        "n_rows_total": int(len(combined)),
        "n_rows_terminal": (
            int(len(terminal_combined)) if terminal_combined is not None else 0
        ),
        "n_genes": int(combined.gene_name.nunique()),
        "n_unique_sequence_hashes": int(
            pd.concat(
                [
                    combined["lead0_sequence_sha256"],
                    combined["lead1_sequence_sha256"],
                ]
            ).nunique()
        ),
    }
    (args.output_dir / "audit_onekg_v68.json").write_text(
        json.dumps(audit, indent=2)
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
