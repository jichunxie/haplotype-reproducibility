#!/usr/bin/env python3
"""AlphaGenome contrasts summarized over complete gene-level exon unions.

Runs either protected fitted rank-matched haplotypes or public-reference
single-lead flips. Full prediction arrays are not retained. A gene is included
only when its complete annotated exon union lies inside the 1,048,576-bp input
window; overlapping exons are counted once and introns are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from alphagenome.models import dna_client

from build_rosmap_top5_alphagenome_v61 import (
    AD_BIOSAMPLES, TARGET_LEN, apply_alt_variants, fetch_reference, load_top5,
    load_terminal_tails, parse_gtf_attributes, parse_locus,
    summarize_pair as summarize_tail_pair, windows_for_interval,
)
from build_single_lead_alphagenome_v63 import arm_membership


def load_exon_unions(gtf_path: Path) -> dict[str, list[dict]]:
    cols = ["chrom", "source", "feature", "start1", "end1", "score", "strand", "frame", "attrs"]
    gtf = pd.read_csv(gtf_path, sep="\t", comment="#", names=cols, compression="gzip")
    gtf = gtf.loc[gtf.feature.eq("exon")].copy()
    attrs = gtf["attrs"].apply(parse_gtf_attributes)
    gtf["gene_id"] = attrs.map(lambda x: x.get("gene_id", ""))
    gtf["gene_name"] = attrs.map(lambda x: x.get("gene_name", ""))
    gtf["transcript_id"] = attrs.map(lambda x: x.get("transcript_id", ""))
    gtf["start0"] = gtf.start1.astype(int) - 1
    gtf["end0"] = gtf.end1.astype(int)

    def merge_intervals(frame: pd.DataFrame) -> list[tuple[int, int]]:
        merged = []
        for start, end in sorted(zip(frame.start0.astype(int), frame.end0.astype(int))):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return [(int(a), int(b)) for a, b in merged]

    by_chrom: dict[str, list[dict]] = {}
    for (chrom, gene_id), group in gtf.groupby(["chrom", "gene_id"], sort=False):
        intervals = merge_intervals(group)
        by_chrom.setdefault(str(chrom), []).append({
            "gene_id": str(gene_id), "gene_name": str(group.gene_name.iloc[0]),
            "transcript_ids": "|".join(sorted(set(group.transcript_id.astype(str)))),
            "strand": str(group.strand.iloc[0]), "intervals": intervals,
            "gene_start0": min(x[0] for x in intervals), "gene_end0": max(x[1] for x in intervals),
            "exonic_bp": sum(b - a for a, b in intervals),
        })
    return by_chrom


def genes_for_window(by_chrom: dict[str, list[dict]], chrom: str, start0: int, end0: int) -> list[dict]:
    genes = []
    for gene in by_chrom.get(chrom, []):
        if gene["gene_start0"] < start0 or gene["gene_end0"] > end0:
            continue
        rec = dict(gene)
        rec["idx_intervals"] = [(a - start0, b - start0) for a, b in gene["intervals"]]
        genes.append(rec)
    return genes


def summarize_total_pair(out0, out1, genes: list[dict], arm: str, locus: str, rank: int,
                         hash0: str, hash1: str) -> pd.DataFrame:
    td0, td1 = out0.rna_seq, out1.rna_seq
    if td0 is None or td1 is None:
        raise RuntimeError("AlphaGenome returned no RNA-seq output")
    m0, m1 = td0.metadata.reset_index(drop=True), td1.metadata.reset_index(drop=True)
    if not m0.equals(m1):
        raise RuntimeError("RNA-seq metadata differ between paired predictions")
    keep = m0.biosample_name.astype(str).isin(AD_BIOSAMPLES).to_numpy()
    meta = m0.loc[keep].reset_index(drop=True)
    v0, v1 = np.asarray(td0.values)[:, keep], np.asarray(td1.values)[:, keep]
    rows = []
    for gene in genes:
        total0 = np.zeros(v0.shape[1]); total1 = np.zeros(v1.shape[1]); n_bp = 0
        for a, b in gene["idx_intervals"]:
            total0 += np.nansum(v0[a:b, :], axis=0)
            total1 += np.nansum(v1[a:b, :], axis=0)
            n_bp += b - a
        p0, p1 = total0 / n_bp, total1 / n_bp
        for j, meta_row in meta.iterrows():
            rows.append({
                "dataset": "ROS/MAP", "arm": arm, "locus": locus, "rank": rank,
                "summary": "complete_gene_exon_union", "lead0_sequence_sha256": hash0,
                "lead1_sequence_sha256": hash1, "gene_id": gene["gene_id"],
                "gene_name": gene["gene_name"], "transcript_id": gene["transcript_ids"],
                "strand": gene["strand"], "gene_start0": gene["gene_start0"],
                "gene_end0": gene["gene_end0"], "exonic_bp": gene["exonic_bp"],
                "track_name": meta_row["name"], "track_strand": meta_row["strand"],
                "assay_title": meta_row["Assay title"], "ontology_curie": meta_row["ontology_curie"],
                "biosample_name": meta_row["biosample_name"],
                "pred_mean_lead0": float(p0[j]), "pred_mean_lead1": float(p1[j]),
                "delta_pred_mean": float(p1[j] - p0[j]),
                "log1p_delta_mean": float(np.log1p(p1[j]) - np.log1p(p0[j])),
            })
    return pd.DataFrame(rows)


def predict_pair(model, seq0: str, seq1: str, ontology_terms: list[str]):
    return model.predict_sequences(
        [seq0, seq1], organism=dna_client.Organism.HOMO_SAPIENS,
        requested_outputs=[dna_client.OutputType.RNA_SEQ], ontology_terms=ontology_terms,
        progress_bar=False, max_workers=2,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fitted", "single_lead"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--gtf", type=Path, required=True)
    ap.add_argument("--only", help="Optional locus identifier for a smoke test")
    ap.add_argument("--min-rank", type=int, default=1)
    ap.add_argument("--max-rank", type=int, default=5)
    ap.add_argument("--also-terminal", action="store_true")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    cache = args.output_dir / "reference_cache"
    genes_by_chrom = load_exon_unions(args.gtf)
    tails_by_chrom = load_terminal_tails(args.gtf) if args.also_terminal else None
    key = os.environ.get("ALPHAGENOME_API_KEY")
    if not key:
        raise RuntimeError("ALPHAGENOME_API_KEY is not set")
    model = dna_client.create(key)
    meta = model.output_metadata(dna_client.Organism.HOMO_SAPIENS).rna_seq
    ontology_terms = sorted(set(meta.loc[
        meta.biosample_name.astype(str).isin(AD_BIOSAMPLES), "ontology_curie"
    ].astype(str)))

    if args.mode == "fitted":
        _, lookup = load_top5(args.root, args.max_rank)
        loci = sorted({(a, l) for a, l, _, _ in lookup})
        if args.only:
            loci = [(a, l) for a, l in loci if l == args.only]
        for arm, locus in loci:
            path = checkpoints / f"{arm}_{locus}.parquet"
            tail_path = args.output_dir / "terminal_checkpoints" / f"{arm}_{locus}.parquet"
            if args.also_terminal:
                tail_path.parent.mkdir(exist_ok=True)
            if path.exists():
                continue
            chrom, start0, end0, ref = fetch_reference(cache, locus)
            genes = genes_for_window(genes_by_chrom, chrom, start0, end0)
            tails = windows_for_interval(tails_by_chrom, chrom, start0, end0) if args.also_terminal else None
            frames, tail_frames = [], []
            ranks = sorted(set(r for a, l, s, r in lookup if a == arm and l == locus and s == 0)
                           & set(r for a, l, s, r in lookup if a == arm and l == locus and s == 1))
            ranks = [rank for rank in ranks if args.min_rank <= rank <= args.max_rank]
            if not ranks:
                print(f"skipped {arm} {locus}: no ranks in {args.min_rank}--{args.max_rank}",
                      flush=True)
                continue
            for rank in ranks:
                rec0, rec1 = lookup[(arm, locus, 0, rank)], lookup[(arm, locus, 1, rank)]
                seq0 = apply_alt_variants(ref, start0, rec0["alternate_variant_ids"])
                seq1 = apply_alt_variants(ref, start0, rec1["alternate_variant_ids"])
                out = predict_pair(model, seq0, seq1, ontology_terms)
                hash0, hash1 = hashlib.sha256(seq0.encode()).hexdigest(), hashlib.sha256(seq1.encode()).hexdigest()
                frames.append(summarize_total_pair(out[0], out[1], genes, arm, locus, rank, hash0, hash1))
                if args.also_terminal:
                    tail_frames.append(summarize_tail_pair(out[0], out[1], tails, arm, locus,
                                                           rank, hash0, hash1))
                time.sleep(1)
            pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
            if args.also_terminal:
                pd.concat(tail_frames, ignore_index=True).to_parquet(tail_path, index=False)
            print(f"completed {arm} {locus}: {len(ranks)} ranks, {len(genes)} genes", flush=True)
        output_name = "fitted_total_transcript_gene_changes_v65.parquet"
    else:
        membership = arm_membership(args.root)
        loci = sorted(membership.locus.unique())
        if args.only:
            loci = [locus for locus in loci if locus == args.only]
            membership = membership.loc[membership.locus.isin(loci)].copy()
        for locus in loci:
            path = checkpoints / f"{locus}.parquet"
            if path.exists():
                continue
            chrom, start0, end0, refseq = fetch_reference(cache, locus)
            _, pos, ref, alt = parse_locus(locus)
            altseq = apply_alt_variants(refseq, start0, [f"{chrom.removeprefix('chr')}:{pos}:{ref}:{alt}"])
            genes = genes_for_window(genes_by_chrom, chrom, start0, end0)
            out = predict_pair(model, refseq, altseq, ontology_terms)
            frame = summarize_pair(out[0], out[1], genes, "single_lead", locus, 1,
                                   hashlib.sha256(refseq.encode()).hexdigest(),
                                   hashlib.sha256(altseq.encode()).hexdigest())
            frame.to_parquet(path, index=False)
            print(f"completed {locus}: {len(genes)} genes", flush=True)
            time.sleep(1)
        unique = pd.concat([pd.read_parquet(p) for p in sorted(checkpoints.glob("*.parquet"))],
                           ignore_index=True).drop(columns="arm")
        combined = membership.merge(unique, on="locus", how="inner", validate="many_to_many")
        combined.to_parquet(args.output_dir / "single_total_transcript_gene_changes_v65.parquet",
                            index=False)
        output_name = None

    if output_name:
        combined = pd.concat([pd.read_parquet(p) for p in sorted(checkpoints.glob("*.parquet"))],
                             ignore_index=True)
        combined.to_parquet(args.output_dir / output_name, index=False)
        if args.also_terminal:
            tail_combined = pd.concat([pd.read_parquet(p) for p in
                                       sorted((args.output_dir / "terminal_checkpoints").glob("*.parquet"))],
                                      ignore_index=True)
            tail_combined.to_parquet(args.output_dir / "fitted_terminal_exon_gene_changes_v65.parquet",
                                     index=False)
    audit = {"script": Path(__file__).name, "mode": args.mode, "target_length": TARGET_LEN,
             "summary": "mean RNA-seq signal over complete gene-level exon union; introns excluded",
             "n_rows": int(len(combined)), "n_arm_loci": int(combined[["arm", "locus"]].drop_duplicates().shape[0]),
             "n_genes": int(combined.gene_name.nunique())}
    (args.output_dir / f"audit_{args.mode}_v65.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
