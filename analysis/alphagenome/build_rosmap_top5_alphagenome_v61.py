#!/usr/bin/env python3
"""AlphaGenome RNA-seq contrasts for protected ROS/MAP fitted top-five haplotypes.

This script must run inside the approved RCC project. It reads the four protected
v60 top-ten files, retains ranks 1--5 under each lead state, builds 1,048,576-bp
GRCh38 sequences, and predicts RNA-seq tracks with AlphaGenome. The estimand is
the rank-matched contrast X0=1 minus X0=0 within each arm and locus. Only
terminal-exon-tail summaries are retained; full prediction arrays are not saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from alphagenome.models import dna_client

TARGET_LEN = 1_048_576
TAIL_NT = 1_000
AD_BIOSAMPLES = [
    "astrocyte", "glutamatergic neuron", "endothelial cell", "neural cell",
    "neural progenitor cell", "neuronal stem cell", "neurosphere", "motor neuron",
    "Purkinje cell", "CD14-positive monocyte", "mononuclear cell",
    "peripheral blood mononuclear cell", "dorsolateral prefrontal cortex",
    "frontal cortex", "temporal lobe", "parietal lobe", "occipital lobe",
    "cerebellum", "cerebellar hemisphere", "caudate nucleus", "putamen",
    "amygdala", "nucleus accumbens", "hypothalamus", "Ammon's horn",
    "substantia nigra", "anterior cingulate cortex", "brain", "H4", "U-87 MG",
    "SK-N-SH", "SK-N-DZ", "BE2C", "PFSK-1", "A172", "WTC11", "H1", "H7",
    "H9", "HUES64",
]


def parse_locus(locus: str) -> tuple[str, int, str, str]:
    chrom, pos, ref, alt = locus.split("_", 3)
    return f"chr{chrom}", int(pos), ref, alt


def interval_for_locus(locus: str) -> tuple[str, int, int]:
    chrom, pos, _, _ = parse_locus(locus)
    start1 = max(1, pos - TARGET_LEN // 2)
    start0 = start1 - 1
    return chrom, start0, start0 + TARGET_LEN


def fetch_reference(cache_dir: Path, locus: str) -> tuple[str, int, int, str]:
    chrom, start0, end0 = interval_for_locus(locus)
    path = cache_dir / f"{locus}.json"
    if path.exists():
        obj = json.load(open(path))
    else:
        url = (
            "https://api.genome.ucsc.edu/getData/sequence?genome=hg38;"
            f"chrom={chrom};start={start0};end={end0}"
        )
        with urllib.request.urlopen(url, timeout=180) as response:
            obj = json.load(response)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))
    seq = obj["dna"].upper()
    assert len(seq) == TARGET_LEN, (locus, len(seq))
    return chrom, start0, end0, seq


def apply_alt_variants(ref_seq: str, start0: int, alt_ids: list[str]) -> str:
    seq = bytearray(ref_seq.encode("ascii"))
    seen: set[tuple[int, str, str]] = set()
    for variant in alt_ids:
        chrom, pos, ref, alt = variant.split(":", 3)
        del chrom
        pos1 = int(pos)
        key = (pos1, ref, alt)
        if key in seen:
            continue
        seen.add(key)
        idx = pos1 - 1 - start0
        observed = bytes(seq[idx:idx + len(ref)]).decode("ascii")
        if observed != ref:
            raise ValueError(f"Reference mismatch at {variant}: observed {observed}")
        seq[idx:idx + len(ref)] = alt.encode("ascii")
    if len(seq) != TARGET_LEN:
        raise ValueError("Indels are not supported in this fixed-length top-haplotype run")
    return seq.decode("ascii")


def parse_gtf_attributes(text: str) -> dict[str, str]:
    return dict(re.findall(r'(\S+) "([^"]*)";', text))


def load_terminal_tails(gtf_path: Path) -> dict[str, pd.DataFrame]:
    cols = ["chrom", "source", "feature", "start1", "end1", "score", "strand", "frame", "attrs"]
    gtf = pd.read_csv(gtf_path, sep="\t", comment="#", names=cols, compression="gzip")
    gtf = gtf.loc[gtf.feature.eq("exon")].copy()
    attrs = gtf["attrs"].apply(parse_gtf_attributes)
    gtf["gene_id"] = attrs.map(lambda x: x.get("gene_id", ""))
    gtf["gene_name"] = attrs.map(lambda x: x.get("gene_name", ""))
    gtf["transcript_id"] = attrs.map(lambda x: x.get("transcript_id", ""))
    gtf["exon_number"] = pd.to_numeric(attrs.map(lambda x: x.get("exon_number")), errors="coerce")
    gtf["start0"] = gtf.start1.astype(int) - 1
    gtf["end0"] = gtf.end1.astype(int)

    def terminal(group: pd.DataFrame) -> pd.Series:
        strand = group.strand.iloc[0]
        if group.exon_number.notna().any():
            idx = group.exon_number.idxmax() if strand == "+" else group.exon_number.idxmin()
        else:
            idx = group.end0.idxmax() if strand == "+" else group.start0.idxmin()
        return group.loc[idx]

    tails = gtf.groupby("transcript_id", group_keys=False).apply(terminal, include_groups=False)
    tails = tails.reset_index(drop=False)
    tails["tail_start0"] = np.where(
        tails.strand.eq("+"), np.maximum(tails.end0 - TAIL_NT, tails.start0), tails.start0
    ).astype(int)
    tails["tail_end0"] = np.where(
        tails.strand.eq("+"), tails.end0, np.minimum(tails.start0 + TAIL_NT, tails.end0)
    ).astype(int)
    return {chrom: frame.reset_index(drop=True) for chrom, frame in tails.groupby("chrom")}


def windows_for_interval(tails_by_chrom: dict[str, pd.DataFrame], chrom: str,
                         start0: int, end0: int) -> pd.DataFrame:
    tails = tails_by_chrom.get(chrom, pd.DataFrame()).copy()
    tails = tails.loc[(tails.tail_end0 > start0) & (tails.tail_start0 < end0)].copy()
    tails["idx_start"] = (tails.tail_start0.clip(lower=start0) - start0).astype(int)
    tails["idx_end"] = (tails.tail_end0.clip(upper=end0) - start0).astype(int)
    return tails.loc[tails.idx_end > tails.idx_start].reset_index(drop=True)


def load_top5(root: Path, n_ranks: int = 5) -> tuple[pd.DataFrame, dict[tuple[str, str, int], dict]]:
    rows = []
    lookup = {}
    for arm in ("armA", "armB"):
        for state in (0, 1):
            src = json.load(open(root / f"top10_contrast_rosmap_{arm}_lead{state}_v60.json"))
            for locus_row in src["per_locus"]:
                locus = locus_row["locus"]
                for hap in locus_row["top10"][:n_ranks]:
                    rank = int(hap["rank"])
                    rec = {
                        "dataset": "ROS/MAP", "arm": arm, "locus": locus,
                        "lead_state": state, "rank": rank,
                        "q1_probability": hap["q1_probability"],
                        "q1_log_probability": hap["q1_log_probability"],
                        "empirical_probability": hap["empirical_probability_given_lead_state"],
                        "empirical_count": hap["empirical_count_given_lead_state"],
                        "n_alternate": hap["n_alternate"],
                        "packed_little_endian_hex": hap["packed_little_endian_hex"],
                        "alternate_variant_ids": hap["alternate_variant_ids"],
                    }
                    rows.append(rec)
                    lookup[(arm, locus, state, rank)] = rec
    return pd.DataFrame(rows), lookup


def summarize_pair(out0, out1, windows: pd.DataFrame, arm: str, locus: str, rank: int,
                   seq_hash0: str, seq_hash1: str) -> pd.DataFrame:
    td0, td1 = out0.rna_seq, out1.rna_seq
    if td0 is None or td1 is None:
        raise RuntimeError("AlphaGenome returned no RNA-seq output")
    m0, m1 = td0.metadata.reset_index(drop=True), td1.metadata.reset_index(drop=True)
    if not m0.equals(m1):
        raise RuntimeError("RNA-seq metadata differ between the paired predictions")
    keep = m0.biosample_name.astype(str).isin(AD_BIOSAMPLES).to_numpy()
    meta = m0.loc[keep].reset_index(drop=True)
    v0 = np.asarray(td0.values)[:, keep]
    v1 = np.asarray(td1.values)[:, keep]
    rows = []
    for w in windows.itertuples(index=False):
        p0 = np.nanmean(v0[w.idx_start:w.idx_end, :], axis=0)
        p1 = np.nanmean(v1[w.idx_start:w.idx_end, :], axis=0)
        for j, m in meta.iterrows():
            rows.append({
                "dataset": "ROS/MAP", "arm": arm, "locus": locus, "rank": rank,
                "contrast": "lead1_minus_lead0", "lead0_sequence_sha256": seq_hash0,
                "lead1_sequence_sha256": seq_hash1, "gene_id": w.gene_id,
                "gene_name": w.gene_name, "transcript_id": w.transcript_id,
                "strand": w.strand, "tail_start0": int(w.tail_start0),
                "tail_end0": int(w.tail_end0), "track_name": m["name"],
                "track_strand": m["strand"], "assay_title": m["Assay title"],
                "ontology_curie": m["ontology_curie"], "biosample_name": m["biosample_name"],
                "pred_mean_lead0": float(p0[j]), "pred_mean_lead1": float(p1[j]),
                "delta_pred_mean": float(p1[j] - p0[j]),
                "log1p_delta_mean": float(np.log1p(p1[j]) - np.log1p(p0[j])),
                "log2fc": float(np.log2((p1[j] + 1e-8) / (p0[j] + 1e-8))),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--gtf", type=Path, required=True)
    ap.add_argument("--only", help="Optional arm:locus smoke-test selector")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "reference_cache"
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    manifest, lookup = load_top5(args.root)
    manifest.to_parquet(args.output_dir / "rosmap_top5_haplotype_manifest_v61.parquet", index=False)
    manifest.drop(columns=["alternate_variant_ids", "packed_little_endian_hex"]).to_csv(
        args.output_dir / "rosmap_top5_haplotype_index_v61.tsv", sep="\t", index=False
    )
    tails_by_chrom = load_terminal_tails(args.gtf)

    api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAGENOME_API_KEY is not set")
    model = dna_client.create(api_key)
    rna_meta = model.output_metadata(dna_client.Organism.HOMO_SAPIENS).rna_seq
    missing = sorted(set(AD_BIOSAMPLES) - set(rna_meta.biosample_name.astype(str)))
    if missing:
        raise RuntimeError(f"Requested AlphaGenome biosamples missing: {missing}")
    ontology_terms = sorted(set(
        rna_meta.loc[rna_meta.biosample_name.astype(str).isin(AD_BIOSAMPLES), "ontology_curie"].astype(str)
    ))
    rna_meta.loc[rna_meta.biosample_name.astype(str).isin(AD_BIOSAMPLES)].to_parquet(
        args.output_dir / "alphagenome_rna_metadata_requested_v61.parquet", index=False
    )

    loci = sorted({(arm, locus) for arm, locus, _, _ in lookup})
    if args.only:
        only_arm, only_locus = args.only.split(":", 1)
        loci = [(a, l) for a, l in loci if a == only_arm and l == only_locus]
    completed = []
    for arm, locus in loci:
        out_path = checkpoint_dir / f"{arm}_{locus}.parquet"
        if out_path.exists():
            completed.append(str(out_path))
            continue
        chrom, start0, end0, ref_seq = fetch_reference(cache_dir, locus)
        windows = windows_for_interval(tails_by_chrom, chrom, start0, end0)
        if windows.empty:
            raise RuntimeError(f"No terminal-exon-tail windows for {arm} {locus}")
        pair_frames = []
        ranks = sorted(set(
            r for a, l, s, r in lookup if a == arm and l == locus and s == 0
        ) & set(
            r for a, l, s, r in lookup if a == arm and l == locus and s == 1
        ))
        for rank in ranks:
            rec0, rec1 = lookup[(arm, locus, 0, rank)], lookup[(arm, locus, 1, rank)]
            seq0 = apply_alt_variants(ref_seq, start0, rec0["alternate_variant_ids"])
            seq1 = apply_alt_variants(ref_seq, start0, rec1["alternate_variant_ids"])
            hash0 = hashlib.sha256(seq0.encode()).hexdigest()
            hash1 = hashlib.sha256(seq1.encode()).hexdigest()
            outputs = model.predict_sequences(
                [seq0, seq1], organism=dna_client.Organism.HOMO_SAPIENS,
                requested_outputs=[dna_client.OutputType.RNA_SEQ],
                ontology_terms=ontology_terms, progress_bar=False, max_workers=2,
            )
            pair_frames.append(summarize_pair(outputs[0], outputs[1], windows, arm, locus,
                                              rank, hash0, hash1))
            time.sleep(1)
        locus_df = pd.concat(pair_frames, ignore_index=True)
        locus_df.to_parquet(out_path, index=False)
        completed.append(str(out_path))
        print(f"completed {arm} {locus}: {len(ranks)} rank pairs, {len(locus_df)} summaries", flush=True)

    all_paths = sorted(checkpoint_dir.glob("*.parquet")) if not args.only else [Path(x) for x in completed]
    if not all_paths:
        print("preparation validation complete; no arm-locus matched --only", flush=True)
        return
    all_df = pd.concat([pd.read_parquet(p) for p in all_paths], ignore_index=True)
    all_df.to_parquet(args.output_dir / "rosmap_top5_alphagenome_gene_changes_v61.parquet", index=False)
    audit = {
        "script": Path(__file__).name, "definition": "rank-matched X0=1 minus X0=0",
        "target_length": TARGET_LEN, "tail_nt": TAIL_NT, "biosample_names": AD_BIOSAMPLES,
        "n_manifest_rows": int(len(manifest)), "n_arm_loci_completed": int(len(all_paths)),
        "n_gene_track_contrasts": int(len(all_df)),
        "n_unique_sequence_hashes": int(pd.concat([
            all_df["lead0_sequence_sha256"], all_df["lead1_sequence_sha256"]
        ]).nunique()),
    }
    (args.output_dir / "audit_v61.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
