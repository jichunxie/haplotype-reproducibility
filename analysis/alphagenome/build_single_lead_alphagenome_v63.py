#!/usr/bin/env python3
"""AlphaGenome RNA-seq contrasts for a GRCh38 single-lead-variant flip.

Runs the same loci, sequence length, RNA tracks, gene windows and terminal-exon
summary used by v61. Each unique lead is predicted once with REF and once with
ALT; arm labels are attached afterward from the protected v60 manifests.
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
    AD_BIOSAMPLES, apply_alt_variants, fetch_reference, load_terminal_tails,
    parse_locus, summarize_pair, windows_for_interval,
)


def arm_membership(root: Path) -> pd.DataFrame:
    rows = []
    for arm in ("armA", "armB"):
        src = json.load(open(root / f"top10_contrast_rosmap_{arm}_lead0_v60.json"))
        rows.extend({"arm": arm, "locus": r["locus"]} for r in src["per_locus"])
    out = pd.DataFrame(rows).drop_duplicates()
    if out.duplicated(["arm", "locus"]).any():
        raise ValueError("duplicate arm-locus membership")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--gtf", type=Path, required=True)
    ap.add_argument("--only", help="Optional locus smoke-test selector")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    cache = args.output_dir / "reference_cache"
    membership = arm_membership(args.root)
    loci = sorted(membership.locus.unique())
    if args.only:
        loci = [x for x in loci if x == args.only]

    tails = load_terminal_tails(args.gtf)
    api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAGENOME_API_KEY is not set")
    model = dna_client.create(api_key)
    meta = model.output_metadata(dna_client.Organism.HOMO_SAPIENS).rna_seq
    missing = sorted(set(AD_BIOSAMPLES) - set(meta.biosample_name.astype(str)))
    if missing:
        raise RuntimeError(f"requested biosamples missing: {missing}")
    ontology_terms = sorted(set(meta.loc[
        meta.biosample_name.astype(str).isin(AD_BIOSAMPLES), "ontology_curie"
    ].astype(str)))

    for locus in loci:
        path = checkpoints / f"{locus}.parquet"
        if path.exists():
            continue
        chrom, start0, end0, ref_seq = fetch_reference(cache, locus)
        _, pos, ref, alt = parse_locus(locus)
        lead_id = f"{chrom.removeprefix('chr')}:{pos}:{ref}:{alt}"
        alt_seq = apply_alt_variants(ref_seq, start0, [lead_id])
        if ref_seq == alt_seq:
            raise ValueError(f"lead flip made no change: {locus}")
        windows = windows_for_interval(tails, chrom, start0, end0)
        outputs = model.predict_sequences(
            [ref_seq, alt_seq], organism=dna_client.Organism.HOMO_SAPIENS,
            requested_outputs=[dna_client.OutputType.RNA_SEQ],
            ontology_terms=ontology_terms, progress_bar=False, max_workers=2,
        )
        frame = summarize_pair(
            outputs[0], outputs[1], windows, "single_lead", locus, 1,
            hashlib.sha256(ref_seq.encode()).hexdigest(),
            hashlib.sha256(alt_seq.encode()).hexdigest(),
        )
        frame.to_parquet(path, index=False)
        print(f"completed {locus}: {len(frame)} summaries", flush=True)
        time.sleep(1)

    paths = sorted(checkpoints.glob("*.parquet")) if not args.only else [checkpoints / f"{x}.parquet" for x in loci]
    unique = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    unique = unique.drop(columns="arm")
    # A locus can belong to both arms, while each locus has many gene-track rows.
    # The deliberate many-to-many join replicates the unique prediction summaries
    # once for each arm membership; it does not duplicate any API call.
    combined = membership.merge(unique, on="locus", how="inner", validate="many_to_many")
    expected = sum(int(membership.locus.eq(x).sum()) * int(unique.locus.eq(x).sum()) for x in loci)
    if len(combined) != expected:
        raise ValueError((len(combined), expected))
    combined.to_parquet(args.output_dir / "single_lead_alphagenome_gene_changes_v63.parquet", index=False)
    audit = {
        "script": Path(__file__).name,
        "definition": "GRCh38 sequence with only the lead changed REF to ALT",
        "n_unique_loci": int(unique.locus.nunique()),
        "n_arm_loci": int(membership.shape[0]),
        "n_gene_track_contrasts": int(len(combined)),
        "biosample_names": AD_BIOSAMPLES,
    }
    (args.output_dir / "single_lead_audit_v63.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
