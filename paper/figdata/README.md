# Released figure and table inputs

This directory contains two kinds of files:

1. disclosure-safe aggregate artifacts (`.json`, `.csv`, `.npz`, `.tsv`); and
2. deterministic builders (`make_*.py`) that turn those artifacts into manuscript displays.

The aggregate files are immutable release inputs covered by `../../manifests/sha256.json`. Do not
edit a number in place. If an upstream result changes, rerun the versioned analysis, disclosure-review
the replacement artifact, update provenance and checksums, and rebuild the affected display.

Run all active builders through:

```bash
make figures
```

Do not invoke similarly named analysis scripts in this folder as an upstream rerun. The copies of
`compare_single_lead_enrichment_v64.py` and `compare_public_baseline_enrichment_v75.py` are retained
to bind released figure artifacts to the exact production source. The reader-facing upstream copies
are under `../../analysis/benchmark/`.

`../../manifests/result_map.csv` is the authoritative panel-level map from display to builder,
locked inputs, upstream analysis, job IDs, and access boundary.

