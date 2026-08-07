# Paper-output layer

This directory converts released aggregate artifacts into manuscript figures and tables. It does
not rerun genotype processing, model fitting, AlphaGenome, or hypothesis-level benchmarking.

## Layout

- `figdata/`: locked disclosure-safe JSON/CSV/NPZ inputs and their display builders;
- `main/`: generated main-paper figures and Table 1;
- `supp/`: generated supplementary figures and tables;
- `ranking-generated/`: temporary output from the combined simulation figure builder.

Run all active builders in the correct order with:

```bash
make figures
```

`scripts/build_paper_outputs.py` defines the ordered build. `manifests/result_map.csv` identifies the
builder, locked inputs, upstream analysis, job IDs, and access classification for every display.

Generated outputs are ignored by Git because `reference/` contains the submitted comparison copies.
Numerical assertions and input checksums, rather than byte-identical PDF files, are the release
gates.
