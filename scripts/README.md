# Release drivers and audits

These scripts operate only on files already included in the public repository.

| Script | Purpose |
|---|---|
| `verify_checksums.py` | Verify every immutable path and SHA-256 digest in `manifests/sha256.json`. |
| `audit_results.py` | Assert the locked numerical claims used in the manuscript. |
| `build_paper_outputs.py` | Run every active figure/table builder in manuscript order and verify that all outputs exist. |
| `clean_generated.py` | Remove only the generated output paths listed in `.gitignore`. |

The normal interface is the repository Makefile:

```bash
make verify
make figures
make test
make clean
```

`make clean` does not remove released aggregate artifacts or submitted reference outputs.

