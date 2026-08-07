# DCC/RCC compute and release layout

The original analysis directories are preserved in place because their paths, logs, job IDs, and
artifacts form the production record. They should not be renamed or reorganized destructively.
The reader-oriented reproducibility checkout is a separate code-only release mirror.

Exact institutional filesystem paths are intentionally not published here. Authorized group
members should use the private cluster index maintained beside the production project.

## Release mirror

Maintain one clean Git checkout on each compute environment:

```text
<public-release-root>/haplotype-reproducibility/
```

The checkout mirrors the public GitHub repository and contains no protected donor-level data. Use
it to read the canonical flow, run public verification, or prepare a release. Do not write
production results into this Git checkout.

## DCC production roles

Keep the existing versioned production roots for:

- the shared estimator and public-panel fitting;
- 1000 Genomes diagnostics and conditional-haplotype construction;
- the final v77 known-truth simulation; and
- public-panel AlphaGenome and approved aggregate benchmark steps.

Outputs and logs remain beside the versioned code that produced them. The original directories are
the record of what ran; the release mirror is the reader-oriented presentation.

## RCC protected roles

Keep the existing approved production roots for:

- ROS/MAP locus extraction and q=1 fitting;
- protected top-haplotype identities and empirical counts;
- protected AlphaGenome prediction rows; and
- protected scored eQTL benchmark hypotheses.

Do not copy row-level content into DCC or GitHub. Mirror only the reviewed code checkout under a
separate code-release directory in the approved project.

The protected flow is:

```text
approved ROS/MAP pfiles
    -> fit_rosmap_locus_v51.py
    -> build_top10_empirical_v59.py
    -> protected AlphaGenome gene-summary scripts
    -> protected benchmark scripts
    -> disclosure review
    -> approved aggregate JSON/CSV only
```

## Safe synchronization rule

1. Make and review documentation/comment-only changes in the public release checkout.
2. Run `make verify` and `make figures` in the recorded environment.
3. Synchronize the reviewed Git commit to the DCC release mirror.
4. Synchronize the same commit to the RCC code-only mirror from an approved VPN-connected session.
5. Never synchronize RCC data, checkpoints, logs containing row-level results, or credentials back
   to DCC or GitHub.

The Git commit, not an ad hoc directory timestamp, identifies the synchronized release. Cluster
operators should record the exact private source and mirror paths in the internal project index,
not in this public repository.
