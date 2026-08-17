#!/usr/bin/env python
"""Build Supplementary Table 1 from the unfactored-correlation audit.

The table reports whether each one-factor constructed haplotype is the
highest-probability candidate when evaluated under the full working
correlation matrix. Conditional probabilities come from cond_probs_v40.json;
gaps and integration diagnostics come from verify_sparse_v33.json. The latter
retains a legacy production filename but the table is not a sparse-PCA
comparison.

This deterministic builder reads and formats released artifacts only.

Run from the repository root:
    python paper/figdata/make_supptable_ST1.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DOC = HERE.parent                  # figdata -> paper
OUT = DOC / "supp" / "supp-tables" / "ST1_verify_mode.tex"

v33 = json.load(open(HERE / "verify_sparse_v33.json"))
rows = sorted(v33["per_locus"], key=lambda r: -r["K"])

# v33 stores the joint rectangle probability Pr(X=x, X_0=1). v40 supplies
# the normalized conditional value. Probability gaps are unchanged by the
# locus-specific normalization and are read directly from v33.
cond = {r["locus"]: r for r in json.load(open(HERE / "cond_probs_v40.json"))["per_locus"]}
assert all(r["locus"] in cond for r in rows), "cond_probs_v40.json does not cover every locus"

assert len(rows) == 27, f"expected 27 loci, got {len(rows)}"
assert v33["config"]["seeds"] == [11, 2027] and v33["config"]["eig_floors"] == [1e-06, 0.0001]

tick, cross = r"\checkmark", r"$\times$"
lines = [
    r"\begin{longtable}{l r c r r r c}",
    r"\caption{\textbf{Per-locus validation of the one-factor candidate against the unfactored "
    r"working correlation matrix $\mathbf{R}$ in 1000 Genomes Arm A.} The top candidate is selected "
    r"under the one-factor model and then evaluated under the Gaussian copula defined by the full "
    r"working matrix. Thus the comparison is not evaluated under the same low-rank approximation "
    r"used to select the candidate. Here $p=k+1$ is the number of variants and "
    r"$\Pr(\hat{x}\mid\mathbf{R},X_0=1)$ is the candidate's conditional probability, estimated by "
    r"GHK. \emph{Argmax} indicates that it has the largest probability among the evaluated candidate "
    r"set, not necessarily among all $2^k$ configurations. \emph{Gap} is its log-probability lead "
    r"over the runner-up in nats. \emph{Gap/SE} divides that lead by the combined relative standard "
    r"error of the two probability estimates, treating them as independent; this is conservative "
    r"because common random numbers induce positive correlation. \emph{Genz} marks the $17$ loci "
    r"small enough for an independent Genz calculation, all of which agree on the argmax. Every "
    r"verdict is identical across two random-number seeds and two eigenvalue floors.}"
    r"\label{tab:supp-verify-mode}\\",
    r"\toprule",
    r"Lead locus & $p$ & argmax & $\Pr(\hat{x}\mid\mathbf{R},X_0{=}1)$ & gap & gap/SE & Genz \\",
    r"\midrule",
    r"\endfirsthead",
    r"\toprule",
    r"Lead locus & $p$ & argmax & $\Pr(\hat{x}\mid\mathbf{R},X_0{=}1)$ & gap & gap/SE & Genz \\",
    r"\midrule",
    r"\endhead",
    r"\bottomrule",
    r"\endfoot",
]

n_argmax = n_genz = n_stable = 0
for r in rows:
    am = bool(r["true_R_argmax_is_dense_top1"])
    n_argmax += am
    n_stable += bool(r["verdict_stable_across_seeds_and_floors"])
    gz = r["genz_cross_check"]
    if gz:
        n_genz += gz["same_argmax"]
    gap = "--" if r["true_R_logp_gap"] is None else f"{r['true_R_logp_gap']:.2f}"
    gse = "--" if not r.get("true_R_gap_over_se") else f"{r['true_R_gap_over_se']:.0f}"
    lines.append(
        f"\\texttt{{{r['locus'].replace('_', chr(92)+'_')}}} & {r['K']} & "
        f"{tick if am else cross} & {cond[r['locus']]['true_R_p_top_cond']:.3f} & {gap} & {gse} & "
        f"{tick if gz and gz['same_argmax'] else ('--' if not gz else cross)} \\\\"
    )
lines.append(r"\end{longtable}")

# The table's own claims, asserted rather than eyeballed.
assert n_argmax == 27, f"the mode is the true-R argmax at only {n_argmax} of 27 loci"
assert n_genz == 17, f"Genz agrees at only {n_genz} of the 17 loci it covers"
assert n_stable == 27, f"verdict is seed/floor-stable at only {n_stable} of 27 loci"
assert all(0.0 < cond[r["locus"]]["true_R_p_top_cond"] <= 1.0 + 1e-9 for r in rows), \
    "a conditional probability is outside (0, 1] -- the p_0 normalisation is wrong"
assert all(r["R0"] == r["K"] for r in rows), \
    "a locus has R0 < p -- the sparse/dense distinction would no longer be vacuous, so retiring " \
    "sparse PCA from the paper would need revisiting (see docs/development_history.md)"

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print("wrote", OUT)
print(f"  27 rows; true-R argmax at {n_argmax}/27; Genz agrees at {n_genz}/17; "
      f"seed/floor-stable at {n_stable}/27")
