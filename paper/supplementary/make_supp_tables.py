"""Supplementary tables for the ArchTree paper (Sadhana). Every number is read from the result CSVs.
usage: python3 make_supp_tables.py <outputs dir> <tables dir>
Each table is written as a LaTeX tabular fragment (tables/<name>.tex); captions live in supp.tex."""
import sys, os, json
import numpy as np, pandas as pd

R, T = sys.argv[1:3]
os.makedirs(T, exist_ok=True)
rd = lambda p: pd.read_csv(os.path.join(R, p))
DS = ["cifar10", "cifar100", "ImageNet16-120"]
DSL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "ImageNet16-120": "ImageNet16-120"}
DSS = {"cifar10": "C10", "cifar100": "C100", "ImageNet16-120": "IN16"}
BUD = [25, 50, 100, 200]

FIX = {"random_search": "Random search", "mcts_nas": "MCTS-style", "enas": "ENAS-style", "llm_nas": "LLM-NAS",
       "zico": "ZiCo-style", "rz_nas": "RZ-style ZC walk", "re": "RE", "re_zc_warmup": "RE + ZC warm-up",
       "local_search": "Local search", "archtree_v1": "S1 ArchTree-base", "archtree_zc": "S2 (S1 + supervised gate)",
       "archtree_llm": "S4 ArchTree-RMG", "archtree_v2": "S4 ArchTree-RMG"}
EGH = {"archtree_zc_unsup": "ArchTree-RMG", "archtree_v2": "ArchTree-RMG", "archtree_none": "ArchTree-RM (no gate)",
       "archtree_params": "ArchTree-RM + \\#params gate", "archtree_costpen": "ArchTree-RM + cost pen. ($\\lambda$=1)",
       "archtree_costpen025": "ArchTree-RM + cost pen. ($\\lambda$=0.25)", "re_zc_unsup": "RE + gate", "re_none": "RE",
       "re_zcwarm": "RE + ZC warm-up", "ls_zc_unsup": "LS + gate", "ls_none": "LS", "random_zc_unsup": "Random + gate",
       "random_none": "Random", "random_costpen": "Random + cost pen. ($\\lambda$=1)",
       "random_costpen025": "Random + cost pen. ($\\lambda$=0.25)"}
EGH_ORDER = ["archtree_zc_unsup", "archtree_none", "archtree_params", "archtree_costpen025", "archtree_costpen",
             "re_zc_unsup", "re_zcwarm", "re_none", "ls_zc_unsup", "ls_none", "random_zc_unsup", "random_none",
             "random_costpen025", "random_costpen"]
PROP = {"archtree": "ArchTree-RM", "random": "Random", "re": "RE", "ls": "LS", "enas": "ENAS-style", "mcts": "MCTS-style"}
GATE = {"none": "none", "params": "\\#params", "zc": "supervised", "zc_unsup": "label-free"}


def f(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "--"
    s_ = f"{x:.{d}f}"
    if s_.startswith("-") and float(s_) == 0: s_ = s_[1:]
    return s_.replace("-", "$-$")


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); mm = len(p); adj = np.empty(mm); run = 0
    for r_, i in enumerate(o): run = max(run, (mm - r_) * p[i]); adj[i] = min(1, run)
    return adj


def wtl_holm(p, diff):
    ph = holm(p); diff = np.asarray(diff)
    w = int(((ph < .05) & (diff > 0)).sum()); l = int(((ph < .05) & (diff < 0)).sum())
    return f"{w}/{len(ph) - w - l}/{l}"


def pm(m, s, d=2): return f"{f(m, d)} $\\pm$ {f(s, d)}"


def pfmt(p):
    if np.isnan(p): return "--"
    if p < 1e-3: return f"{p:.0e}".replace("e-0", "e-").replace("e-", "$\\times10^{-") + "}$" if False else "$<$0.001"
    return f"{p:.3f}"


def write(name, s):
    with open(os.path.join(T, name + ".tex"), "w") as fh: fh.write(s)


def tab(cols, header, rows, extra_top=""):
    s = "\\begin{tabular}{@{}" + cols + "@{}}\n\\toprule\n" + extra_top + header + "\\\\\n\\midrule\n"
    for r_ in rows:
        s += ("\\midrule\n" if r_ == "MID" else r_ + "\\\\\n")
    return s + "\\bottomrule\n\\end{tabular}\n"


def wtl(out):
    out = list(out)
    return f"{out.count('win')}/{out.count('tie')}/{out.count('loss')}"


# ---------------------------------------------------------------- S-regret fit (budget 200)
rf = rd("final/results/regret_fit.csv")
order = ["random_exact", "random_search", "mcts_nas", "enas", "llm_nas", "zico", "rz_nas", "re", "re_zc_warmup",
         "local_search", "archtree_v1", "archtree_zc", "archtree_llm", "flat_ucb_none[A7]", "flat_ucb_set[A7]", "archtree_llm[A7_trie]"]
lab = dict(FIX, **{"random_exact": "Uniform sampling (exact)", "flat_ucb_none[A7]": "Flat UCB, no dedup (A7)",
                   "flat_ucb_set[A7]": "Flat UCB + set dedup (A7)", "archtree_llm[A7_trie]": "Trie (A7 arm)"})
rows = []
for m in order:
    g = rf[(rf.method == m) & (rf.budget == 200)].set_index("dataset")
    cells = []
    for d in DS:
        if d in g.index:
            x = g.loc[d]
            ci = "" if np.isnan(x.b_ci_lo) else f" [{f(x.b_ci_lo)}, {f(x.b_ci_hi)}]"
            cells += [f"{int(x.T_real_evals)}", f(x.b_exponent) + ci, f(x.simple_regret_at_T)]
        else: cells += ["--"] * 3
    rows.append(lab[m] + " & " + " & ".join(cells))
hdr = "\\multirow{2}{*}{Method} & \\multicolumn{3}{c}{CIFAR-10} & \\multicolumn{3}{c}{CIFAR-100} & \\multicolumn{3}{c}{ImageNet16-120}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n & $T$ & $b$ [95\\% CI] & $r_T$ & $T$ & $b$ [95\\% CI] & $r_T$ & $T$ & $b$ [95\\% CI] & $r_T$"
write("regret_fit", tab("lrcrrcrrcr", hdr, rows))

# ---------------------------------------------------------------- landscape smoothness
ls = rd("final/results/landscape_smoothness.csv")
rows = []
for d in DS:
    g = ls[ls.dataset == d].sort_values("depth")
    for _, x in g.iterrows():
        rows.append((DSL[d] if x.depth == 1 else "") + f" & {int(x.depth)} & {int(x.n_prefixes):,} & {f(x.eta2, 3)} & {f(x.within_std)} & "
                    + " & ".join(str(int(x[f'N_k_eps{e}'])) for e in ["0.25", "0.5", "1", "2", "4"]))
    gm = g.iloc[0]
    rows.append(" & \\multicolumn{4}{r}{fitted $\\gamma$} & " + " & ".join(f(gm[f'gamma_eps{e}']) for e in ["0.25", "0.5", "1", "2", "4"]))
    if d != DS[-1]: rows.append("MID")
hdr = "Dataset & $k$ & prefixes & $\\eta^2$ & within s.d. & $N_k(0.25)$ & $N_k(0.5)$ & $N_k(1)$ & $N_k(2)$ & $N_k(4)$"
write("landscape", tab("lrrrrrrrrr", hdr, rows))

# ---------------------------------------------------------------- proxy calibration
pc = rd("phase2/results/phase2_proxy_calibration.csv"); pc1 = rd("nb101/results/phase2_proxy_calibration.csv")
PN = {"E": "$E$ (SVD entropy)", "S": "$S$ (SynFlow)", "T": "$T$ (NASWOT)", "C": "$C$ (FLOPs/params cost)",
      "ensemble": "supervised ensemble", "ensemble_unsupervised": "label-free ensemble $P$"}
rows = []
for p in PN:
    cells = []
    for d in DS:
        x = pc[(pc.dataset == d) & (pc.proxy_name == p)].iloc[0]; cells += [f(x.kendall_tau), f(x.spearman_rho)]
    y = pc1[pc1.proxy_name == p]
    cells += [f(y.iloc[0].kendall_tau), f(y.iloc[0].spearman_rho)] if len(y) else ["--", "--"]
    rows.append(PN[p] + " & " + " & ".join(cells))
hdr = "\\multirow{2}{*}{Proxy} & \\multicolumn{2}{c}{NB201 C10} & \\multicolumn{2}{c}{NB201 C100} & \\multicolumn{2}{c}{NB201 IN16} & \\multicolumn{2}{c}{NB101 C10}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n & $\\tau$ & $\\rho$ & $\\tau$ & $\\rho$ & $\\tau$ & $\\rho$ & $\\tau$ & $\\rho$"
write("proxy_calib", tab("lrrrrrrrr", hdr, rows))

# ---------------------------------------------------------------- proxy-cost correlation (all pairs)
cc = rd("equal_gpu_hours/results/cost_correlation.csv")
SC = {"E": "$E$", "S": "$S$", "T": "$T$", "C": "$C$", "ensemble_unsup": "$P$ (label-free)", "params": "\\#params", "valid_acc (oracle)": "val. acc. (oracle)"}
pairs = [("nb201", d) for d in DS] + [("nb101", "cifar10")] + [("natssss", d) for d in DS]
BLN = {"nb201": "NB201", "nb101": "NB101", "natssss": "NATS-SSS"}
rows = []
for b, d in pairs:
    g = cc[(cc.benchmark == b) & (cc.dataset == d)].set_index("score")
    first = True
    for s in SC:
        x = g.loc[s]
        rows.append((f"{BLN[b]} {DSS[d]}" if first else "") + f" & {SC[s]} & {f(x.spearman_with_train_time)} & {f(x.kendall_with_valid_acc)} & "
                    f"{f(x.top35_cost_ratio)} & {f(x.excess_cost_vs_oracle * 100, 1)} & {f(x.top35_mean_valid_acc)}")
        first = False
    if (b, d) != pairs[-1]: rows.append("MID")
hdr = "Space & Score & $\\rho$(score, $\\tau$) & $\\tau_K$(score, $A$) & cost ratio of top 35\\% & excess vs oracle (\\%) & mean $A$ of top 35\\%"
write("cost_corr", tab("llrrrrr", hdr, rows))

# ---------------------------------------------------------------- full fixed-budget results
mr = rd("final/results/main_results_table.csv"); mr = mr[mr.benchmark == "nb201"]
mo = ["random_search", "mcts_nas", "enas", "llm_nas", "zico", "rz_nas", "re", "re_zc_warmup", "local_search", "archtree_v1", "archtree_zc", "archtree_llm"]
rows = []
for d in DS:
    rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{DSL[d]}}}}}")
    best = {b: mr[(mr.dataset == d) & (mr.budget == b)].best_acc_mean.max() for b in BUD}
    for m in mo:
        cells = []
        for b in BUD:
            x = mr[(mr.method == m) & (mr.dataset == d) & (mr.budget == b)].iloc[0]
            c = pm(x.best_acc_mean, x.best_acc_std)
            cells.append("\\textbf{" + c + "}" if abs(x.best_acc_mean - best[b]) < 1e-9 else c)
        rows.append(("\\quad " + FIX[m]) + " & " + " & ".join(cells))
    if d != DS[-1]: rows.append("MID")
write("main_full", tab("lcccc", "Method & $B$ = 25 & $B$ = 50 & $B$ = 100 & $B$ = 200", rows))

# ---------------------------------------------------------------- paired tests vs baselines (fixed budget)
st = rd("final/results/statistical_tests.csv"); sts = rd("final/results/equal_compute_statistical_tests_strict.csv")
bo = ["random_search", "mcts_nas", "enas", "llm_nas", "zico", "rz_nas", "re", "re_zc_warmup", "local_search", "archtree_v1", "archtree_zc"]
rows = []
for m in bo:
    a = st[st.method_b == m]; b = sts[sts.baseline == m]
    rows.append(f"{FIX[m]} & {wtl_holm(a.p_value, a.mean_diff)} & {f(a.mean_diff.min())} to {f(a.mean_diff.max())} & {f(a.mean_diff.mean())} & {f(a.cohens_d.median())} & "
                f"{wtl_holm(b.p_value, b.mean_diff_archtree_minus_baseline)} & {f(b.mean_diff_archtree_minus_baseline.mean())}")
hdr = ("\\multirow{2}{*}{Baseline} & \\multicolumn{4}{c}{Same proposal budget (12 cells)} & \\multicolumn{2}{c}{Matched real evaluations (12 cells)}\\\\\n"
       "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\n & W/T/L & range of $\\Delta$ & mean $\\Delta$ & median $d$ & W/T/L & mean $\\Delta$")
write("fixed_tests", tab("lcccccc", hdr, rows))

# ---------------------------------------------------------------- duplicate evaluations
du = rd("final/results/duplicate_evals_summary.csv")
dm = ["random_search", "mcts_nas", "enas", "llm_nas", "zico", "rz_nas", "re", "re_zc_warmup", "local_search", "archtree_v1", "archtree_zc", "archtree_llm",
      "flat_ucb_none[A7]", "flat_ucb_set[A7]", "archtree_llm[A7_trie]"]
rows = []
for m in dm:
    g = du[du.method == m]
    cells = []
    for b in BUD:
        h = g[g.budget == b]
        cells += [f(100 * h.dup_rate_mean.mean(), 1), str(int(h.dup_real_max.max()))]
    rows.append(lab[m] + " & " + " & ".join(cells))
hdr = ("\\multirow{2}{*}{Method} & \\multicolumn{2}{c}{$B$ = 25} & \\multicolumn{2}{c}{$B$ = 50} & \\multicolumn{2}{c}{$B$ = 100} & \\multicolumn{2}{c}{$B$ = 200}\\\\\n"
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n & \\% & max & \\% & max & \\% & max & \\% & max")
write("dups", tab("lrrrrrrrr", hdr, rows))

# ---------------------------------------------------------------- equal GPU-hours: per-benchmark means
es = rd("equal_gpu_hours/results/egh_summary.csv")
ep = rd("equal_gpu_hours/results/egh_paired_tests.csv")
for b in ["nb201", "natssss", "nb101"]:
    dsb = DS if b != "nb101" else ["cifar10"]
    g = es[es.benchmark == b]
    arms = [a for a in EGH_ORDER if a in set(g.arm)]
    cells_def = [(d, k) for d in dsb for k in [25, 50, 100]]
    rows = []
    best = {c: g[(g.dataset == c[0]) & (g.k == c[1])].test_mean.max() for c in cells_def}
    for a in arms:
        cs = []
        for (d, k) in cells_def:
            x = g[(g.arm == a) & (g.dataset == d) & (g.k == k)]
            if not len(x): cs.append("--"); continue
            x = x.iloc[0]
            if b == "nb101":
                c = pm(x.test_mean, x.test_std)
            else:
                c = f(x.test_mean)
            if a != "archtree_zc_unsup":
                t = ep[(ep.benchmark == b) & (ep.dataset == d) & (ep.k_mean_arch_costs == k) & (ep.other == a)]
                if len(t) and t.iloc[0].outcome_holm == "win": c += "$^{\\triangledown}$"
                elif len(t) and t.iloc[0].outcome_holm == "loss": c += "$^{\\blacktriangle}$"
            if abs(x.test_mean - best[(d, k)]) < 1e-9: c = "\\textbf{" + c + "}"
            cs.append(c)
        if b == "nb101":
            ev = [g[(g.arm == a) & (g.k == k)].iloc[0].real_evals for k in [25, 50, 100]]
            cs += [" / ".join(f"{v:.0f}" for v in ev)]
        rows.append(EGH[a] + " & " + " & ".join(cs))
    if b == "nb101":
        hdr = "Arm & $k$ = 25 & $k$ = 50 & $k$ = 100 & real evals ($k$ = 25 / 50 / 100)"
        write("egh_" + b, tab("lcccc", hdr, rows))
    else:
        hdr = ("\\multirow{2}{*}{Arm} & \\multicolumn{3}{c}{CIFAR-10} & \\multicolumn{3}{c}{CIFAR-100} & \\multicolumn{3}{c}{ImageNet16-120}\\\\\n"
               "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n & 25 & 50 & 100 & 25 & 50 & 100 & 25 & 50 & 100")
        write("egh_" + b, tab("lrrrrrrrrr", hdr, rows))

# W/T/L per arm and benchmark + under-budget runs
rows = []
for a in EGH_ORDER[1:]:
    cs = []
    for b in ["nb201", "nb101", "natssss"]:
        t = ep[(ep.benchmark == b) & (ep.other == a)]
        cs += [wtl(t.outcome_holm) if len(t) else "--", f(t.mean_diff_archtree_minus_other.mean()) if len(t) else "--"]
    rows.append(EGH[a] + " & " + " & ".join(cs))
tot = [wtl(ep[ep.benchmark == b].outcome_holm) for b in ["nb201", "nb101", "natssss"]]
rows += ["MID", "All comparisons & " + " & & ".join(tot) + " & "]
hdr = ("\\multirow{2}{*}{Comparator} & \\multicolumn{2}{c}{NB201 (9 cells)} & \\multicolumn{2}{c}{NB101 (3 cells)} & \\multicolumn{2}{c}{NATS-SSS (9 cells)}\\\\\n"
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n & W/T/L & mean $\\Delta$ & W/T/L & mean $\\Delta$ & W/T/L & mean $\\Delta$")
write("egh_wtl", tab("lcccccc", hdr, rows))

# ---------------------------------------------------------------- Friedman
fr = rd("final/results/stats_friedman.csv")
SRC = {"nb201_full_budget": "NB201, fixed proposal budget", "nb201_equal_gpu_h": "NB201, equal GPU-hours",
       "nb101_same_budget": "NB101, fixed proposal budget", "nb101_equal_gpu_h": "NB101, equal GPU-hours",
       "natssss_equal_gpu_h": "NATS-SSS, equal GPU-hours"}
SO = list(SRC)
rows = []
for s in SO:
    x = fr[fr.source == s].iloc[0]
    e = int(np.floor(np.log10(x.friedman_p)))
    rows.append(f"{SRC[s]} & {int(x.n_methods)} & {int(x.n_blocks)} & {x.friedman_chi2:.1f} & ${x.friedman_p / 10 ** e:.1f}\\times10^{{{e}}}$ & {f(x.nemenyi_cd)} & "
                f"{f(x.archtree_v2_avg_rank)} & {(FIX | EGH).get(x.best_method, x.best_method)} ({f(x.best_avg_rank)})")
hdr = "Comparison & methods & blocks & $\\chi^2_F$ & $p$ & CD & rank of RMG & best method (rank)"
write("friedman", tab("lrrrrrrl", hdr, rows))

# ---------------------------------------------------------------- average ranks (all)
ar = rd("final/results/stats_avg_ranks.csv")
NAME = dict(EGH); NAME.update({k: v for k, v in FIX.items()}); NAME.update({"random": "Random", "re": "RE", "archtree_v2": "ArchTree-RMG"})
def rank_table(srcs, fname):
    meth = []
    for s in srcs:
        for m in ar[ar.source == s].sort_values("avg_rank").method:
            if m not in meth: meth.append(m)
    rows = []
    for m in meth:
        cs = []
        for s in srcs:
            x = ar[(ar.source == s) & (ar.method == m)]
            if not len(x): cs.append("--"); continue
            x = x.iloc[0]; c = f(x.avg_rank) + ("" if x.within_cd_of_archtree else "$^{*}$")
            cs.append("\\textbf{" + c + "}" if m == "archtree_v2" else c)
        rows.append(("\\textbf{ArchTree-RMG}" if m == "archtree_v2" else NAME[m]) + " & " + " & ".join(cs))
    hdr = "Method & " + " & ".join(SRC[s].replace(", ", "\\newline ") for s in srcs)
    write(fname, tab("l" + "c" * len(srcs), "Method & " + " & ".join(SRC[s] for s in srcs), rows))
rank_table(["nb201_full_budget", "nb101_same_budget"], "ranks_fixed")
rank_table(["nb201_equal_gpu_h", "nb101_equal_gpu_h", "natssss_equal_gpu_h"], "ranks_egh")

# ---------------------------------------------------------------- P(improvement), all sources
pi = rd("final/results/stats_prob_improvement.csv")
def pi_table(srcs, fname):
    meth = []
    for s in srcs:
        for m in pi[pi.source == s].sort_values("P_archtree_better").other:
            if m not in meth: meth.append(m)
    rows = []
    for m in meth:
        cs = []
        for s in srcs:
            x = pi[(pi.source == s) & (pi.other == m)]
            if not len(x): cs.append("--"); continue
            x = x.iloc[0]; c = f"{f(x.P_archtree_better)} [{f(x.ci95_lo)}, {f(x.ci95_hi)}]"
            if round(x.ci95_lo, 2) > 0.5: c = "\\textbf{" + c + "}"
            elif round(x.ci95_hi, 2) < 0.5: c = "\\textit{" + c + "}"
            cs.append(c)
        rows.append(NAME[m] + " & " + " & ".join(cs))
    npairs = " & ".join(f"$n$ = {int(pi[pi.source == s].n_pairs.iloc[0])}" for s in srcs)
    rows += ["MID", "Paired runs & " + npairs]
    write(fname, tab("l" + "c" * len(srcs), "Comparator & " + " & ".join(SRC[s] for s in srcs), rows))
pi_table(["nb201_full_budget", "nb101_same_budget"], "pi_fixed")
pi_table(["nb201_equal_gpu_h", "nb101_equal_gpu_h", "natssss_equal_gpu_h"], "pi_egh")

# ---------------------------------------------------------------- TOST counts
to = rd("final/results/stats_tost.csv")
rows = []
for s in SO:
    g = to[to.source == s]
    for o in g.other.unique():
        h = g[g.other == o]; n = h.cell.nunique()
        e5 = int(h[(h.delta == 0.5)].equivalent.sum()); e25 = int(h[(h.delta == 0.25)].equivalent.sum())
        rows.append((SRC[s] if o == g.other.unique()[0] else "") + f" & {NAME[o]} & {n} & {e5} & {e25} & {f(h[h.delta == 0.5].mean_diff.abs().max())}")
    if s != SO[-1]: rows.append("MID")
hdr = "Comparison & Comparator & cells & equiv. at $\\pm$0.5 & equiv. at $\\pm$0.25 & max $|\\Delta|$"
write("tost", tab("llrrrr", hdr, rows))

# ---------------------------------------------------------------- proposer x gate grid (B = 200)
gs = rd("gated_baselines/results/gated_summary.csv")
rows = []
for p in ["archtree", "re", "ls", "random", "enas", "mcts"]:
    for gt in ["none", "params", "zc", "zc_unsup"]:
        cs = []
        for d in DS:
            x = gs[(gs.proposer == p) & (gs.gate == gt) & (gs.dataset == d) & (gs.budget == 200)].iloc[0]
            cs += [pm(x.acc_mean, x.acc_std), f"{x.real_evals:.0f}"]
        x = gs[(gs.proposer == p) & (gs.gate == gt) & (gs.budget == 200)]
        rows.append((PROP[p] if gt == "none" else "") + f" & {GATE[gt]} & " + " & ".join(cs) + f" & {x.train_hours.sum():.0f}")
    if p != "mcts": rows.append("MID")
hdr = ("\\multirow{2}{*}{Proposer} & \\multirow{2}{*}{Gate} & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{CIFAR-100} & \\multicolumn{2}{c}{ImageNet16-120} & GPU-h\\\\\n"
       "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n & & acc. & evals & acc. & evals & acc. & evals & (3 sets)")
write("grid", tab("llcrcrcrr", hdr, rows))

gp = rd("gated_baselines/results/gated_paired_tests.csv")
rows = []
for o in ["random", "re", "ls", "enas", "mcts"]:
    cs = []
    for gt in ["none", "params", "zc", "zc_unsup"]:
        h = gp[(gp.other == o) & (gp.gate == gt)]
        cs += [wtl(h.outcome_holm), f(h.mean_diff_archtree_minus_other.mean())]
    rows.append(PROP[o] + " & " + " & ".join(cs))
hdr = ("\\multirow{2}{*}{Other proposer} & \\multicolumn{2}{c}{no gate} & \\multicolumn{2}{c}{\\#params} & \\multicolumn{2}{c}{supervised} & \\multicolumn{2}{c}{label-free}\\\\\n"
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n & W/T/L & mean $\\Delta$ & W/T/L & mean $\\Delta$ & W/T/L & mean $\\Delta$ & W/T/L & mean $\\Delta$")
write("grid_tests", tab("lcccccccc", hdr, rows))

# ---------------------------------------------------------------- held-out design ablation
de = rd("design_ablation/results/design_summary.csv"); dt = rd("design_ablation/results/design_paired_tests.csv")
DA = {"v1": "ArchTree-base + gate", "rank": "rank reward only", "mut_only": "mutation only", "re": "RE + same gate", "v2": "ArchTree-RMG"}
rows = []
for d in DS:
    rows.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{DSL[d]}}}}}")
    for a in ["v1", "rank", "mut_only", "re", "v2"]:
        cs = []
        for b in BUD:
            x = de[(de.arm == a) & (de.dataset == d) & (de.budget == b)].iloc[0]
            c = pm(x.test_mean, x.test_std)
            if a != "v2":
                t = dt[(dt.arm == a) & (dt.dataset == d) & (dt.budget == b)].iloc[0]
                if t.outcome_vs_v2 == "v2 better": c += "$^{\\triangledown}$"
                elif t.outcome_vs_v2 not in ("tie",): c += "$^{\\blacktriangle}$"
            cs.append(c)
        ev = de[(de.arm == a) & (de.dataset == d) & (de.budget == 200)].iloc[0].real_evals
        rows.append("\\quad " + DA[a] + " & " + " & ".join(cs) + f" & {ev:.1f}")
    if d != DS[-1]: rows.append("MID")
write("design", tab("lccccr", "Arm & $B$ = 25 & $B$ = 50 & $B$ = 100 & $B$ = 200 & evals ($B$=200)", rows))

# ---------------------------------------------------------------- A7 trie vs flat
a7 = rd("ablations/results/A7_summary.csv"); a7t = rd("ablations/results/A7_paired_tests.csv")
A7N = {"trie": "trie (dedup by construction)", "flat_set": "flat UCB + set dedup", "flat_none": "flat UCB, no dedup"}
rows = []
for d in DS:
    for b in BUD:
        cs = []
        for a in ["trie", "flat_set", "flat_none"]:
            x = a7[(a7.arm == a) & (a7.dataset == d) & (a7.budget == b)].iloc[0]
            cs += [pm(x.best_acc_mean, x.best_acc_std), f"{x.unique_real_mean:.1f} / {x.dup_real_mean:.1f}"]
        ps = []
        for o in ["flat_set", "flat_none"]:
            t = a7t[(a7t.dataset == d) & (a7t.budget == b) & (a7t.arm_b == o)].iloc[0]
            ps.append(f"{f(t.mean_diff_pp)} ({t.wilcoxon_p:.3f})")
        rows.append((DSL[d] if b == 25 else "") + f" & {b} & " + " & ".join(cs) + " & " + " & ".join(ps))
    if d != DS[-1]: rows.append("MID")
hdr = ("\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{$B$} & \\multicolumn{2}{c}{trie} & \\multicolumn{2}{c}{flat UCB + set dedup} & \\multicolumn{2}{c}{flat UCB, no dedup} & \\multicolumn{2}{c}{trie minus ($p$)}\\\\\n"
       "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n & & acc. & uniq./dup. & acc. & uniq./dup. & acc. & uniq./dup. & flat+set & flat")
write("a7", tab("lrcccccccc", hdr, rows))

# ---------------------------------------------------------------- UCB sensitivity
us = rd("ucb_sensitivity/results/ucb_summary.csv"); ut = rd("ucb_sensitivity/results/ucb_paired_tests.csv")
rows = []
for bk in ["mean", "max"]:
    for Cv in sorted(us.C.unique()):
        cs = []
        for d in DS:
            for b in [100, 200]:
                x = us[(us.backup == bk) & (us.C == Cv) & (us.dataset == d) & (us.budget == b)].iloc[0]
                c = f(x.acc_mean)
                if bk == "max" and abs(Cv - 0.3) < 1e-9: c = "\\textbf{" + c + "}"
                cs.append(c)
        x = us[(us.backup == bk) & (us.C == Cv)]
        Cs = "$\\sqrt{2}$" if abs(Cv - 1.414) < 1e-3 else f"{Cv:g}"
        rows.append((bk if Cv == 0.1 else "") + f" & {Cs} & " + " & ".join(cs) + f" & {x.dup_proposals.mean():.1f}")
    if bk == "mean": rows.append("MID")
hdr = ("\\multirow{2}{*}{Backup} & \\multirow{2}{*}{$C$} & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{CIFAR-100} & \\multicolumn{2}{c}{ImageNet16-120} & dup.\\\\\n"
       "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n & & 100 & 200 & 100 & 200 & 100 & 200 & proposals")
write("ucb", tab("llrrrrrrr", hdr, rows))

# ---------------------------------------------------------------- proxy subsets
pa = rd("proxy_ablation/results/proxy_ablation_summary.csv"); pt = rd("proxy_ablation/results/proxy_ablation_paired_tests.csv")
subs = ["E", "S", "T", "C", "ES", "ET", "EC", "ST", "SC", "TC", "EST", "ESC", "ETC", "STC", "ESTC", "none"]
rows = []
for s in subs:
    cs = []
    for d in DS:
        x = pa[(pa.arm == s) & (pa.dataset == d)].iloc[0]
        cs += [f(x.kendall_tau) if s != "none" else "--", pm(x.best_acc_mean, x.best_acc_std), f"{x.real_evals_mean:.1f}"]
    nm = "no gate" if s == "none" else "$" + s + "$"
    rows.append(nm + " & " + " & ".join(cs))
hdr = ("\\multirow{2}{*}{Proxies} & \\multicolumn{3}{c}{CIFAR-10} & \\multicolumn{3}{c}{CIFAR-100} & \\multicolumn{3}{c}{ImageNet16-120}\\\\\n"
       "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n & $\\tau$ & acc. & evals & $\\tau$ & acc. & evals & $\\tau$ & acc. & evals")
write("proxy_subsets", tab("lrcrrcrrcr", hdr, rows))

# ---------------------------------------------------------------- gate-cost model (full correlations)
gm = rd("equal_gpu_hours/results/gate_cost_model.csv").drop_duplicates(["benchmark", "dataset"])
gv = rd("equal_gpu_hours/results/gate_cost_validation.csv")
rows = []
for b, d in pairs:
    x = gm[(gm.benchmark == b) & (gm.dataset == d)].iloc[0]
    v = gv[(gv.benchmark == b) & (gv.dataset == d) & (gv["lambda"] == 1.0)]
    rows.append(f"{BLN[b]} {DSS[d]} & {f(x.r_PA)} & {f(x.r_PD)} & {f(x.r_DA)} & {f(x.r_PC)} & {f(x.r_AC)} & {f(x.r_DC)} & {f(x.slope_at_0, 3)} & "
                f"{f(v.predicted_delta.mean())} & {f(v.measured_delta.mean())}")
hdr = "Space & $r_{PA}$ & $r_{PD}$ & $r_{DA}$ & $r_{PC}$ & $r_{AC}$ & $r_{DC}$ & slope & pred. $\\Delta$ & meas. $\\Delta$"
write("gatecost_full", tab("lrrrrrrrrr", hdr, rows))

# ---------------------------------------------------------------- toggle ablations A1-A6
A = {k: rd(f"ablations/results/ablation_{k}.csv") for k in ["A1_no_llm", "A2_no_zc", "A3_no_weight_inherit", "A6_budget_sensitivity"]}
ph3 = rd("phase3/results/phase3_main_results.csv")
rows = []
for d in DS:
    for b in BUD:
        h = ph3[(ph3.dataset == d) & (ph3.budget == b)]
        cs = [f"{f(h.best_test_acc.mean())} ({h.real_eval_count.mean():.1f})"]
        for k in ["A1_no_llm", "A2_no_zc", "A3_no_weight_inherit", "A6_budget_sensitivity"]:
            g = A[k][(A[k].dataset == d) & (A[k].budget == b)]
            cs.append(f"{f(g.best_test_acc.mean())} ({g.real_eval_count.mean():.1f})")
        rows.append((DSL[d] if b == 25 else "") + f" & {b} & " + " & ".join(cs))
    if d != DS[-1]: rows.append("MID")
hdr = "Dataset & $B$ & S4 \\rmg{} & A1 no LLM & A2 no gate & A3 no inheritance & A6 budget grid"
write("toggles", tab("lrccccc", hdr, rows))
# identity checks used in the text
ident = {}
for k in ["A1_no_llm", "A3_no_weight_inherit", "A6_budget_sensitivity"]:
    mrg = ph3.merge(A[k], on=["dataset", "seed", "budget"], suffixes=("", "_a"))
    ident[k] = (int((np.isclose(mrg.best_test_acc, mrg.best_test_acc_a) & (mrg.real_eval_count == mrg.real_eval_count_a)).sum()), len(mrg))
a5 = rd("ablations/results/ablation_A5_prune_threshold.csv"); a4 = rd("ablations/results/ablation_A4_alpha_sensitivity.csv")
rows = []
for pv in [0, 15, 25, 35]:
    cs = []
    for d in DS:
        g = a5[(a5.param_value == pv) & (a5.dataset == d)]
        cs.append(pm(g.best_test_acc.mean(), g.best_test_acc.std(ddof=1)) + f" ({g.real_eval_count.mean():.1f})")
    rows.append(f"{pv}\\%" + (" (default)" if pv == 15 else "") + " & " + " & ".join(cs))
write("a5", tab("lccc", "Pruned share & CIFAR-10 & CIFAR-100 & ImageNet16-120", rows))

# ---------------------------------------------------------------- alpha sweep
al = rd("alpha_sweep/results/alpha_vs_baseline_stats.csv"); asr = rd("alpha_sweep/results/alpha_sweep_results.csv")
rows = []
for a in sorted(asr.alpha.unique()):
    cs = []
    for d in DS:
        for b in [50, 100]:
            g = asr[(asr.alpha == a) & (asr.dataset == d) & (asr.budget == b)]
            c = f(g.best_test_acc.mean())
            if a > 0:
                t = al[(al.alpha == a) & (al.dataset == d) & (al.budget == b)]
                if len(t):
                    t = t.iloc[0]; c += f" ({'+' if t.mean_diff_vs_alpha0 >= 0 else ''}{f(t.mean_diff_vs_alpha0)})"
                    if t.significant: c = "\\textbf{" + c + "}"
            cs.append(c)
    rows.append(f"{a:g}" + (" (off)" if a == 0 else "") + " & " + " & ".join(cs))
hdr = ("\\multirow{2}{*}{$\\alpha$} & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{CIFAR-100} & \\multicolumn{2}{c}{ImageNet16-120}\\\\\n"
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n & $B$ = 50 & $B$ = 100 & $B$ = 50 & $B$ = 100 & $B$ = 50 & $B$ = 100")
write("alpha", tab("lcccccc", hdr, rows))
alpha_same = al.groupby("alpha").same_runs_as_alpha.first().to_dict()

# ---------------------------------------------------------------- convergence metrics
cm = rd("convergence/results/convergence_per_run_metrics.csv"); ct = rd("convergence/results/convergence_paired_tests.csv")
CA = {"llm_off": "prior off ($\\alpha$=0)", "llm_on": "LLM prior (standard)", "llm_described": "LLM prior (described ops)", "size_prior": "size prior (no LLM)"}
rows = []
for d in DS:
    for a in ["llm_off", "llm_on", "llm_described", "size_prior"]:
        g = cm[(cm.arm == a) & (cm.dataset == d) & (cm.budget == 100)]
        def ev(col, reach):
            ok = g[g[reach]]
            return f"{g[col].mean():.1f}" + ("" if len(ok) == len(g) else f" ({len(ok)}/12)")
        cs = [f(g.best_at_10.mean()), f(g.best_at_25.mean()), f(g.final_best.mean()), ev("evals_to_2.0pp", "reached_2.0pp"),
              ev("evals_to_1.0pp", "reached_1.0pp"), ev("evals_to_0.5pp", "reached_0.5pp"), f"{g.n_real_evals.mean():.1f}"]
        rows.append((DSL[d] if a == "llm_off" else "") + f" & {CA[a]} & " + " & ".join(cs))
    if d != DS[-1]: rows.append("MID")
hdr = "Dataset & Arm & best@10 & best@25 & final & evals to 2\\,pt & to 1\\,pt & to 0.5\\,pt & real evals"
write("convergence", tab("llrrrrrrr", hdr, rows))

# ---------------------------------------------------------------- LLM probes
cs_ = rd("llm_contamination/results/contamination_summary.csv"); ctt = rd("llm_contamination/results/contamination_tests.csv")
ls_ = rd("llm_size/results/llm_size_summary.csv")
MODE = {"standard": "standard prompt (op names)", "described": "operations described, names hidden", "anonymous": "anonymous tokens", "size_prior": "size prior (no LLM)"}
rows = []
for mo_ in ["standard", "described", "anonymous", "size_prior"]:
    cs = []
    for d in DS:
        x = cs_[(cs_["mode"] == mo_) & (cs_.dataset == d)].iloc[0]
        cs += [f(x.spearman_mean), f(x.top1_hit_rate)]
    rows.append(MODE[mo_] + " & " + " & ".join(cs))
hdr = ("\\multirow{2}{*}{Probe} & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{CIFAR-100} & \\multicolumn{2}{c}{ImageNet16-120}\\\\\n"
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n & $\\rho$ & top-1 & $\\rho$ & top-1 & $\\rho$ & top-1")
write("probe", tab("lrrrrrr", hdr, rows))
rows = []
for d in DS:
    for c in ctt[ctt.dataset == d].comparison:
        x = ctt[(ctt.dataset == d) & (ctt.comparison == c)].iloc[0]
        rows.append((DSL[d] if c == ctt[ctt.dataset == d].comparison.iloc[0] else "") + " & " + c.replace("_", " ") +
                    f" & {int(x.n_paired_seeds)} & {f(x.mean_diff, 3)} & {pfmt(x.p_holm)} & {'yes' if x.significant else 'no'}")
    if d != DS[-1]: rows.append("MID")
write("probe_tests", tab("llrrrl", "Dataset & Comparison & $n$ & mean diff. ($\\rho$) & $p_{\\mathrm{Holm}}$ & sig.", rows))
lt = rd("llm_size/results/llm_size_paired_tests.csv"); lsr = rd("llm_size/results/llm_size_search.csv")
rows = []
for _, x in ls_.iterrows():
    mname = x.model.split(":")[1]
    rows.append(f"\\texttt{{{mname}}} & {int(x.n)} & {f(x.spearman_mean)} & {f(x.top1_hit_rate)} & {f(100 * x.uniform_fallback_rate, 1)} & {f(x.name_mismatch_per_call)}")
write("llm_size", tab("lrrrrr", "Model & prefixes & mean $\\rho$ & top-1 hit & fallback (\\%) & op-name errors / call", rows))

# ---------------------------------------------------------------- NB101
nb = rd("nb101/results/nb101_summary.csv"); nbt = rd("nb101/results/nb101_paired_tests.csv")
rows = []
for p in ["archtree", "re", "random"]:
    for gt in ["none", "params", "zc", "zc_unsup"]:
        cs = []
        for b in [50, 100, 200, 400]:
            x = nb[(nb.proposer == p) & (nb.gate == gt) & (nb.budget == b)].iloc[0]
            cs.append(pm(x.test_mean, x.test_std) + f" ({x.real_evals:.0f})")
        rows.append((PROP[p] if gt == "none" else "") + f" & {GATE[gt]} & " + " & ".join(cs))
    if p != "random": rows.append("MID")
write("nb101", tab("llcccc", "Proposer & Gate & $B$ = 50 & $B$ = 100 & $B$ = 200 & $B$ = 400", rows))
rows = []
for gt in ["none", "params", "zc", "zc_unsup"]:
    cs = []
    for o in ["random", "re"]:
        h = nbt[(nbt.gate == gt) & (nbt.other == o)]
        cs += [wtl(h.outcome_holm), f(h.mean_diff_archtree_minus_other.mean(), 3)]
    rows.append(GATE[gt] + " & " + " & ".join(cs))
hdr = ("\\multirow{2}{*}{Gate (same for both)} & \\multicolumn{2}{c}{ArchTree-RM vs random} & \\multicolumn{2}{c}{ArchTree-RM vs RE}\\\\\n"
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & W/T/L & mean $\\Delta$ & W/T/L & mean $\\Delta$")
write("nb101_tests", tab("lcccc", hdr, rows))

# ---------------------------------------------------------------- NB101 matched evaluations
ne = rd("nb101/results/nb101_equal_compute_tests.csv")
rows = []
for gt in ["params", "zc", "zc_unsup"]:
    for b in [50, 100, 200, 400]:
        h = ne[(ne.gate == gt) & (ne.archtree_budget == b)]
        cs = []
        for o in ["random_ungated", "re_ungated", "archtree_ungated"]:
            x = h[h.other == o].iloc[0]
            c = f(x.mean_diff_archtree_minus_other) + f" ({x.p_holm:.3f})"
            if x.outcome_holm == "win": c = "\\textbf{" + c + "}"
            elif x.outcome_holm == "loss": c = "\\textit{" + c + "}"
            cs.append(c)
        rows.append((GATE[gt] if b == 50 else "") + f" & {b} & {int(h.iloc[0].archtree_real_evals_max)} & {int(h.iloc[0].other_budget)} & " + " & ".join(cs))
    if gt != "zc_unsup": rows.append("MID")
hdr = ("Gate & $B$ & max evals & other $B$ & vs random & vs RE & vs \\algname-RM")
write("nb101_matched", tab("lrrrccc", hdr, rows))

# ---------------------------------------------------------------- hardware
hv = rd("phase4/results/phase4_hv_igd_summary.csv"); hb = rd("phase4/results/phase4_baselines_hv_igd_summary.csv")
ht = rd("phase4/results/phase4_hw_paired_tests.csv")
HW = ["edgegpu", "raspi4", "edgetpu", "pixel3", "eyeriss", "fpga"]
HWN = {"edgegpu": "Edge GPU", "raspi4": "Raspberry Pi 4", "edgetpu": "Edge TPU", "pixel3": "Pixel 3", "eyeriss": "Eyeriss", "fpga": "FPGA"}
rows = []
for h in HW:
    for b in BUD:
        a = hv[(hv.hardware == h) & (hv.budget == b)]
        r_ = hb[(hb.hardware == h) & (hb.budget == b) & (hb.method == "random_hw")]
        n_ = hb[(hb.hardware == h) & (hb.budget == b) & (hb.method == "nsga2_hw")]
        base = r_.hypervolume.mean()
        cs = [f"{100 * a.hypervolume.mean() / base:.2f}", f"{100 * n_.hypervolume.mean() / base:.2f}",
              f(a.igd.mean()), f(n_.igd.mean()), f(r_.igd.mean()), f"{a.pareto_front_size.mean():.1f}"]
        rows.append((HWN[h] if b == 25 else "") + f" & {b} & " + " & ".join(cs))
    if h != HW[-1]: rows.append("MID")
hdr = ("\\multirow{2}{*}{Device} & \\multirow{2}{*}{$B$} & \\multicolumn{2}{c}{HV (\\% of random)} & \\multicolumn{3}{c}{IGD} & front\\\\\n"
       "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\n & & ArchTree-HW & NSGA-II-style & ArchTree-HW & NSGA-II-style & random & size")
write("hw", tab("lrrrrrrr", hdr, rows))

co = rd("final/results/cross_device_truth_overlap.csv"); ctr = rd("final/results/cross_device_truth_transfer.csv")
rows = []
for _, x in co.iterrows():
    t1 = ctr[(ctr.from_hw == x.hw_a) & (ctr.to_hw == x.hw_b)].iloc[0]; t2 = ctr[(ctr.from_hw == x.hw_b) & (ctr.to_hw == x.hw_a)].iloc[0]
    rows.append(f"{HWN[x.hw_a]} -- {HWN[x.hw_b]} & {int(x.front_a)} / {int(x.front_b)} & {int(x.n_shared)} & {f(x.jaccard)} & {f(x.latency_spearman)} & "
                f"{f(t1.hv_ratio, 3)} & {f(t2.hv_ratio, 3)}")
hdr = "Device pair ($a$ -- $b$) & front sizes & shared & Jaccard & $\\rho$(latency) & HV ratio $a\\to b$ & HV ratio $b\\to a$"
write("crossdev", tab("lrrrrrr", hdr, rows))

# ---------------------------------------------------------------- test inventory (1,074)
ex = rd("final/results/extended_findings_summary.csv")
EXN = {"gated_baselines": "proposer $\\times$ gate grid", "equal_gpu_hours": "equal GPU-hours", "convergence_study": "LLM convergence study",
       "design_ablation_heldout": "held-out design ablation", "A7_trie_vs_flat_ucb1": "trie vs flat UCB (A7)",
       "proxy_composition_ablation": "proxy subsets", "nb101_same_gate": "NB101, same gate", "nb101_equal_compute": "NB101, matched evaluations",
       "llm_contamination_probe": "LLM knowledge probe", "llm_size_sensitivity": "LLM model size"}
rows = []
for e_ in ex.experiment.value_counts().index:
    g = ex[ex.experiment == e_]
    pv = g.p_value.dropna()
    rows.append(f"{EXN[e_]} & {len(g)} & {len(pv)} & {(pv < 0.05).sum()} & {int(g.significant.fillna(False).astype(bool).sum())} & "
                f"{int((g.significant.fillna(False).astype(bool) & (g.mean_diff > 0)).sum())} / {int((g.significant.fillna(False).astype(bool) & (g.mean_diff < 0)).sum())}")
rows += ["MID", f"Total & {len(ex)} & {ex.p_value.notna().sum()} & {(ex.p_value < 0.05).sum()} & {int(ex.significant.fillna(False).astype(bool).sum())} & "]
write("inventory", tab("lrrrrr", "Experiment & rows & tested & $p<0.05$ & significant (as filed) & $+$ / $-$", rows))

# ---------------------------------------------------------------- facts used in the text
egr = rd("equal_gpu_hours/results/egh_runs.csv")
facts = dict(ident=ident, alpha_same=alpha_same, egh_runs=len(egr), under_budget=int(egr.under_budget.sum()),
             under_budget_rows=egr[egr.under_budget][["benchmark", "dataset", "arm", "seed", "k"]].to_dict("records"),
             dup_archtree=float(du[du.method.isin(["archtree_v1", "archtree_zc", "archtree_llm", "archtree_llm[A7_trie]"])].dup_real_max.max()))
json.dump(facts, open(os.path.join(T, "facts.json"), "w"), indent=1, default=str)
print(json.dumps(facts, indent=1, default=str)[:3000])
