"""Supplementary figures for the ArchTree paper (Sadhana); same style as make_figures.py.
usage: python3 make_supp_figures.py <outputs dir> <figs dir>"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

R, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
for f in ["texgyretermes-regular.otf", "texgyretermes-bold.otf", "texgyretermes-italic.otf"]:
    fm.fontManager.addfont("/usr/share/texmf/fonts/opentype/public/tex-gyre/" + f)
INK, MUTED, GRID = "#262626", "#6b6b6b", "#e4e4e1"
plt.rcParams.update({
    "font.family": "TeX Gyre Termes", "mathtext.fontset": "custom", "mathtext.rm": "TeX Gyre Termes", "mathtext.it": "TeX Gyre Termes:italic", "mathtext.bf": "TeX Gyre Termes:bold", "mathtext.sf": "TeX Gyre Termes", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.linewidth": 0.6, "axes.edgecolor": "#3a3a3a",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "axes.labelcolor": INK, "text.color": INK,
    "lines.linewidth": 1.5, "lines.markersize": 4, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3,
    "savefig.dpi": 600, "pdf.fonttype": 3, "figure.dpi": 150, "lines.solid_capstyle": "round", "patch.linewidth": 0, "xtick.direction": "out", "ytick.direction": "out", "axes.titlepad": 5, "legend.handletextpad": 0.5})
C = dict(arch="#2a78d6", re="#eb6834", ls="#13976a", warm="#4a3aa7", cost="#a3367e", y="#b88300",
         rnd="#8a8985", ink=INK, s1="#9ec3ec", s2="#5f9be0", s3="#1d5fae")
COL, FULL = 3.35, 6.89
DS = ["cifar10", "cifar100", "ImageNet16-120"]
DSL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "ImageNet16-120": "ImageNet16-120"}
DSS = {"cifar10": "C10", "cifar100": "C100", "ImageNet16-120": "IN16"}
BL = {"nb201": "NB201", "nb101": "NB101", "natssss": "NATS-SSS"}
DCOL = {"cifar10": C["arch"], "cifar100": C["re"], "ImageNet16-120": C["ls"]}
DMK = {"cifar10": "o", "cifar100": "s", "ImageNet16-120": "^"}
rd = lambda p: pd.read_csv(os.path.join(R, p))
BLUES = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9ec3ec", "#2a78d6", "#123f78"])
DIV = LinearSegmentedColormap.from_list("d", ["#b8322f", "#f0b8a8", "#f7f7f5", "#a9c9ef", "#1d5fae"])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.03, dpi=600)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.03, dpi=600)
    plt.close(fig)


import re as _re
def plabel(ax, s, fs=8, pad=4):
    m_ = _re.match(r"^\(([a-z])\)\s*(.*)$", s, _re.S)
    if m_:
        s = r"$\mathbf{(" + m_.group(1) + r")}$ " + m_.group(2)
    ax.set_title(s, loc="left", fontsize=fs, pad=pad)


def logticks(ax):
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(1, 10)))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=1.5, width=0.4)


def logx_budget(ax, ticks=(25, 50, 100, 200)):
    ax.set_xscale("log", base=2); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def heat(ax, M, rows, cols, cmap, vmin, vmax, fmt="{:.2f}", fs=6.2, norm=None, thr=0.6):
    im = ax.imshow(M, cmap=cmap, vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(cols); ax.set_yticklabels(rows); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]): continue
            rel = (M[i, j] - vmin) / (vmax - vmin) if norm is None else abs(norm(M[i, j]) - 0.5) * 2
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=fs,
                    color="white" if rel > thr else INK)
    return im


# ======================================================================================
# Supplementary figures (S-numbered). Every value is read from the result CSVs in <outputs>.
# ======================================================================================
FIX = {"random_search": "Random search", "mcts_nas": "MCTS-style", "enas": "ENAS-style", "llm_nas": "LLM-NAS",
       "zico": "ZiCo-style", "rz_nas": "RZ-style", "re": "RE", "re_zc_warmup": "RE + ZC warm-up",
       "local_search": "Local search", "archtree_v1": "S1 ArchTree-base", "archtree_zc": "S2 + supervised gate",
       "archtree_llm": "S4 ArchTree-RMG"}
EGH = {"archtree_zc_unsup": "ArchTree-RMG", "archtree_none": "ArchTree-RM (no gate)", "archtree_params": "ArchTree-RM + #params gate",
       "archtree_costpen025": "ArchTree-RM + cost pen. (0.25)", "archtree_costpen": "ArchTree-RM + cost pen. (1)",
       "re_zc_unsup": "RE + gate", "re_zcwarm": "RE + ZC warm-up", "re_none": "RE", "ls_zc_unsup": "LS + gate", "ls_none": "LS",
       "random_zc_unsup": "random + gate", "random_none": "random", "random_costpen025": "random + cost pen. (0.25)",
       "random_costpen": "random + cost pen. (1)"}
BUD = [25, 50, 100, 200]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); mm = len(p); adj = np.empty(mm); run = 0
    for r_, i in enumerate(o): run = max(run, (mm - r_) * p[i]); adj[i] = min(1, run)
    return adj


def sigmark(ax, i, j, sig, raw, col_dark):
    if sig: ax.plot(j + 0.36, i - 0.3, marker="o", ms=2.4, color=INK, mec="white", mew=0.3, zorder=5)
    elif raw: ax.plot(j + 0.36, i - 0.3, marker="o", ms=2.4, mfc="none", mec=INK, mew=0.45, zorder=5)


# ------------------------------------------------------------------ S2: regret curves for every method
rc = rd("final/results/regret_curves.csv"); rf = rd("final/results/regret_fit.csv")
order = [("random_exact", "uniform sampling (exact)", C["rnd"], (0, (1, 1.2)), 1.0),
         ("random_search", "Random search", C["rnd"], "-", 0.9), ("mcts_nas", "MCTS-style", C["y"], "-", 0.9),
         ("enas", "ENAS-style", "#c9a227", (0, (4, 1.5)), 0.9), ("llm_nas", "LLM-NAS", C["warm"], "-", 0.9),
         ("zico", "ZiCo-style", C["cost"], (0, (4, 1.5)), 0.9), ("rz_nas", "RZ-style", C["cost"], "-", 0.9),
         ("re", "RE", C["re"], "-", 0.9), ("re_zc_warmup", "RE + ZC warm-up", C["re"], (0, (4, 1.5)), 0.9),
         ("local_search", "Local search", C["ls"], "-", 0.9), ("archtree_v1", "S1 ArchTree-base", C["s1"], "-", 1.0),
         ("archtree_zc", "S2 + supervised gate", C["s2"], "-", 1.0), ("archtree_llm", "S4 ArchTree-RMG", C["arch"], "-", 1.9)]
fig = plt.figure(figsize=(FULL, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.55, wspace=0.28, top=0.88)
for j, d in enumerate(DS):
    ax = fig.add_subplot(gs[0, j])
    for m, lab, c, ls_, lw in order:
        t = rc[(rc.method == m) & (rc.dataset == d) & (rc.budget == 200)].sort_values("t")
        y = t.mean_regret.clip(lower=1e-2)
        ax.plot(t.t, y, color=c, ls=ls_, lw=lw, label=lab, zorder=5 if m == "archtree_llm" else 3)
        if m == "archtree_llm":
            ax.fill_between(t.t, (t.mean_regret - t["sem"]).clip(lower=1e-2), t.mean_regret + t["sem"], color=c, alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); logticks(ax)
    ax.set_xlim(1, 200); ax.set_ylim(0.05, 40)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("real evaluations $t$")
    if j == 0: ax.set_ylabel("mean simple regret (pt)")
    plabel(ax, f"({'abc'[j]}) {DSL[d]}, $B$ = 200")
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=7, fontsize=6.2, handlelength=2.2, columnspacing=1.0)
ax = fig.add_subplot(gs[1, :])
ms = [m for m, *_ in order if m != "random_exact"]
for k, d in enumerate(DS):
    g = rf[(rf.dataset == d) & (rf.budget == 200)].set_index("method")
    xs = np.arange(len(ms)) + (k - 1) * 0.22
    b = [g.loc[m, "b_exponent"] for m in ms]; lo = [g.loc[m, "b_ci_lo"] for m in ms]; hi = [g.loc[m, "b_ci_hi"] for m in ms]
    ax.errorbar(xs, b, yerr=[np.array(b) - lo, np.array(hi) - b], fmt=DMK[d], color=DCOL[d], ms=3.4, lw=0.8, capsize=1.5, label=DSL[d])
    if "random_exact" in g.index:
        ax.axhline(g.loc["random_exact", "b_exponent"], color=DCOL[d], lw=0.7, ls=(0, (1, 1.2)))
ax.set_xticks(range(len(ms))); ax.set_xticklabels([dict((m, l_) for m, l_, *_ in order)[m] for m in ms], rotation=24, ha="right", fontsize=6.4)
ax.set_ylabel("fitted exponent $b$  ($r_t \\approx a\\,t^{-b}$)"); ax.axhline(0, color=GRID, lw=0.6)
ax.legend(loc="upper left", ncol=3, fontsize=6.4)
plabel(ax, "(d) Power-law exponent of the regret curve with 95% CI (dotted: uniform sampling without replacement)")
save(fig, "figS2_regret")

# ------------------------------------------------------------------ S3: landscape smoothness
ls = rd("final/results/landscape_smoothness.csv")
fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.15))
for d in DS:
    g = ls[ls.dataset == d].sort_values("depth")
    axs[0].plot(g.depth, g.eta2, marker=DMK[d], color=DCOL[d], label=DSL[d], ms=3.6)
    axs[1].plot(g.depth, g.within_std, marker=DMK[d], color=DCOL[d], ms=3.6)
    for e, ls_ in [("0.5", "-"), ("1", (0, (4, 1.5))), ("2", (0, (1, 1.2)))]:
        axs[2].plot(g.depth, g[f"N_k_eps{e}"], ls=ls_, marker=DMK[d], color=DCOL[d], ms=2.8, lw=1.0)
axs[0].set_ylabel("variance explained $\\eta^2$"); axs[0].set_ylim(0, 1.05); axs[0].legend(loc="upper left")
axs[1].set_ylabel("within-prefix s.d. (pt)")
axs[2].set_yscale("log"); axs[2].set_ylabel("cells with a leaf within $\\varepsilon$ of $f^*$")
axs[2].legend([Line2D([], [], color=INK, ls=s_, lw=1) for s_ in ["-", (0, (4, 1.5)), (0, (1, 1.2))]],
              ["$\\varepsilon$ = 0.5 pt", "$\\varepsilon$ = 1 pt", "$\\varepsilon$ = 2 pt"], loc="upper left")
for ax, t in zip(axs, ["(a) Prefix locality", "(b) Spread within a prefix", "(c) Near-optimal cells $N_k(\\varepsilon)$"]):
    ax.set_xlabel("prefix depth $k$ (tokens fixed)"); ax.set_xticks(range(1, 7)); plabel(ax, t)
fig.tight_layout(w_pad=1.2)
save(fig, "figS3_landscape")

# ------------------------------------------------------------------ S4: fixed-budget paired differences
st = rd("final/results/statistical_tests.csv"); sts = rd("final/results/equal_compute_statistical_tests_strict.csv")
bo = ["random_search", "mcts_nas", "enas", "llm_nas", "zico", "rz_nas", "re", "re_zc_warmup", "local_search", "archtree_v1", "archtree_zc"]
cells = [(d, b) for d in DS for b in BUD]
fig, axs = plt.subplots(1, 2, figsize=(FULL, 3.05))
norm = TwoSlopeNorm(vmin=-1.6, vcenter=0, vmax=1.6)
for ax, (df, dcol, bcol, tt) in zip(axs, [(st, "mean_diff", "method_b", "(a) Same proposal budget"),
                                          (sts, "mean_diff_archtree_minus_baseline", "baseline", "(b) Baseline given at least as many real evaluations")]):
    M = np.full((len(bo), len(cells)), np.nan); S = np.zeros_like(M, bool); Rw = np.zeros_like(M, bool)
    for i, m in enumerate(bo):
        g = df[df[bcol] == m].copy(); g["ph"] = holm(g.p_value.values)
        for j, (d, b) in enumerate(cells):
            bc = "budget" if "budget" in g.columns else "archtree_nominal_budget"
            x = g[(g.dataset == d) & (g[bc] == b)]
            if len(x): M[i, j] = x.iloc[0][dcol]; S[i, j] = x.iloc[0].ph < .05; Rw[i, j] = x.iloc[0].p_value < .05
    heat(ax, M, [FIX[m] for m in bo], [f"{b}" for d, b in cells], DIV, -1.6, 1.6, fmt="{:+.2f}", fs=5.4, norm=norm, thr=0.62)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): sigmark(ax, i, j, S[i, j], Rw[i, j], None)
    for k in (4, 8): ax.axvline(k - 0.5, color="white", lw=2.2)
    for k, d in enumerate(DS): ax.text(1.5 + 4 * k, -0.95, DSL[d], ha="center", fontsize=6.6)
    ax.set_xlabel("proposal budget $B$"); plabel(ax, tt, pad=14)
axs[1].set_yticklabels([])
fig.tight_layout(w_pad=0.6)
cax = fig.add_axes([0.35, -0.04, 0.3, 0.025])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=DIV), cax=cax, orientation="horizontal")
cb.set_label("mean paired difference, RMG minus baseline (pt);  $\\bullet$ Holm $p<0.05$,  $\\circ$ raw $p<0.05$ only", fontsize=6.6)
cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
save(fig, "figS4_fixed_tests")

# ------------------------------------------------------------------ S5: equal GPU-hours paired differences
ep = rd("equal_gpu_hours/results/egh_paired_tests.csv")
arms = [a for a in EGH if a != "archtree_zc_unsup"]
fig = plt.figure(figsize=(FULL, 3.35))
gsx = fig.add_gridspec(1, 3, width_ratios=[9, 3, 9], wspace=0.08)
norm = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
for n_, (b, dsb, t) in enumerate([("nb201", DS, "(a) NAS-Bench-201"), ("nb101", ["cifar10"], "(b) NB101"), ("natssss", DS, "(c) NATS-Bench SSS")]):
    ax = fig.add_subplot(gsx[n_])
    cl = [(d, k) for d in dsb for k in [25, 50, 100]]
    M = np.full((len(arms), len(cl)), np.nan); S = np.zeros_like(M, bool); Rw = np.zeros_like(M, bool)
    for i, a in enumerate(arms):
        for j, (d, k) in enumerate(cl):
            x = ep[(ep.benchmark == b) & (ep.other == a) & (ep.dataset == d) & (ep.k_mean_arch_costs == k)]
            if len(x): M[i, j] = x.iloc[0].mean_diff_archtree_minus_other; S[i, j] = x.iloc[0].p_holm < .05; Rw[i, j] = x.iloc[0].wilcoxon_p < .05
    heat(ax, M, [EGH[a] for a in arms], [str(k) for d, k in cl], DIV, -2.5, 2.5, fmt="{:+.2f}", fs=5.2, norm=norm, thr=0.62)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): sigmark(ax, i, j, S[i, j], Rw[i, j], None)
    for kk in range(1, len(dsb)): ax.axvline(3 * kk - 0.5, color="white", lw=2.2)
    for kk, d in enumerate(dsb): ax.text(1 + 3 * kk, -0.95, DSS[d], ha="center", fontsize=6.6)
    if n_ > 0: ax.set_yticklabels([])
    ax.set_xlabel("$k$ ($H_k = k\\bar\\tau$)"); plabel(ax, t, pad=14)
cax = fig.add_axes([0.35, -0.035, 0.3, 0.022])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=DIV), cax=cax, orientation="horizontal")
cb.set_label("mean paired difference, RMG minus arm (pt);  $\\bullet$ Holm $p<0.05$,  $\\circ$ raw $p<0.05$ only;  blank: arm not run", fontsize=6.6)
cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
save(fig, "figS5_egh_tests")

# ------------------------------------------------------------------ S6: proposer x gate grid
gsm = rd("gated_baselines/results/gated_summary.csv")
props = [("archtree", "ArchTree-RM"), ("re", "RE"), ("ls", "LS"), ("random", "random"), ("enas", "ENAS-style"), ("mcts", "MCTS-style")]
gates = [("params", "#params"), ("zc", "supervised"), ("zc_unsup", "label-free")]
rows_ = [(p, g) for p, _ in props for g, _ in gates]
M1 = np.full((len(rows_), len(cells)), np.nan); M2 = np.full_like(M1, np.nan)
for i, (p, g) in enumerate(rows_):
    for j, (d, b) in enumerate(cells):
        x = gsm[(gsm.proposer == p) & (gsm.gate == g) & (gsm.dataset == d) & (gsm.budget == b)].iloc[0]
        y = gsm[(gsm.proposer == p) & (gsm.gate == "none") & (gsm.dataset == d) & (gsm.budget == b)].iloc[0]
        M1[i, j] = x.acc_mean - y.acc_mean; M2[i, j] = 100 * (1 - x.train_hours / y.train_hours)
fig, axs = plt.subplots(1, 2, figsize=(FULL, 4.1))
labs = [f"{dict(props)[p]} + {dict(gates)[g]}" for p, g in rows_]
n1 = TwoSlopeNorm(vmin=-1.2, vcenter=0, vmax=1.2)
heat(axs[0], M1, labs, [str(b) for d, b in cells], DIV, -1.2, 1.2, fmt="{:+.2f}", fs=5.0, norm=n1, thr=0.62)
GR = LinearSegmentedColormap.from_list("g", ["#f6faf8", "#9fd3bd", "#13976a", "#0a4f37"])
heat(axs[1], M2, labs, [str(b) for d, b in cells], GR, 0, 80, fmt="{:.0f}", fs=5.2, thr=0.55)
axs[1].set_yticklabels([])
for ax, t in zip(axs, ["(a) Change in test accuracy from adding the gate (pt)", "(b) GPU-hours saved by the gate (%)"]):
    for k in (4, 8): ax.axvline(k - 0.5, color="white", lw=2.2)
    for k in range(3, len(rows_), 3): ax.axhline(k - 0.5, color="white", lw=1.6)
    for k, d in enumerate(DS): ax.text(1.5 + 4 * k, -0.95, DSL[d], ha="center", fontsize=6.6)
    ax.set_xlabel("proposal budget $B$"); plabel(ax, t, pad=14)
fig.tight_layout(w_pad=0.5)
save(fig, "figS6_gate_grid")

# ------------------------------------------------------------------ S7: UCB sensitivity
us = rd("ucb_sensitivity/results/ucb_summary.csv"); ut = rd("ucb_sensitivity/results/ucb_paired_tests.csv")
Cs = sorted(us.C.unique()); cl = [(bk, d, b) for bk in ["mean", "max"] for d in DS for b in [100, 200]]
M = np.full((len(Cs), len(cl)), np.nan)
for i, Cv in enumerate(Cs):
    for j, (bk, d, b) in enumerate(cl):
        ref = us[(us.backup == "max") & (np.isclose(us.C, 0.3)) & (us.dataset == d) & (us.budget == b)].iloc[0].acc_mean
        M[i, j] = us[(us.backup == bk) & (np.isclose(us.C, Cv)) & (us.dataset == d) & (us.budget == b)].iloc[0].acc_mean - ref
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.25), gridspec_kw=dict(width_ratios=[2.2, 1]))
nU = TwoSlopeNorm(vmin=-0.8, vcenter=0, vmax=0.8)
heat(axs[0], M, ["√2" if abs(c - 1.414) < 1e-3 else f"{c:g}" for c in Cs], [str(b) for _, _, b in cl], DIV, -0.8, 0.8, fmt="{:+.2f}", fs=5.3, norm=nU, thr=0.62)
for k in range(2, 12, 2): axs[0].axvline(k - 0.5, color="white", lw=1.4 if k != 6 else 3)
for k, (bk, d) in enumerate([(bk, d) for bk in ["mean", "max"] for d in DS]): axs[0].text(0.5 + 2 * k, -0.9, DSS[d], ha="center", fontsize=6.4)
axs[0].text(2.5, -1.75, "mean backup", ha="center", fontsize=6.8, fontweight="bold"); axs[0].text(8.5, -1.75, "max backup", ha="center", fontsize=6.8, fontweight="bold")
axs[0].set_ylabel("exploration constant $C$"); axs[0].set_xlabel("proposal budget $B$")
plabel(axs[0], "(a) Test accuracy minus default ($C$ = 0.3, max) (pt)", pad=31)
ax = axs[1]
pv = ut.p_value.values
ax.hist(pv, bins=np.linspace(0, 1, 21), color=C["arch"], alpha=0.85, edgecolor="white", lw=0.4)
ax.axvline(0.05, color=C["re"], lw=0.9, ls=(0, (3, 1.5)))
ax.text(0.3, ax.get_ylim()[1] * 0.92, f"{(pv < 0.05).sum()} of {len(pv)} raw $p<0.05$\n{(ut.p_holm < 0.05).sum()} after Holm", fontsize=6.4, va="top")
ax.set_xlabel("raw $p$ (vs default)"); ax.set_ylabel("tests"); plabel(axs[1], "(b) Distribution of $p$-values")
fig.tight_layout(w_pad=1.0)
save(fig, "figS7_ucb")

# ------------------------------------------------------------------ S8: proxy subsets
pa = rd("proxy_ablation/results/proxy_ablation_summary.csv"); prs = rd("proxy_ablation/results/proxy_ablation_rank_vs_search.csv").set_index("dataset")
fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.3))
szc = {1: C["rnd"], 2: C["y"], 3: C["re"], 4: C["warm"]}
for ax, d in zip(axs, DS):
    g = pa[(pa.dataset == d) & (~pa.arm.isin(["none", "saved"]))]
    for _, x in g.iterrows():
        ax.errorbar(x.kendall_tau, x.best_acc_mean, yerr=x.best_acc_std / np.sqrt(x.n_seeds), fmt="o", ms=3.6 + 0.6 * x["size"], color=szc[int(x["size"])],
                    mec="white", mew=0.4, lw=0.6, capsize=0, zorder=3)
        ax.annotate(x.arm, (x.kendall_tau, x.best_acc_mean), xytext=(3, 2), textcoords="offset points", fontsize=5.4, color=MUTED)
    n0 = pa[(pa.dataset == d) & (pa.arm == "none")].iloc[0]
    ax.axhline(n0.best_acc_mean, color=INK, lw=0.7, ls=(0, (3, 1.5)))
    ax.text(ax.get_xlim()[0], n0.best_acc_mean, " no gate", fontsize=5.8, va="bottom", color=INK)
    rr = prs.loc[d]
    ax.text(0.98, 0.04, f"Spearman($\\tau$, acc.) = {rr.rho_tau_vs_best_acc:.2f} ($p$ = {rr.p_tau:.2f})", transform=ax.transAxes, ha="right", fontsize=6.0)
    ax.set_xlabel("Kendall $\\tau$ of the gate score with validation acc."); plabel(ax, f"({'abc'[DS.index(d)]}) {DSL[d]}, $B$ = 100")
axs[0].set_ylabel("test accuracy (%), mean $\\pm$ s.e.")
axs[2].legend([Line2D([], [], marker="o", ls="", color=szc[k], ms=3.6 + 0.6 * k) for k in szc], [f"{k} prox{'y' if k == 1 else 'ies'}" for k in szc], loc="upper left", fontsize=6)
fig.tight_layout(w_pad=0.6)
save(fig, "figS8_proxy_subsets")

# ------------------------------------------------------------------ S9: gate-cost validation
gv = rd("equal_gpu_hours/results/gate_cost_validation.csv")
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.45), gridspec_kw=dict(width_ratios=[1, 1.25]))
BC = {"nb201": C["arch"], "nb101": C["warm"], "natssss": C["ls"]}
PM = {"archtree": "o", "random": "s"}
ax = axs[0]
for _, x in gv.iterrows():
    ax.scatter(x.predicted_delta, x.measured_delta, s=14 if x["lambda"] == 1 else 8, marker=PM[x.proposer], color=BC[x.benchmark],
               alpha=0.9 if x["lambda"] == 1 else 0.5, edgecolors="white", linewidths=0.3, zorder=3)
lim = [-2.8, 0.5]
ax.set_xlim(-0.75, 0.25); ax.set_ylim(-2.8, 0.5)
ax.plot(lim, lim, color=MUTED, lw=0.6, ls=(0, (2, 1.5))); ax.axhline(0, color=GRID, lw=0.6); ax.axvline(0, color=GRID, lw=0.6)
from scipy.stats import spearmanr
rs = spearmanr(gv.predicted_delta, gv.measured_delta)
agree = int(((np.sign(gv.predicted_delta) == np.sign(gv.measured_delta))).sum())
ax.text(0.03, 0.97, f"Spearman $\\rho$ = {rs.correlation:.2f}\nsign agrees in {agree}/{len(gv)} cells", transform=ax.transAxes, va="top", fontsize=6.3)
ax.set_xlabel("predicted change (pt), Gaussian copula"); ax.set_ylabel("measured change at equal GPU-h (pt)")
ax.legend([Line2D([], [], marker="o", ls="", color=BC[b]) for b in BC] + [Line2D([], [], marker=PM[p], ls="", color=MUTED) for p in PM],
          ["NB201", "NB101", "NATS-SSS", "ArchTree proposer", "random proposer"], loc="lower left", fontsize=5.8)
plabel(ax, "(a) Predicted vs measured effect of a cost penalty")
ax = axs[1]
gm = gv.groupby(["benchmark", "dataset", "lambda"]).agg(meas=("measured_delta", "mean"), pred=("predicted_delta", "mean")).reset_index()
ordr = [("nb201", "cifar10"), ("nb201", "cifar100"), ("nb201", "ImageNet16-120"), ("nb101", "cifar10"), ("natssss", "cifar10"), ("natssss", "cifar100"), ("natssss", "ImageNet16-120")]
labs = [f"{BL[b]}\n{DSS[d]}" for b, d in ordr]
xs = np.arange(len(labs))
for off, lam, al_ in [(-0.2, 0.25, 0.55), (0.2, 1.0, 1.0)]:
    h = gm[gm["lambda"] == lam]
    hh = [h[(h.benchmark == b) & (h.dataset == d)] for b, d in ordr]
    ax.bar(xs + off, [x.meas.iloc[0] if len(x) else np.nan for x in hh], width=0.38, color=C["cost"], alpha=al_, label=f"measured, $\\lambda$ = {lam:g}")
    ax.scatter(xs + off, [x.pred.iloc[0] if len(x) else np.nan for x in hh], marker="_", s=60, color=INK, lw=1.2, zorder=4, label="predicted" if lam == 1 else None)
ax.axhline(0, color=INK, lw=0.6); ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=6.3, linespacing=1.0)
ax.set_ylabel("change in test accuracy (pt)"); ax.legend(loc="lower left", fontsize=6)
plabel(ax, "(b) Mean effect per space (averaged over $k$ and proposers)")
fig.tight_layout(w_pad=1.0)
save(fig, "figS9_gatecost_validation")

# ------------------------------------------------------------------ S10: LLM prior strength and pruning
al = rd("alpha_sweep/results/alpha_vs_baseline_stats.csv"); a5 = rd("ablations/results/ablation_A5_prune_threshold.csv")
fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.2), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
for k, b in enumerate([50, 100]):
    ax = axs[k]
    for d in DS:
        g = al[(al.dataset == d) & (al.budget == b)].sort_values("alpha")
        xs = np.r_[0, g.alpha.values]; ys = np.r_[0, g.mean_diff_vs_alpha0.values]
        ax.plot(xs, ys, marker=DMK[d], color=DCOL[d], ms=3.2, label=DSL[d])
        raw = g[g.p_value < 0.05]
        ax.scatter(raw.alpha, raw.mean_diff_vs_alpha0, s=26, facecolors="none", edgecolors=INK, lw=0.6, zorder=5)
    ax.axhline(0, color=INK, lw=0.6); ax.axvspan(0.2, 1.02, color=GRID, alpha=0.5, lw=0)
    ax.text(0.6, 0.5, "identical runs for $\\alpha\\geq0.2$", ha="center", va="center", transform=ax.get_xaxis_transform(), fontsize=6, color=MUTED)
    ax.set_xlabel("prior strength $\\alpha$"); ax.set_xlim(-0.02, 1.02)
    plabel(ax, f"({'ab'[k]}) Change vs $\\alpha$ = 0, $B$ = {b}")
axs[0].set_ylabel("test accuracy difference (pt)")
axs[1].text(0.98, 0.03, "circled: raw $p<0.05$ (none after Holm)", transform=axs[1].transAxes, ha="right", fontsize=5.8, color=MUTED)
ax = axs[2]
for d in DS:
    g = a5[a5.dataset == d].groupby("param_value").best_test_acc.agg(["mean", "std", "count"]).reset_index()
    base = g[g.param_value == 15]["mean"].iloc[0]
    ax.errorbar(g.param_value, g["mean"] - base, yerr=g["std"] / np.sqrt(g["count"]), marker=DMK[d], color=DCOL[d], ms=3.2, capsize=1.5, lw=1)
ax.axhline(0, color=INK, lw=0.6); ax.set_xticks([0, 15, 25, 35]); ax.set_xlabel("share of nodes pruned every 50 steps (%)")
plabel(ax, "(c) Pruning (A5), vs default 15%, $B$ = 100")
ax.legend([Line2D([], [], color=DCOL[d], marker=DMK[d], ms=3) for d in DS], [DSL[d] for d in DS], loc="lower left", fontsize=6)
fig.tight_layout(w_pad=0.8)
save(fig, "figS10_alpha_prune")

# ------------------------------------------------------------------ S11: convergence distributions
cmx = rd("convergence/results/convergence_per_run_metrics.csv")
CA = [("llm_off", "prior off", C["rnd"], "-"), ("llm_on", "LLM prior", C["warm"], "-"), ("llm_described", "LLM, described ops", C["cost"], (0, (4, 1.5))),
      ("size_prior", "size prior (no LLM)", C["y"], (0, (1, 1.2)))]
fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.2))
for ax, d in zip(axs, DS):
    for a, lab_, c, ls_ in CA:
        g = cmx[(cmx.arm == a) & (cmx.dataset == d) & (cmx.budget == 100)]
        v = np.sort(np.where(g["reached_1.0pp"], g["evals_to_1.0pp"], np.inf))
        xs = np.r_[0, np.repeat(v[np.isfinite(v)], 2), 60]
        ys = np.r_[0, 0, np.repeat(np.arange(1, np.isfinite(v).sum() + 1) / len(v), 2)[:-1], np.isfinite(v).sum() / len(v)]
        ax.plot(xs, ys[:len(xs)], color=c, ls=ls_, lw=1.4 if a in ("llm_on", "llm_off") else 1.1, label=lab_)
    ax.set_xlim(0, 55); ax.set_ylim(0, 1.03); ax.set_xlabel("real evaluations to within 1 pt of final")
    plabel(ax, f"({'abc'[DS.index(d)]}) {DSL[d]}, $B$ = 100")
axs[0].set_ylabel("share of the 12 seeds"); axs[0].legend(loc="lower right", fontsize=6)
fig.tight_layout(w_pad=0.6)
save(fig, "figS11_convergence_ecdf")

# ------------------------------------------------------------------ S12: LLM knowledge probes
cs_ = rd("llm_contamination/results/contamination_summary.csv"); lsz = rd("llm_size/results/llm_size_summary.csv")
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax = axs[0]
modes = [("standard", "standard\n(op names)", C["warm"]), ("described", "ops described,\nnames hidden", C["cost"]),
         ("anonymous", "anonymous\ntokens", C["rnd"]), ("size_prior", "size prior\n(no LLM)", C["y"])]
xs = np.arange(len(DS))
for k, (m, lab_, c) in enumerate(modes):
    g = cs_[cs_["mode"] == m].set_index("dataset").loc[DS]
    ax.bar(xs + (k - 1.5) * 0.2, g.spearman_mean, width=0.19, color=c, label=lab_.replace("\n", " "))
    for x_, v in zip(xs + (k - 1.5) * 0.2, g.spearman_mean): ax.text(x_, v + 0.015, f"{v:.2f}", ha="center", fontsize=5.3)
ax.set_xticks(xs); ax.set_xticklabels([DSL[d] for d in DS]); ax.set_ylabel("mean Spearman $\\rho$ (op ranking)")
ax.set_ylim(0, 1.08); ax.legend(loc="upper left", ncol=2, fontsize=5.8); ax.axhline(0, color=INK, lw=0.6)
plabel(ax, "(a) What the LLM knows about NAS-Bench-201 (56 prefixes per bar)")
ax = axs[1]
lsz["nm"] = lsz.model.str.split(":").str[1].str.replace("openai/", "")
xs = np.arange(len(lsz))
ax.bar(xs - 0.18, lsz.spearman_mean, width=0.34, color=C["warm"], label="mean $\\rho$")
ax.bar(xs + 0.18, lsz.top1_hit_rate, width=0.34, color=C["s1"], label="top-1 hit rate")
for x_, v, w in zip(xs, lsz.spearman_mean, lsz.top1_hit_rate):
    ax.text(x_ - 0.18, v + 0.015, f"{v:.2f}", ha="center", fontsize=5.6); ax.text(x_ + 0.18, w + 0.015, f"{w:.2f}", ha="center", fontsize=5.6)
ax.set_xticks(xs); ax.set_xticklabels(lsz.nm, fontsize=6.4); ax.set_ylim(0, 0.8); ax.legend(loc="upper left", fontsize=6)
plabel(ax, "(b) Model size (CIFAR-10, 26 prefixes)")
fig.tight_layout(w_pad=1.0)
save(fig, "figS12_llm_probes")

# ------------------------------------------------------------------ S13: hardware IGD and front size per device
hv = rd("phase4/results/phase4_hv_igd_summary.csv"); hb = rd("phase4/results/phase4_baselines_hv_igd_summary.csv")
HW = ["edgegpu", "raspi4", "edgetpu", "pixel3", "eyeriss", "fpga"]
HWN = {"edgegpu": "Edge GPU", "raspi4": "Raspberry Pi 4", "edgetpu": "Edge TPU", "pixel3": "Pixel 3", "eyeriss": "Eyeriss", "fpga": "FPGA"}
fig, axs = plt.subplots(2, 6, figsize=(FULL, 3.3), sharex=True)
src = [("ArchTree-HW", hv, C["arch"], "o"), ("NSGA-II-style", hb[hb.method == "nsga2_hw"], C["re"], "s"), ("random", hb[hb.method == "random_hw"], C["rnd"], "^")]
for j, h in enumerate(HW):
    for lab_, df, c, mk in src:
        g = df[df.hardware == h].groupby("budget")
        m_ = g.igd.mean(); se = g.igd.std() / np.sqrt(g.igd.count())
        axs[0, j].errorbar(m_.index, m_.values, yerr=se.values, color=c, marker=mk, ms=2.8, lw=1, capsize=1.2, label=lab_)
        f_ = g.pareto_front_size.mean()
        axs[1, j].plot(f_.index, f_.values, color=c, marker=mk, ms=2.8, lw=1)
    axs[0, j].set_title(HWN[h], fontsize=7)
    for i in range(2): logx_budget(axs[i, j]); axs[i, j].tick_params(labelsize=6)
    axs[1, j].set_xlabel("budget $B$", fontsize=6.6)
axs[0, 0].set_ylabel("IGD (lower = better)"); axs[1, 0].set_ylabel("found front size")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper right", ncol=3, fontsize=6.2, bbox_to_anchor=(1.0, 1.0))
fig.text(0.005, 0.98, r"$\mathbf{(a)}$ IGD to the true front (mean $\pm$ s.e., 12 seeds)", fontsize=7.4, va="top")
fig.text(0.005, 0.5, r"$\mathbf{(b)}$ Number of non-dominated architectures found", fontsize=7.4, va="top")
fig.tight_layout(h_pad=2.2, w_pad=0.4, rect=(0, 0, 1, 0.96))
save(fig, "figS13_hw_igd")

# ------------------------------------------------------------------ S14: NAS-Bench-101 by gate
nb = rd("nb101/results/nb101_summary.csv")
fig, axs = plt.subplots(1, 4, figsize=(FULL, 2.1), sharey=True)
GN = [("none", "(a) no gate"), ("params", "(b) #params gate"), ("zc", "(c) supervised gate"), ("zc_unsup", "(d) label-free gate")]
PC = [("archtree", "ArchTree-RM", C["arch"], "o"), ("re", "RE", C["re"], "s"), ("random", "random", C["rnd"], "^")]
for ax, (g_, t) in zip(axs, GN):
    for p, lab_, c, mk in PC:
        h = nb[(nb.gate == g_) & (nb.proposer == p)].sort_values("budget")
        ax.errorbar(h.budget, h.test_mean, yerr=h.test_std / np.sqrt(h.n), color=c, marker=mk, ms=3, lw=1.1, capsize=1.3, label=lab_)
        if g_ != "none" and p == "archtree":
            for _, x in h.iterrows(): ax.annotate(f"{x.real_evals:.0f}", (x.budget, x.test_mean), xytext=(6, -8), textcoords="offset points", fontsize=5.2, ha="center", color=C["arch"])
    logx_budget(ax, (50, 100, 200, 400)); ax.set_xlabel("proposal budget $B$"); plabel(ax, t)
axs[0].set_ylabel("test accuracy (%)"); axs[0].legend(loc="upper left", fontsize=6)
fig.text(0.99, 0.01, "numbers: real evaluations of ArchTree-RM", ha="right", fontsize=5.8, color=MUTED)
fig.tight_layout(w_pad=0.4)
save(fig, "figS14_nb101")

# ------------------------------------------------------------------ S15: volcano of every accuracy comparison
fam = []
t = st[st.method_b.isin(bo[:9])].copy()
for m, g in t.groupby("method_b"): fam.append(pd.DataFrame(dict(d=g.mean_diff, p=g.p_value, ph=holm(g.p_value.values), f="fixed budget vs 9 baselines")))
g = rd("gated_baselines/results/gated_paired_tests.csv"); fam.append(pd.DataFrame(dict(d=g.mean_diff_archtree_minus_other, p=g.p_value, ph=g.p_holm, f="same gate, 5 proposers")))
g = ep; fam.append(pd.DataFrame(dict(d=g.mean_diff_archtree_minus_other, p=g.wilcoxon_p, ph=g.p_holm, f="equal GPU-hours")))
g = rd("nb101/results/nb101_paired_tests.csv"); fam.append(pd.DataFrame(dict(d=g.mean_diff_archtree_minus_other, p=g.p_value, ph=g.p_holm, f="NB101 (same gate + matched evals)")))
g = rd("nb101/results/nb101_equal_compute_tests.csv"); fam.append(pd.DataFrame(dict(d=g.mean_diff_archtree_minus_other, p=g.p_value, ph=g.p_holm, f="NB101 (same gate + matched evals)")))
g = rd("design_ablation/results/design_paired_tests.csv"); fam.append(pd.DataFrame(dict(d=-g.mean_diff_arm_minus_v2, p=g.wilcoxon_p, ph=g.p_holm, f="design ablation + A7")))
g = rd("ablations/results/A7_paired_tests.csv")
for o_, h_ in g.groupby("arm_b"): fam.append(pd.DataFrame(dict(d=h_.mean_diff_pp, p=h_.wilcoxon_p, ph=holm(h_.wilcoxon_p.values), f="design ablation + A7")))
V = pd.concat(fam, ignore_index=True)
FC = {"fixed budget vs 9 baselines": C["re"], "same gate, 5 proposers": C["ls"], "equal GPU-hours": C["arch"],
      "NB101 (same gate + matched evals)": C["warm"], "design ablation + A7": C["y"]}
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]
for f_, c in FC.items():
    h = V[V.f == f_]
    ns = h[h.ph >= .05]; sg = h[h.ph < .05]
    ax.scatter(ns.d, -np.log10(ns.p), s=7, facecolors="none", edgecolors=c, lw=0.5, alpha=0.75)
    ax.scatter(sg.d, -np.log10(sg.p), s=10, color=c, edgecolors="white", lw=0.25, label=f"{f_.replace(' (same gate + matched evals)', '').replace(', 5 proposers', '')} ({len(h)})")
ax.axhline(-np.log10(0.05), color=MUTED, lw=0.6, ls=(0, (3, 1.5))); ax.axvline(0, color=INK, lw=0.6)
ax.text(-3.5, -np.log10(0.05) + 0.05, "raw $p$ = 0.05", fontsize=5.8, color=MUTED, va="bottom")
ax.set_xlabel("mean paired difference, ArchTree minus other (pt)"); ax.set_ylabel("$-\\log_{10}$ raw $p$")
ax.legend(loc="upper left", fontsize=5.6, title="filled: Holm $p<0.05$", title_fontsize=5.6, borderaxespad=0.1, handletextpad=0.2)
ax.set_xlim(-3.6, 3.2)
plabel(ax, f"(a) All {len(V)} accuracy comparisons")
ax = axs[1]
cats = list(FC)
w_ = [int(((V.f == c_) & (V.ph < .05) & (V.d > 0)).sum()) for c_ in cats]
l_ = [int(((V.f == c_) & (V.ph < .05) & (V.d < 0)).sum()) for c_ in cats]
n_ = [int((V.f == c_).sum()) for c_ in cats]
ys = np.arange(len(cats))[::-1]
ax.barh(ys, n_, color="#ededeb", height=0.62, label="tie")
ax.barh(ys, w_, color=C["arch"], height=0.62, label="ArchTree better")
ax.barh(ys, l_, left=w_, color=C["re"], height=0.62, label="other better")
for y_, n0, w0, l0 in zip(ys, n_, w_, l_): ax.text(n0 + 4, y_, f"{w0}/{n0 - w0 - l0}/{l0}", va="center", fontsize=6)
ax.set_yticks(ys); ax.set_yticklabels([c_.replace(" (same gate + matched evals)", "").replace(", 5 proposers", "") for c_ in cats], fontsize=6.4)
ax.set_xlabel("tests (W/T/L after Holm)"); ax.set_xlim(0, max(n_) * 1.32); ax.legend(loc="lower right", fontsize=5.8); ax.grid(axis="y", lw=0)
plabel(ax, "(b) Outcome by family")
fig.tight_layout(w_pad=1.0)
save(fig, "figS15_volcano")
json.dump(dict(n_tests=len(V), wins=int(((V.ph < .05) & (V.d > 0)).sum()), losses=int(((V.ph < .05) & (V.d < 0)).sum()),
               by_family={c_: [w0, n0 - w0 - l0, l0] for c_, w0, l0, n0 in zip(cats, w_, l_, n_)},
               gatecost_rho=rs.correlation, gatecost_sign=[agree, len(gv)]), open(os.path.join(OUT, "fig_facts.json"), "w"), indent=1)

# ------------------------------------------------------------------ S16: repeated evaluations
du = rd("final/results/duplicate_evals_summary.csv")
fig, ax = plt.subplots(figsize=(COL, 2.2))
DM = [("random_search", "Random search", C["rnd"], "-"), ("mcts_nas", "MCTS-style", C["y"], "-"), ("enas", "ENAS-style", "#c9a227", (0, (4, 1.5))),
      ("llm_nas", "LLM-NAS", C["warm"], "-"), ("re", "RE", C["re"], "-"), ("local_search", "Local search", C["ls"], "-"),
      ("flat_ucb_none[A7]", "flat UCB, no dedup", INK, (0, (1, 1.2))), ("archtree_llm", "ArchTree (all stages)", C["arch"], "-")]
for m, lab_, c, ls_ in DM:
    g = du[du.method == m].groupby("budget").dup_rate_mean.mean() * 100
    ax.plot(g.index, g.values, color=c, ls=ls_, marker="o", ms=2.6, lw=1.6 if m == "archtree_llm" else 1.0, label=lab_)
logx_budget(ax); ax.set_xlabel("proposal budget $B$"); ax.set_ylabel("repeated real evaluations (%)")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6)
save(fig, "figS16_duplicates")
print("figures done")
