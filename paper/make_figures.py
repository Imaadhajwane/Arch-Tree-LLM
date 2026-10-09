"""Journal figures for the ArchTree paper (Sadhana), version 2 (unified style).
Every value is read from the experiment result CSVs (outputs/) or from tables derived from the
benchmark caches by analysis/derive.py (analysis/out/). Lettering is lower case except symbols
and acronyms (journal rule); vector PDF plus 300 dpi PNG.
usage: python3 make_figures.py <outputs dir> <figs dir> <analysis/out dir> <data dir>"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

R, OUT, A, DATA = sys.argv[1:5]
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
ra = lambda p: pd.read_csv(os.path.join(A, p))
BLUES = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9ec3ec", "#2a78d6", "#123f78"])
DIV = LinearSegmentedColormap.from_list("d", ["#b8322f", "#f0b8a8", "#f7f7f5", "#a9c9ef", "#1d5fae"])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.03, dpi=600)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.03, dpi=600)
    plt.close(fig)


import re as _re
def plabel(ax, s, fs=8):
    m_ = _re.match(r"^\(([a-z])\)\s*(.*)$", s, _re.S)
    if m_:
        s = r"$\mathbf{(" + m_.group(1) + r")}$ " + m_.group(2)
    ax.set_title(s, loc="left", fontsize=fs, pad=4)


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


# =============================================================== F3: dataset characterisation
sa = np.load(os.path.join(A, "space_arrays.npz"))
fig = plt.figure(figsize=(FULL, 3.5))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.02], hspace=0.62, wspace=1.6)
ax1 = fig.add_subplot(gs[0, :3]); ax2 = fig.add_subplot(gs[0, 3:])
pairs = [("NB201", d, sa["nb201_t"][i], sa["nb201_s"][i]) for i, d in enumerate(DS)] + \
        [("NB101", "cifar10", sa["nb101_t"], sa["nb101_s"])] + \
        [("NATS", d, sa["nats_t"][i], sa["nats_s"][i]) for i, d in enumerate(DS)]
bcol = {"NB201": C["arch"], "NB101": C["re"], "NATS": C["ls"]}
labs = [f"{b}\n{DSS[d]}" for b, d, _, _ in pairs]
for k, (ax, key, ylab) in enumerate([(ax1, 2, "test accuracy (%)"), (ax2, 3, "training time per architecture (h)")]):
    data = [p[key] if key == 2 else p[key] / 3600 for p in pairs]
    vp = ax.violinplot(data, positions=range(len(pairs)), widths=0.8, showextrema=False, bw_method=0.25)
    for b, (bn, *_ ) in zip(vp["bodies"], pairs):
        b.set_facecolor(bcol[bn]); b.set_alpha(0.55); b.set_edgecolor("none")
    for i, x in enumerate(data):
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        ax.vlines(i, q1, q3, color=INK, lw=2.2); ax.scatter([i], [med], s=7, c="white", zorder=3, lw=0)
        if key == 2:
            ax.scatter([i], [x.max()], marker="_", s=40, c=INK, lw=1)
            ax.text(i, x.max() + 2.2, f"{x.max():.1f}", ha="center", va="bottom", fontsize=5.2, color=MUTED)
            ax.text(i + 0.12, med - 7, f"{med:.1f}", ha="left", va="top", fontsize=5.2, color=INK)
        else:
            ax.text(i + 0.12, med, f"{med:.2f}" if med < 1 else f"{med:.1f}", ha="left", va="center", fontsize=5.2, color=INK)
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels(labs, fontsize=6.4)
    ax.set_ylabel(ylab)
    if key == 3: ax.set_yscale("log")
ax1.set_ylim(0, 106)
plabel(ax1, "(a) test accuracy (dot: median; bar: IQR; dash: best)")
plabel(ax2, "(b) training time per architecture (median shown)")
hex_specs = [("NB201, CIFAR-10", sa["nb201_s"][0] / 3600, sa["nb201_t"][0], C["arch"], (80, 95)),
             ("NB101, CIFAR-10", sa["nb101_s"] / 3600, sa["nb101_t"], C["re"], (80, 95)),
             ("NATS-SSS, CIFAR-10", sa["nats_s"][0] / 3600, sa["nats_t"][0], C["ls"], (86, 94))]
for j, (t, x, y, c, yl) in enumerate(hex_specs):
    ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
    cm = LinearSegmentedColormap.from_list("h", ["#f7f7f5", c, "#10243f"])
    m = (y >= yl[0])
    hb = ax.hexbin(x[m], y[m], gridsize=34, bins="log", cmap=cm, mincnt=1, linewidths=0.1, rasterized=True)
    from scipy.stats import spearmanr
    r = spearmanr(x, y).statistic
    qb = np.quantile(x, np.linspace(0, 1, 21)); mid = []; med_ = []; p25 = []; p75 = []
    for lo, hi in zip(qb[:-1], qb[1:]):
        sel = (x >= lo) & (x <= hi)
        if sel.sum() > 20:
            mid.append(np.median(x[sel])); med_.append(np.median(y[sel])); p25.append(np.percentile(y[sel], 25)); p75.append(np.percentile(y[sel], 75))
    ax.plot(mid, med_, color="white", lw=2.6, zorder=4); ax.plot(mid, med_, color=INK, lw=1.1, zorder=5, label="binned median")
    ax.plot(mid, p75, color=INK, lw=0.6, ls="--", zorder=5, label="upper quartile")
    if j == 0: ax.legend(loc="lower right", fontsize=5.4, handlelength=1.6, framealpha=0.85, frameon=True, edgecolor="none")
    ax.set_xlabel("training time (h)"); ax.set_ylim(*yl); ax.grid(False)
    if j == 0: ax.set_ylabel("test accuracy (%)")
    plabel(ax, f"({'cde'[j]}) {t} ($\\rho$ = {r:.2f})")
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.06); cb.ax.tick_params(labelsize=5.5, length=1.5); cb.outline.set_linewidth(0.3)
save(fig, "fig3_datasets")

# =============================================================== F4: proxies (quality, cost, anatomy)
pc = rd("phase2/results/phase2_proxy_calibration.csv")
cc = rd("equal_gpu_hours/results/cost_correlation.csv")
pcorr = ra("proxy_corr.csv")
fig = plt.figure(figsize=(FULL, 3.25))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.08], hspace=0.62, wspace=0.28)
ax = fig.add_subplot(gs[0, 0])
names = [("E", "SVD\nentropy"), ("S", "SynFlow"), ("T", "NASWOT"), ("ensemble_unsupervised", "label-free\nensemble"), ("ensemble", "supervised\nensemble")]
w = 0.26; x = np.arange(len(names))
for i, d in enumerate(DS):
    v = [pc[(pc.dataset == d) & (pc.proxy_name == n)].kendall_tau.iloc[0] for n, _ in names]
    ax.bar(x + (i - 1) * w, v, w * 0.9, color=DCOL[d], label=DSL[d], lw=0)
    for xx, vv in zip(x + (i - 1) * w, v):
        ax.text(xx, vv + 0.012, f"{vv:.2f}", ha="center", va="bottom", fontsize=4.9, rotation=90, color=MUTED)
ax.axvspan(2.5, 4.5, color="#f1f5fb", zorder=0, lw=0)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in names], fontsize=6.6); ax.set_ylim(0, 1.08)
ax.set_ylabel(r"Kendall $\tau$ with validation acc."); plabel(ax, "(a) ranking quality, NB201 (1,500 architectures)")
ax.legend(loc="upper center", ncol=3, fontsize=6.2, columnspacing=0.9, handlelength=1.0, bbox_to_anchor=(0.5, 1.02))
ax = fig.add_subplot(gs[0, 1])
sc = [("E", "SVD\nentropy"), ("S", "SynFlow"), ("T", "NASWOT"), ("ensemble_unsup", "label-free\nensemble"), ("params", "#params")]
benches = [("nb201", "cifar10", "NB201", C["arch"]), ("nb101", "cifar10", "NB101", C["re"]), ("natssss", "cifar10", "NATS-SSS", C["ls"])]
for i, (bn, d, lab, c) in enumerate(benches):
    v = [cc[(cc.benchmark == bn) & (cc.dataset == d) & (cc.score == s)].spearman_with_train_time.iloc[0] for s, _ in sc]
    ax.bar(x + (i - 1) * w, v, w * 0.9, color=c, label=lab, lw=0)
    for xx, vv in zip(x + (i - 1) * w, v):
        ax.text(xx, vv + 0.015, f"{vv:.2f}", ha="center", va="bottom", fontsize=4.9, rotation=90, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in sc], fontsize=6.6); ax.set_ylim(0, 1.3)
ax.set_ylabel(r"Spearman $\rho$ with training time"); plabel(ax, "(b) proxies as cost detectors (CIFAR-10)")
ax.legend(loc="upper center", ncol=3, fontsize=6.2, columnspacing=0.9, handlelength=1.0, bbox_to_anchor=(0.5, 1.02))
order = ["E", "S", "T", "P", "acc", "time"]
nice = ["$E$", "$S$", "$T$", "$P$", "acc.", "time"]
for j, (bn, ds, ttl) in enumerate([("NB201", "cifar10", "(c) Spearman correlations, NB201 CIFAR-10 (all 15,625)"),
                                   ("NATS-SSS", "cifar10", "(d) Spearman correlations, NATS-SSS CIFAR-10 (all 32,768)")]):
    ax = fig.add_subplot(gs[1, j])
    t = pcorr[(pcorr.benchmark == bn) & (pcorr.dataset == ds)]
    M = np.array([[t[(t.x == a) & (t.y == b)].rho.iloc[0] for b in order] for a in order])
    M[np.triu_indices(len(order), 1)] = np.nan
    im = heat(ax, M, nice, nice, DIV, -1, 1, norm=TwoSlopeNorm(0, -1, 1), fs=6.4, thr=0.55)
    ax.add_patch(plt.Rectangle((2.5, 3.5), 1, 2, fill=False, ec=C["cost"], lw=1.1))
    plabel(ax, ttl)
cb = fig.colorbar(im, ax=fig.axes[-2:], fraction=0.025, pad=0.02); cb.ax.tick_params(labelsize=6, length=1.5); cb.outline.set_linewidth(0.3)
cb.set_label(r"$\rho$", fontsize=7)
save(fig, "fig4_proxies")

# =============================================================== F5: why these settings (gate curve, UCB, proxies)
go = ra("gate_operating.csv")
fig, axs = plt.subplots(2, 2, figsize=(FULL, 3.6), gridspec_kw=dict(hspace=0.6, wspace=0.28))
ax = axs[0, 0]
for d in DS:
    for bn, ls in [("NB201", "-"), ("NATS-SSS", "--")]:
        t = go[(go.benchmark == bn) & (go.dataset == d)]
        ax.plot(t.q * 100, t.recall_top5 * 100, ls=ls, color=DCOL[d], lw=1.3, marker="o" if bn == "NB201" else None, ms=2.2)
ax.plot([0, 100], [0, 100], color=C["rnd"], lw=0.8, ls=":")
ax.axvline(35, color=INK, lw=0.7); ax.text(36.5, 52, "pass rate 35%\n(used)", fontsize=6.4, color=INK)
ax.set_xlim(0, 100); ax.set_ylim(0, 104); ax.set_xlabel("gate pass rate (% of space admitted)"); ax.set_ylabel("top-5% architectures kept (%)")
h = [Line2D([], [], color=DCOL[d], lw=1.3) for d in DS] + [Line2D([], [], color=MUTED, ls=s_, lw=1.0) for s_ in ["-", "--"]] + [Line2D([], [], color=C["rnd"], ls=":", lw=0.8)]
ax.legend(h, [DSL[d] for d in DS] + ["NB201", "NATS-SSS", "random gate"], loc="lower right", fontsize=6.2, ncol=2, columnspacing=0.8)
ax.grid(True, axis="both"); plabel(ax, "(a) gate operating curve over the full spaces")
ax = axs[0, 1]
for bn, ls in [("NB201", "-"), ("NATS-SSS", "--")]:
    t = go[(go.benchmark == bn) & (go.dataset == "cifar10")]
    ax.plot(t.q * 100, t.rel_cost_kept, ls=ls, color=C["cost"], lw=1.4, label=f"{bn}: admitted by proxy")
    ax.plot(t.q * 100, t.rel_cost_oracle, ls=ls, color=INK, lw=1.0, label=f"{bn}: true top by accuracy")
ax.axvline(35, color=INK, lw=0.7); ax.axhline(1, color=C["rnd"], lw=0.7, ls=":")
ax.set_xlim(0, 100); ax.set_xlabel("gate pass rate (% of space admitted)"); ax.set_ylabel("mean training time / space mean")
ax.legend(fontsize=6.2, loc="upper right"); ax.grid(True, axis="both"); plabel(ax, "(b) training cost of what passes (CIFAR-10)")
ax = axs[1, 0]
u = rd("ucb_sensitivity/results/ucb_summary.csv")
Cs = sorted(u.C.unique())
base = u[(u.C == 0.3) & (u.backup == "max")].set_index(["dataset", "budget"]).acc_mean
M = np.array([[(u[(u.C == c) & (u.backup == b)].set_index(["dataset", "budget"]).acc_mean - base).mean() for c in Cs] for b in ["max", "mean"]])
heat(ax, M, ["max backup", "mean backup"], [f"{c:g}" for c in Cs], DIV, -0.35, 0.35, fmt="{:+.2f}", fs=6.6, norm=TwoSlopeNorm(0, -0.35, 0.35), thr=0.7)
ax.add_patch(plt.Rectangle((Cs.index(0.3) - 0.5, -0.5), 1, 1, fill=False, ec=INK, lw=1.3))
ax.set_xlabel("UCB exploration constant $C$ (box: setting used)")
plabel(ax, "(c) UCB setting: mean $\\Delta$ test acc. vs the setting used (pt)")
ax = axs[1, 1]
pa = rd("proxy_ablation/results/proxy_ablation_summary.csv")
pa = pa[pa["size"] > 0]
sz_c = {1: "#c9c8c3", 2: C["s1"], 3: C["s2"], 4: C["s3"]}
for d in DS:
    t = pa[pa.dataset == d]
    base_acc = t.best_acc_mean.mean()
    ax.scatter(t.kendall_tau, t.best_acc_mean - base_acc, c=[sz_c[s_] for s_ in t["size"]], marker=DMK[d], s=18, lw=0.3, edgecolors="white", zorder=3)
    e = t[t.arm == "EST"]
    ax.scatter(e.kendall_tau, e.best_acc_mean - base_acc, marker=DMK[d], s=48, facecolors="none", edgecolors=C["cost"], lw=1.1, zorder=4)
ax.axhline(0, color=C["rnd"], lw=0.7, ls=":"); ax.set_ylim(-1.75, 0.5)
ax.set_xlabel(r"Kendall $\tau$ of the proxy subset"); ax.set_ylabel("search acc. $-$ dataset mean (pt)")
h = [Line2D([], [], marker="o", ls="", color=sz_c[s_], ms=4) for s_ in [1, 2, 3, 4]] + [Line2D([], [], marker="o", ls="", mfc="none", mec=C["cost"], ms=5.5)] + \
    [Line2D([], [], marker=DMK[d], ls="", color=MUTED, ms=3.5) for d in DS]
ax.legend(h, ["1 proxy", "2 proxies", "3 proxies", "4 proxies", "$E+S+T$ (used)"] + [DSS[d] for d in DS], fontsize=6.0, loc="lower right", ncol=2, columnspacing=0.6, handletextpad=0.3)
ax.grid(True, axis="both"); plabel(ax, "(d) 15 proxy subsets as gates (budget 100, 12 seeds)")
save(fig, "fig5_design_choices")

# =============================================================== F6: stage-wise progression
tc = rd("final/results/train_cost_summary.csv")
stages = [("phase0", "random_search", "none", "random search", C["rnd"], "v", "--"),
          ("phase0", "re", "none", "RE", C["re"], "s", "--"),
          ("phase0", "local_search", "none", "LS", C["ls"], "D", "--"),
          ("phase1", "archtree_trie_only", "none", "S1 ArchTree-base", C["s1"], "o", "-"),
          ("phase2", "archtree_phase2", "zc", "S2 + supervised gate", C["s2"], "o", "-"),
          ("gated", "archtree", "none", "S3 ArchTree-RM", C["warm"], "o", "-"),
          ("phase3", "archtree_headline", "unsupervised", "S4 ArchTree-RMG", C["arch"], "o", "-")]
fig = plt.figure(figsize=(FULL, 3.25))
gs = fig.add_gridspec(2, 3, hspace=0.58, wspace=0.3)
for j, d in enumerate(DS):
    ax = fig.add_subplot(gs[0, j])
    for src, m, g, lab, c, mk, ls in stages:
        t = tc[(tc.source == src) & (tc.method == m) & (tc.gate == g) & (tc.dataset == d)].sort_values("budget")
        lw = 1.9 if "S4" in lab else 1.2
        ax.plot(t.train_hours, t.test, color=c, marker=mk, ls=ls, lw=lw, ms=3.6 if "S4" in lab else 3, label=lab,
                mec="white", mew=0.4, zorder=4 if "S4" in lab else 3)
    t = tc[(tc.source == "phase3") & (tc.dataset == d)].sort_values("budget")
    for b, xx, yy in zip(t.budget, t.train_hours, t.test):
        ax.annotate(f"b={b}", (xx, yy), xytext=(-3, 6), textcoords="offset points", fontsize=5.3, ha="right", color=C["arch"],
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=6)
    ax.set_xscale("log"); ax.set_xlabel("GPU-hours spent (log scale)"); logticks(ax)
    if j == 0: ax.set_ylabel("test accuracy (%)")
    plabel(ax, f"({'abc'[j]}) {DSL[d]}: accuracy vs cost")
    ax.grid(True, axis="both")
h, l = fig.axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=7, fontsize=6.3, bbox_to_anchor=(0.5, 1.0), columnspacing=0.9, handlelength=2.2)
g = rd("gated_baselines/results/gated_summary.csv"); ga = g[g.proposer == "archtree"]
for k, (col, ttl) in enumerate([("real_evals", "(d) real evaluations avoided by the gate"), ("train_hours", "(e) GPU-hours avoided by the gate")]):
    ax = fig.add_subplot(gs[1, k])
    for d in DS:
        on = ga[(ga.gate == "zc_unsup") & (ga.dataset == d)].set_index("budget")
        off = ga[(ga.gate == "none") & (ga.dataset == d)].set_index("budget")
        v = 100 * (1 - on[col] / off[col])
        ax.plot(v.index, v.values, marker=DMK[d], color=DCOL[d], label=DSL[d], mec="white", mew=0.4)
        if k == 0: ax.annotate(f"{v.values[-1]:.0f}%", (200, v.values[-1]), xytext=(4, (DS.index(d) - 1) * 6), textcoords="offset points",
                    fontsize=5.8, color=DCOL[d], va="center")
    if k == 1: ax.annotate("53% (all three)", (200, 53), xytext=(0, -11), textcoords="offset points", fontsize=5.8, ha="center", color=INK)
    logx_budget(ax); ax.set_xlim(20, 290); ax.set_ylim(20, 68); ax.set_xlabel("proposal budget"); ax.set_ylabel("saving (%)")
    plabel(ax, ttl)
    if k == 0: ax.legend(loc="lower right", fontsize=6.2)
ax = fig.add_subplot(gs[1, 2])
du = rd("final/results/duplicate_evals_summary.csv")
for m, lab, c, mk in [("flat_ucb_none[A7]", "flat UCB,\nno memory", C["rnd"], "v"), ("llm_nas", "LLM-NAS", C["warm"], "s"),
                      ("mcts_nas", "MCTS-style: 0.3%", C["y"], "D"), ("archtree_llm", "ArchTree-RMG: 0%", C["arch"], "o")]:
    t = du[du.method == m].groupby("budget").dup_rate_mean.mean() * 100
    ax.plot(t.index, t.values, marker=mk, color=c, mec="white", mew=0.4, zorder=4 if "RMG" in lab else 3)
    dy = {"MCTS-style: 0.3%": 8, "ArchTree-RMG: 0%": -7}.get(lab, 0)
    ax.annotate(lab, (t.index[-1], t.values[-1]), xytext=(6, dy), textcoords="offset points", fontsize=5.8, va="center", color=INK)
for bb_, vv_ in zip([25, 50, 100, 200], du[du.method == "llm_nas"].groupby("budget").dup_rate_mean.mean() * 100):
    ax.annotate(f"{vv_:.0f}%", (bb_, vv_), xytext=(-3, 6), textcoords="offset points", fontsize=5.3, ha="right", color=C["warm"])
logx_budget(ax); ax.set_xlim(20, 560); ax.set_ylim(-4, 85); ax.set_xlabel("proposal budget"); ax.set_ylabel("evaluations spent on repeats (%)")
ax.spines["bottom"].set_bounds(20, 230); plabel(ax, "(f) repeated evaluations (mean of 3 datasets)")
save(fig, "fig6_stages")

# =============================================================== F7: convergence (simple regret)
rc = rd("final/results/regret_curves.csv")
rc = rc[rc.budget == 200]
meths = [("random_exact", "random (analytic)", C["rnd"], ":"), ("random_search", "random search", C["rnd"], "-"),
         ("llm_nas", "LLM-NAS", C["warm"], "-"), ("re", "RE", C["re"], "-"), ("local_search", "LS", C["ls"], "-"),
         ("archtree_v1", "S1 ArchTree-base", C["s1"], "-"), ("archtree_zc", "S2 + supervised gate", C["s2"], "-"),
         ("archtree_llm", "S4 ArchTree-RMG", C["arch"], "-")]
fig, axs = plt.subplots(1, 4, figsize=(FULL, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.12]))
for ax, d in zip(axs[:3], DS):
    for m, lab, c, ls in meths:
        t = rc[(rc.method == m) & (rc.dataset == d)].sort_values("t")
        if t.empty: continue
        y = np.clip(t.mean_regret.values, 1e-2, None)
        ax.plot(t.t, y, color=c, ls=ls, lw=1.9 if m == "archtree_llm" else 1.0, label=lab, zorder=4 if m == "archtree_llm" else 3)
        if m == "archtree_llm":
            lo = np.clip(t.mean_regret - t["sem"], 1e-2, None); hi = t.mean_regret + t["sem"]
            ax.fill_between(t.t, lo, hi, color=c, alpha=0.15, lw=0)
            ax.scatter([t.t.iloc[-1]], [y[-1]], color=c, s=12, zorder=5)
    ax.axhline(0.5, color=MUTED, lw=0.6, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("real evaluations $t$"); ax.grid(True, axis="both", which="major")
    ax.set_ylim(1e-2, 30); plabel(ax, f"({'abc'[DS.index(d)]}) {DSL[d]}")
axs[0].set_ylabel("simple regret (validation pt)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=8, fontsize=6.0, bbox_to_anchor=(0.5, 1.07), columnspacing=0.8, handlelength=1.8)
ax = axs[3]
full = rd("final/results/regret_curves.csv"); full = full[full.budget == 200]
mm = [("random_search", "random"), ("mcts_nas", "MCTS-style"), ("enas", "ENAS-style"), ("llm_nas", "LLM-NAS"),
      ("archtree_v1", "S1 base"), ("archtree_zc", "S2 + sup. gate"), ("re", "RE"), ("local_search", "LS"), ("rz_nas", "RZ-style"),
      ("re_zc_warmup", "RE + ZC warm-up$^\\dagger$"), ("zico", "ZiCo-style$^\\dagger$"), ("archtree_llm", "S4 ArchTree-RMG")]
for i, (m, lab) in enumerate(mm):
    for d in DS:
        g_ = full[(full.method == m) & (full.dataset == d)].sort_values("t"); hit = g_[g_.mean_regret <= 0.5]
        v = hit.t.iloc[0] if len(hit) else 260
        ax.scatter([v], [i + (DS.index(d) - 1) * 0.22], marker=DMK[d] if len(hit) else ">", s=13 if len(hit) else 9, c=DCOL[d],
                   edgecolors="white" if len(hit) else DCOL[d], lw=0.3, zorder=3, alpha=1 if len(hit) else 0.6)
ax.axvline(225, color=GRID, lw=4); ax.text(255, -1.25, "not\nreached", fontsize=5.4, ha="center", color=MUTED)
ax.set_yticks(range(len(mm))); ax.set_yticklabels([l for _, l in mm], fontsize=6.0); ax.set_xscale("log")
for lab in ax.get_yticklabels():
    if "RMG" in lab.get_text(): lab.set_fontweight("bold")
ax.set_xlim(3, 300); ax.set_ylim(-1.6, len(mm) - 0.4); ax.set_xlabel("evaluations to regret $\\leq$ 0.5 pt")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.grid(axis="y", lw=0); ax.grid(axis="x", lw=0.4, color=GRID)
plabel(ax, "(d) sample efficiency")
ax.legend([Line2D([], [], marker=DMK[d], ls="", color=DCOL[d], ms=3.5) for d in DS], [DSS[d] for d in DS], fontsize=5.6, loc="lower left", handletextpad=0.1, borderaxespad=0.2)
fig.tight_layout(w_pad=0.6)
save(fig, "fig7_convergence")

# =============================================================== F8: equal GPU-hours
es = rd("equal_gpu_hours/results/egh_summary.csv")
arms = [("archtree_zc_unsup", "ArchTree-RMG", C["arch"], "-", "o"),
        ("archtree_none", "ArchTree-RM (no gate)", C["arch"], ":", "o"),
        ("re_zc_unsup", "RE + gate", C["re"], "-", "s"),
        ("re_zcwarm", "RE + ZC warm-up", C["warm"], "-", "D"),
        ("ls_zc_unsup", "LS + gate", C["ls"], "-", "^"),
        ("random_none", "random search", C["rnd"], "--", "v"),
        ("archtree_costpen", r"ArchTree-RM + cost penalty ($\lambda$ = 1)", C["cost"], "--", "x")]
cells = [("nb201", d) for d in DS] + [("nb101", "cifar10")] + [("natssss", d) for d in DS]
fig, axs = plt.subplots(2, 4, figsize=(FULL, 3.0))
for n_, (ax, (bn, d)) in enumerate(zip(axs.flat, cells)):
    s = es[(es.benchmark == bn) & (es.dataset == d)]
    for arm, lab, c, ls, m in arms:
        t = s[s.arm == arm].sort_values("k")
        if t.empty: continue
        lw = 1.8 if arm == "archtree_zc_unsup" else 1.1
        se_ = t.test_std / np.sqrt(t.n)
        ax.fill_between(t.k, t.test_mean - se_, t.test_mean + se_, color=c, alpha=0.13 if arm == "archtree_zc_unsup" else 0.07, lw=0, zorder=1)
        ax.plot(t.k, t.test_mean, color=c, ls=ls, marker=m, ms=3.4, lw=lw, label=lab, zorder=4 if arm == "archtree_zc_unsup" else 3,
                mec="white" if m != "x" else c, mew=0.5)
    plabel(ax, f"({'abcdefg'[n_]}) {BL[bn]}, {DSL[d]}"); logx_budget(ax, (25, 50, 100)); ax.set_xlim(21, 118)
for ax in axs[1, :3]: ax.set_xlabel(r"$k$ (budget $H = k\,\bar{\tau}$)")
for ax in axs[:, 0]: ax.set_ylabel("test accuracy (%)")
axs[1, 3].axis("off")
h, l = axs[0, 0].get_legend_handles_labels()
axs[1, 3].legend(h, l, loc="center", fontsize=6.6, handlelength=2.6, title="arms (12 seeds; band: $\\pm$1 s.e.)", title_fontsize=6.6)
fig.tight_layout(w_pad=0.5, h_pad=0.9)
save(fig, "fig8_equal_gpu_hours")

# =============================================================== F9: average ranks
ar = rd("final/results/stats_avg_ranks.csv"); fr = rd("final/results/stats_friedman.csv").set_index("source")
nice = {"archtree_v2": "ArchTree-RMG", "archtree_none": "ArchTree-RM (no gate)", "archtree_params": "ArchTree-RM + #params gate",
        "archtree_costpen": "ArchTree-RM + cost pen. (1)", "archtree_costpen025": "ArchTree-RM + cost pen. (0.25)",
        "re_zc_unsup": "RE + gate", "re_none": "RE", "re_zcwarm": "RE + ZC warm-up", "ls_zc_unsup": "LS + gate",
        "ls_none": "LS", "random_zc_unsup": "random + gate", "random_none": "random", "random_costpen": "random + cost pen. (1)",
        "random_costpen025": "random + cost pen. (0.25)"}
fam = lambda m: C["arch"] if m.startswith("archtree") else C["re"] if m.startswith("re") else C["ls"] if m.startswith("ls") else C["rnd"]
srcs = [("nb201_equal_gpu_h", "(a) NB201: $N$ = 108, $\\chi^2_F$ = 428, $p<10^{-82}$"), ("nb101_equal_gpu_h", "(b) NB101: $N$ = 36, $\\chi^2_F$ = 87, $p<10^{-13}$"), ("natssss_equal_gpu_h", "(c) NATS-SSS: $N$ = 108, $\\chi^2_F$ = 417, $p<10^{-80}$")]
fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.6))
for ax, (s, t) in zip(axs, srcs):
    r = ar[ar.source == s].sort_values("avg_rank")
    y = np.arange(len(r))[::-1]
    cd = fr.loc[s, "nemenyi_cd"]; best = r.avg_rank.min()
    ax.axvspan(best, best + cd, color=C["arch"], alpha=0.09, lw=0)
    ax.hlines(y, best, r.avg_rank, color=GRID, lw=1.0, zorder=1)
    ax.scatter(r.avg_rank, y, c=[fam(m) for m in r.method], s=[40 if m == "archtree_v2" else 17 for m in r.method], zorder=3,
               edgecolors=[INK if m == "archtree_v2" else "white" for m in r.method], lw=[0.8 if m == "archtree_v2" else 0.3 for m in r.method])
    for xx, yy, m in zip(r.avg_rank, y, r.method):
        ax.text(xx + 0.42, yy, f"{xx:.2f}", fontsize=5.6, va="center", ha="right", color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels([nice[m] for m in r.method], fontsize=6.4)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "ArchTree-RMG": lab.set_fontweight("bold")
    ax.set_xlim(r.avg_rank.max() + 2.8, best - 0.5); ax.set_ylim(-0.7, len(r) - 0.3 + 0.9)
    ax.set_xlabel("average rank (1 = best)"); ax.grid(axis="y", lw=0); ax.grid(axis="x", lw=0.4, color=GRID)
    ax.annotate("", xy=(best, len(r) - 0.1), xytext=(best + cd, len(r) - 0.1), arrowprops=dict(arrowstyle="<->", color=C["arch"], lw=0.9))
    plabel(ax, t, fs=6.9); ax.text(best + cd / 2, len(r) + 0.15, f"CD = {cd:.2f}", fontsize=6.4, ha="center", va="bottom", color=C["arch"])
fig.tight_layout(w_pad=0.3)
save(fig, "fig9_ranks")

# =============================================================== F10: gate-cost model
gv = rd("equal_gpu_hours/results/gate_cost_validation.csv")
gm = rd("equal_gpu_hours/results/gate_cost_model.csv").drop_duplicates(["benchmark", "dataset"])
fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.2), gridspec_kw=dict(width_ratios=[0.95, 1.0, 1.1]))
lab = [f"{BL[b]} {DSS[d]}" for b, d in zip(gm.benchmark, gm.dataset)]
yv = gm.slope_at_0.values; yy = np.arange(len(yv))[::-1]
cols = [C["arch"] if b != "natssss" else C["ls"] for b in gm.benchmark]
axs[0].barh(yy, yv, color=cols, height=0.62, lw=0)
for v_, y_ in zip(yv, yy):
    axs[0].text(v_ - 0.008, y_, f"{v_:.3f}", ha="right", va="center", fontsize=5.8, color=INK)
axs[0].set_yticks(yy); axs[0].set_yticklabels(lab, fontsize=6.4); axs[0].axvline(0, color=INK, lw=0.8)
axs[0].set_xlim(-0.5, 0.03); axs[0].set_xticks([-0.4, -0.2, 0])
axs[0].set_xlabel(r"slope $r_{PA}r_{PD}-r_{DA}$"); plabel(axs[0], "(a) criterion (Eq. 6)")
axs[0].grid(axis="y", lw=0); axs[0].grid(axis="x", lw=0.4, color=GRID)
lam = np.linspace(0, 1.5, 121)
for _, r in gm.iterrows():
    rho = (r.r_PA - lam * r.r_DA) / np.sqrt(1 + lam ** 2 - 2 * lam * r.r_PD)
    c = C["ls"] if r.benchmark == "natssss" else C["re"] if r.benchmark == "nb101" else C["arch"]
    axs[1].plot(lam, rho, color=c, lw=1.1, ls="-" if r.benchmark != "nb101" else "--")
axs[1].axvline(0.25, color=MUTED, lw=0.6, ls=":"); axs[1].axvline(1.0, color=MUTED, lw=0.6, ls=":")
axs[1].text(0.27, 0.92, "$\\lambda$ = 0.25", fontsize=5.6, color=MUTED); axs[1].text(1.02, 0.92, "$\\lambda$ = 1", fontsize=5.6, color=MUTED)
h = [Line2D([], [], color=C["arch"], lw=1.1), Line2D([], [], color=C["re"], lw=1.1, ls="--"), Line2D([], [], color=C["ls"], lw=1.1)]
axs[1].legend(h, ["NB201 (3 datasets)", "NB101", "NATS-SSS (3 datasets)"], fontsize=5.8, loc="lower left")
axs[1].set_xlabel(r"cost-penalty weight $\lambda$"); axs[1].set_ylabel(r"$\rho(S_\lambda, A)$ (Eq. 5)"); axs[1].set_ylim(-0.4, 1.0)
axs[1].grid(True, axis="both"); plabel(axs[1], "(b) admitted quality vs penalty")
for prop, c, m in [("archtree", C["arch"], "o"), ("random", C["rnd"], "s")]:
    for lam_, fill in [(1.0, True), (0.25, False)]:
        t = gv[(gv.proposer == prop) & (gv["lambda"] == lam_)]
        axs[2].scatter(t.predicted_delta, t.measured_delta, marker=m, s=14, facecolors=c if fill else "none", edgecolors=c, lw=0.8,
                       label=f"{'ArchTree-RMG' if prop == 'archtree' else 'random'}, $\\lambda$ = {lam_:g}", zorder=3)
lim = [-2.8, 0.6]
axs[2].fill_between([-1.0, 0], -2.8, 0, color="#eef4fc", lw=0, zorder=0); axs[2].fill_between([0, 0.3], 0, 0.6, color="#eef4fc", lw=0, zorder=0)
axs[2].plot(lim, lim, color=INK, lw=0.8, ls="--"); axs[2].axhline(0, color="#999", lw=0.6); axs[2].axvline(0, color="#999", lw=0.6)
axs[2].set_xlim(-1.0, 0.3); axs[2].set_ylim(lim); axs[2].grid(True, axis="both")
axs[2].set_xlabel("predicted change (pt)"); axs[2].set_ylabel("measured change (pt)")
plabel(axs[2], "(c) predicted vs measured"); axs[2].legend(loc="upper left", fontsize=5.6, handletextpad=0.2, framealpha=0.9, frameon=True, edgecolor="none")
fig.tight_layout(w_pad=0.7)
save(fig, "fig10_gate_cost")

# =============================================================== F11: ablations
a7 = rd("ablations/results/A7_summary.csv"); a7 = a7[a7.budget == 200]
dp = rd("design_ablation/results/design_paired_tests.csv")
ls_ = rd("final/results/landscape_smoothness.csv")
fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.4), gridspec_kw=dict(width_ratios=[1, 1.05, 0.8]))
armsA = [("trie", "trie (dedup)"), ("flat_set", "flat UCB + set dedup"), ("flat_none", "flat UCB, no dedup")]
x = np.arange(3); w = 0.26
for i, (arm, l) in enumerate(armsA):
    t = a7[a7.arm == arm].set_index("dataset").loc[DS]
    uniq = t.real_evals_mean - t.dup_real_mean
    axs[0].bar(x + (i - 1) * w, uniq, w * 0.9, color=[C["arch"], C["ls"], C["rnd"]][i], label=l, lw=0)
    if t.dup_real_mean.sum() > 0:
        axs[0].bar(x + (i - 1) * w, t.dup_real_mean, w * 0.9, bottom=uniq, color="white", edgecolor=C["cost"], hatch="//////", lw=0.5,
                   label="repeated (wasted)")
        for xx, tot, dup in zip(x + (i - 1) * w, t.real_evals_mean, t.dup_real_mean):
            axs[0].text(xx, tot + 2, f"{100*dup/tot:.0f}%", ha="center", fontsize=5.6, color=C["cost"])
axs[0].set_xticks(x); axs[0].set_xticklabels([DSS[d] for d in DS], fontsize=6.8); axs[0].set_ylabel("real evaluations (budget 200)")
plabel(axs[0], "(a) where the budget goes"); axs[0].set_ylim(0, 250); axs[0].legend(loc="upper left", fontsize=5.6, ncol=2, columnspacing=0.6, handlelength=1.2)
order = [("v1", "base (raw\nreward)"), ("rank", "rank reward\nonly"), ("mut_only", "mutation\nonly"), ("re", "RE,\nsame gate")]
for i, (arm, l) in enumerate(order):
    v = dp[dp.arm == arm].mean_diff_arm_minus_v2.values
    sig = dp[dp.arm == arm].outcome_vs_v2.values
    yy = np.full(len(v), len(order) - 1 - i) + np.linspace(-0.22, 0.22, len(v))
    axs[1].scatter(v, yy, s=11, c=[C["cost"] if s == "v2 better" else "#a6a5a1" for s in sig], zorder=3, lw=0)
    axs[1].plot([v.mean()] * 2, [len(order) - 1 - i - 0.32, len(order) - 1 - i + 0.32], color=INK, lw=1.6)
    axs[1].text(v.mean(), len(order) - 1 - i + 0.36, f"mean {v.mean():+.2f}", fontsize=5.8, va="bottom", ha="center", color=INK)
axs[1].axvline(0, color=C["arch"], lw=0.9, ls="--")
axs[1].set_yticks(range(len(order))[::-1]); axs[1].set_yticklabels([l for _, l in order], fontsize=6.6); axs[1].grid(axis="y", lw=0)
axs[1].grid(axis="x", lw=0.4, color=GRID); axs[1].set_xlim(-1.6, 0.68); axs[1].set_ylim(-0.5, 3.75)
axs[1].set_xlabel("test acc. minus ArchTree-RMG (pt)"); plabel(axs[1], "(b) design ablation, held-out seeds")
for d in DS:
    t = ls_[ls_.dataset == d].sort_values("depth")
    axs[2].plot(t.depth, t.eta2, marker=DMK[d], color=DCOL[d], label=DSS[d], mec="white", mew=0.4)
axs[2].axvspan(0.6, 3.4, color="#f6f3ee", lw=0, zorder=0)
axs[2].set_xticks(range(1, 7)); axs[2].set_xlabel("trie depth (tokens fixed)"); axs[2].set_ylabel(r"variance explained $\eta^2$")
axs[2].set_ylim(0, 1.05); axs[2].legend(fontsize=6.0, loc="upper left"); plabel(axs[2], "(c) information near the root")
fig.subplots_adjust(wspace=0.42)
save(fig, "fig11_ablations")

# =============================================================== F12: LLM prior
cv = rd("convergence/results/convergence_per_run_metrics.csv"); cv = cv[cv.budget == 100]
ct = rd("llm_contamination/results/contamination_summary.csv")
cp = rd("convergence/results/convergence_paired_tests.csv"); cp = cp[cp.budget == 100]
asw = rd("alpha_sweep/results/alpha_sweep_results.csv")
fig, axs = plt.subplots(2, 2, figsize=(FULL, 3.7), gridspec_kw=dict(hspace=0.8, wspace=0.5))
armsL = [("llm_off", "no prior", C["rnd"]), ("llm_on", "LLM prior", C["arch"]), ("llm_described", "LLM, described ops", C["warm"]), ("size_prior", "size prior (no LLM)", C["y"])]
x = np.arange(3); w = 0.2
ax = axs[0, 0]
for i, (arm, l, c) in enumerate(armsL):
    m = [cv[(cv.arm == arm) & (cv.dataset == d)]["evals_to_1.0pp"].mean() for d in DS]
    se = [cv[(cv.arm == arm) & (cv.dataset == d)]["evals_to_1.0pp"].std() / np.sqrt(12) for d in DS]
    ax.bar(x + (i - 1.5) * w, m, w * 0.9, yerr=se, color=c, label=l, capsize=1.4, error_kw=dict(lw=0.6), lw=0)
    for xx, mm_, ss in zip(x + (i - 1.5) * w, m, se):
        ax.text(xx, mm_ + ss + 0.8, f"{mm_:.1f}", ha="center", fontsize=5.2, color=MUTED, rotation=90, va="bottom")
ax.set_xticks(x); ax.set_xticklabels([DSL[d] for d in DS]); ax.set_ylabel("evaluations to final $-$ 1 pt")
sec = ax.secondary_yaxis("right", functions=(lambda e: e * 10 / 60, lambda h: h * 60 / 10))
sec.set_ylabel("hours at 10 min per evaluation", fontsize=6.2); sec.tick_params(labelsize=6)
plabel(ax, "(a) early convergence (budget 100)"); ax.set_ylim(0, 56); ax.legend(loc="upper left", fontsize=6.0, ncol=2)
ax = axs[0, 1]
pairs = [("llm_on", "llm_off", "LLM vs none", C["arch"]), ("llm_described", "llm_off", "described vs none", C["warm"]),
         ("size_prior", "llm_off", "size prior vs none", C["y"]), ("llm_on", "size_prior", "LLM vs size prior", C["ink"])]
mets = [("best_at_10", "best@10"), ("auc30", "AUC$_{30}$"), ("evals_to_1.0pp", "evals to $-$1 pt"), ("final_best", "final")]
for j, (met, ml) in enumerate(mets):
    for i, (a_, b_, pl, c) in enumerate(pairs):
        for d in DS:
            r = cp[(cp.metric == met) & (cp.arm_a == a_) & (cp.arm_b == b_) & (cp.dataset == d)]
            if r.empty or pd.isna(r.cohens_d.iloc[0]): continue
            dd = r.cohens_d.iloc[0] * (-1 if met == "evals_to_1.0pp" else 1)
            dd = float(np.clip(dd, -3.2, 3.2)); sig = r.p_holm.iloc[0] < 0.05
            yy_ = j + (i - 1.5) * 0.19 + (DS.index(d) - 1) * 0.05
            ax.scatter([dd], [yy_], marker=DMK[d], s=16 if sig else 9, facecolors=c if sig else "none", edgecolors=c, lw=0.7, zorder=3)
ax.axvline(0, color=INK, lw=0.7); ax.axvspan(-0.5, 0.5, color="#f4f4f2", lw=0, zorder=0)
ax.set_yticks(range(len(mets))); ax.set_yticklabels([m for _, m in mets], fontsize=6.6); ax.invert_yaxis(); ax.set_xlim(-3.4, 3.4)
ax.set_xlabel("Cohen's $d$ (positive = first arm better)", labelpad=1); ax.grid(axis="y", lw=0); ax.grid(axis="x", lw=0.4, color=GRID)
h = [Line2D([], [], marker="o", ls="", color=c, ms=4) for *_, c in pairs] + [Line2D([], [], marker="o", ls="", mfc="none", mec=MUTED, ms=3.5)] + [Line2D([], [], marker=DMK[d], ls="", color=MUTED, ms=3.5) for d in DS]
ax.legend(h, [p[2] for p in pairs] + ["open: n.s. (Holm)"] + [DSS[d] for d in DS], fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, handletextpad=0.2, columnspacing=0.5)
plabel(ax, "(b) effect sizes, 3 datasets (budget 100)")
ax = axs[1, 0]
for d in DS:
    for bud, ls in [(50, "--"), (100, "-")]:
        t = asw[(asw.dataset == d) & (asw.budget == bud)].groupby("alpha").best_test_acc.mean()
        ax.plot(t.index, t.values - t.loc[0.0], color=DCOL[d], ls=ls, marker=DMK[d], ms=3, mec="white", mew=0.3, lw=1.1)
ax.axhline(0, color=INK, lw=0.7); ax.axvspan(0.2, 1.0, color="#f4f4f2", lw=0, zorder=0)
ax.text(0.6, -0.55, "prior saturated:\nidentical runs", fontsize=5.8, ha="center", color=MUTED)
h = [Line2D([], [], color=DCOL[d], marker=DMK[d], ms=3, lw=1.1) for d in DS] + [Line2D([], [], color=MUTED, ls=s_) for s_ in ["--", "-"]]
ax.legend(h, [DSS[d] for d in DS] + ["budget 50", "budget 100"], fontsize=5.6, ncol=5, loc="upper center", columnspacing=0.6, handlelength=1.4)
ax.set_xlabel(r"prior weight $\alpha$"); ax.set_ylabel(r"test acc. minus $\alpha$ = 0 (pt)"); ax.set_ylim(-0.75, 1.3)
ax.grid(True, axis="both"); plabel(ax, "(c) $\\alpha$ sweep (12 seeds; 0 of 48 tests significant)")
ax = axs[1, 1]
modes = [("anonymous", "anonymous", C["rnd"]), ("standard", "standard (NB201 names)", C["arch"]), ("described", "described", C["warm"]), ("size_prior", "size prior", C["y"])]
for i, (md, l, c) in enumerate(modes):
    v = [ct[(ct["mode"] == md) & (ct.dataset == d)].spearman_mean.iloc[0] for d in DS]
    ax.bar(x + (i - 1.5) * w, v, w * 0.9, color=c, label=l, lw=0)
    for xx, vv in zip(x + (i - 1.5) * w, v):
        ax.text(xx, vv + 0.015, f"{vv:.2f}", ha="center", fontsize=5.2, color=MUTED, rotation=90, va="bottom")
ax.set_xticks(x); ax.set_xticklabels([DSL[d] for d in DS]); ax.set_ylim(0, 1.38)
ax.set_ylabel(r"Spearman $\rho$ with true op ranking"); plabel(ax, "(d) knowledge probe (56 prefixes per dataset)")
ax.legend(loc="upper left", ncol=2, fontsize=6.0)
save(fig, "fig12_llm")

# =============================================================== F13: hardware Pareto fronts
hw = np.load(os.path.join(DATA, "hw_table__cifar10.npz"))
fv = rd("final/results/cross_device_found_vs_truth.csv")
pr = rd("phase4/results/phase4_pareto_results.csv"); pr = pr[pr.budget == 200]
dev = ["edgegpu", "raspi4", "edgetpu", "pixel3", "eyeriss", "fpga"]
devn = {"edgegpu": "edge GPU", "raspi4": "Raspberry Pi 4", "edgetpu": "edge TPU", "pixel3": "Pixel 3", "eyeriss": "Eyeriss", "fpga": "FPGA"}

def nd_front(acc, lat):
    o = np.lexsort((-acc, lat)); best = -np.inf; keep = []
    for i in o:
        if acc[i] > best: keep.append(i); best = acc[i]
    return np.array(keep)

fig, axs = plt.subplots(2, 3, figsize=(FULL, 2.65))
for ax, d in zip(axs.flat, dev):
    acc, lat = hw["acc"], hw[d]; ok = ~np.isnan(lat) & (acc > 80)
    ax.scatter(lat[ok], acc[ok], s=0.6, c="#c9c8c3", lw=0, rasterized=True, label="all architectures")
    tf = np.array(json.load(open(os.path.join(DATA, f"true_pareto_front__{d}.json"))))
    tf = tf[np.argsort(tf[:, 1])]
    ax.step(tf[:, 1], tf[:, 0], where="post", color=INK, lw=1.0, label=f"true front")
    ax.scatter(tf[:, 1], tf[:, 0], s=6, c=INK, zorder=4, lw=0)
    found = []
    for s_, g_ in pr[pr.hardware == d].groupby("seed"):
        a_, l_ = g_.accuracy.values, g_.latency_ms.values
        k = nd_front(a_, l_); found.append(np.c_[a_[k], l_[k]])
    F = np.vstack(found)
    ax.scatter(F[:, 1], F[:, 0], s=9, facecolors="none", edgecolors=C["arch"], lw=0.6, zorder=3, alpha=0.7, label="found (12 seeds, b = 200)")
    n_true = len(tf); ax.set_ylim(84, 92.2)
    hvr = fv[(fv.hardware == d) & (fv.budget == 200)].hv_ratio.mean()
    ax.text(0.97, 0.18 if d == "edgetpu" else 0.05, f"$|F^*|$ = {n_true};  HV/HV$^*$ = {hvr:.3f}", transform=ax.transAxes, ha="right", fontsize=6.0, color=INK, bbox=dict(fc="white", ec="none", pad=1, alpha=0.9))
    plabel(ax, f"({'abcdef'[dev.index(d)]}) {devn[d]}"); ax.grid(True, axis="both")
    if d == "edgetpu": ax.text(0.97, 0.05, f"{np.isnan(lat).mean()*100:.0f}% of latencies missing", transform=ax.transAxes, ha="right", fontsize=6.0, color=MUTED, bbox=dict(fc="white", ec="none", pad=1))
for ax in axs[1]: ax.set_xlabel("latency (ms)")
for ax in axs[:, 0]: ax.set_ylabel("validation accuracy (%)")
h, l = axs[0, 0].get_legend_handles_labels()
h[0] = Line2D([], [], marker="o", ls="", color="#c9c8c3", ms=3)
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=6.6)
fig.tight_layout(h_pad=0.8, w_pad=0.6)
save(fig, "fig13_hw_fronts")

# =============================================================== F14: hardware summary
fv = rd("final/results/cross_device_found_vs_truth.csv")
tro = rd("final/results/cross_device_truth_overlap.csv")
trf = rd("final/results/cross_device_truth_transfer.csv")
fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.2), gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
a4 = rd("phase4/results/phase4_hv_igd_summary.csv"); a4["method"] = "archtree"
b4 = rd("phase4/results/phase4_baselines_hv_igd_summary.csv"); x4 = pd.concat([a4, b4])
base4 = x4[x4.method == "random_hw"].groupby(["hardware", "budget"]).hypervolume.mean()
for meth, lab, c, mk in [("archtree", "ArchTree-HW", C["arch"], "o"), ("nsga2_hw", "NSGA-II-style", C["warm"], "s")]:
    t = (x4[x4.method == meth].groupby(["hardware", "budget"]).hypervolume.mean() - base4).unstack(0)
    for d in dev:
        axs[0].plot(t.index, t[d].values, color=c, lw=0.5, alpha=0.35)
    axs[0].plot(t.index, t.mean(axis=1).values, color=c, marker=mk, lw=1.7, label=f"{lab} (mean of 6)", mec="white", mew=0.4)
axs[0].axhline(0, color=C["rnd"], lw=0.9, ls="--", label="random search"); axs[0].set_ylim(-130, 560)
logx_budget(axs[0]); axs[0].set_xlabel("proposal budget"); axs[0].set_ylabel("HV minus random search")
axs[0].legend(fontsize=5.8, loc="upper left"); plabel(axs[0], "(a) HV gain over random (thin: devices)")
M = np.eye(6)
for _, r in tro.iterrows():
    i, j = dev.index(r.hw_a), dev.index(r.hw_b); M[i, j] = M[j, i] = r.jaccard
short = ["eGPU", "RPi4", "eTPU", "Pix3", "Eyer.", "FPGA"]
heat(axs[1], M, short, short, BLUES, 0, 1, fs=5.8)
plabel(axs[1], "(b) Jaccard overlap of true fronts")
T = np.full((6, 6), np.nan)
for _, r in trf.iterrows():
    T[dev.index(r.from_hw), dev.index(r.to_hw)] = r.hv_ratio
heat(axs[2], T, short, short, BLUES, 0.8, 1.0, fs=5.8, thr=0.55)
axs[2].set_ylabel("front optimal for", fontsize=7); axs[2].set_xlabel("evaluated on", fontsize=7)
plabel(axs[2], "(c) HV retained when a front is reused")
fig.tight_layout(w_pad=0.8)
save(fig, "fig14_hw_summary")
# =============================================================== F15: transfer and portability of the gate
nb1 = rd("nb101/results/nb101_summary.csv")
es = rd("equal_gpu_hours/results/egh_summary.csv")
fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.3), gridspec_kw=dict(width_ratios=[1.0, 1.25, 1.25]))
ax = axs[0]
for prop, lab, c, mk in [("random", "random", C["rnd"], "v"), ("re", "RE", C["re"], "s"), ("archtree", "ArchTree", C["arch"], "o")]:
    for gate, ls, fill in [("none", ":", False), ("zc_unsup", "-", True)]:
        t = nb1[(nb1.proposer == prop) & (nb1.gate == gate)].sort_values("budget")
        ax.plot(t.train_hours, t.test_mean, color=c, ls=ls, marker=mk, ms=3.2, lw=1.6 if (prop == "archtree" and fill) else 1.0,
                mfc=c if fill else "white", mec=c, mew=0.6, label=f"{lab}{' + gate' if fill else ''}")
ax.set_xscale("log"); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("GPU-hours (log; budgets 50-400)"); ax.set_ylabel("test accuracy (%)"); ax.grid(True, axis="both")
ax.set_ylim(93.25, 94.35); ax.legend(fontsize=5.4, ncol=2, loc="upper left", columnspacing=0.5, handlelength=1.5); plabel(ax, "(a) NB101, CIFAR-10 (fixed budget)")
cols = [("nb201", d) for d in DS] + [("nb101", "cifar10")] + [("natssss", d) for d in DS]
clab = [f"{BL[b].replace('-SSS', '')}\n{DSS[d]}" for b, d in cols]
props = [("archtree", "ArchTree"), ("re", "RE"), ("ls", "LS"), ("random", "random")]
D = np.full((4, 7), np.nan); Rr = np.full((4, 7), np.nan)
for i, (p, _) in enumerate(props):
    for j, (b, d) in enumerate(cols):
        on = es[(es.benchmark == b) & (es.dataset == d) & (es.arm == f"{p}_zc_unsup")].set_index("k")
        off = es[(es.benchmark == b) & (es.dataset == d) & (es.arm == f"{p}_none")].set_index("k")
        if on.empty or off.empty: continue
        D[i, j] = (on.test_mean - off.test_mean).mean()
        Rr[i, j] = (on.real_evals / off.real_evals).loc[100]
heat(axs[1], D, [p[1] for p in props], clab, DIV, -1.2, 1.2, fmt="{:+.2f}", fs=5.6, norm=TwoSlopeNorm(0, -1.2, 1.2), thr=0.75)
axs[1].tick_params(axis="x", labelsize=5.8); plabel(axs[1], "(b) gate effect on test acc. at equal GPU-h (pt)")
heat(axs[2], Rr, [p[1] for p in props], clab, BLUES, 0.5, 1.1, fmt="{:.2f}", fs=5.6, thr=0.62)
axs[2].tick_params(axis="x", labelsize=5.8); plabel(axs[2], "(c) evaluations afforded, gated / ungated ($k$ = 100)")
fig.tight_layout(w_pad=0.6)
save(fig, "fig15_transfer")
# =============================================================== F16: probability of improvement and equivalence
pi = rd("final/results/stats_prob_improvement.csv"); to = rd("final/results/stats_tost.csv")
fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.3), gridspec_kw=dict(width_ratios=[0.95, 1.25, 0.8]))
ext = {"random_search": "random", "mcts_nas": "MCTS-style", "enas": "ENAS-style", "llm_nas": "LLM-NAS", "re": "RE",
       "zico": "ZiCo-style", "rz_nas": "RZ-style", "local_search": "LS", "re_zc_warmup": "RE + ZC warm-up"}
t = pi[pi.source == "nb201_full_budget"].sort_values("P_archtree_better")
ax = axs[0]; y = np.arange(len(t))
ax.hlines(y, t.ci95_lo, t.ci95_hi, color=C["arch"], lw=1.6, alpha=0.55)
ax.scatter(t.P_archtree_better, y, color=C["arch"], s=16, zorder=3, edgecolors="white", lw=0.4)
ax.axvline(0.5, color=INK, lw=0.7, ls="--"); ax.set_yticks(y); ax.set_yticklabels([ext[o] for o in t.other], fontsize=6.4)
ax.set_xlim(0.25, 1.0); ax.set_xlabel("P(ArchTree-RMG better)"); ax.grid(axis="y", lw=0); ax.grid(axis="x", lw=0.4, color=GRID)
plabel(ax, "(a) fixed budget, NB201 (144 pairs)")
arm = {"random_costpen": "random + pen. (1)", "archtree_costpen": "ArchTree-RM + pen. (1)", "random_none": "random",
       "random_costpen025": "random + pen. (0.25)", "random_zc_unsup": "random + gate", "archtree_none": "ArchTree-RM (no gate)",
       "re_none": "RE", "ls_none": "LS", "archtree_costpen025": "ArchTree-RM + pen. (0.25)", "archtree_params": "ArchTree-RM + #params",
       "re_zc_unsup": "RE + gate", "re_zcwarm": "RE + ZC warm-up", "ls_zc_unsup": "LS + gate"}
srcs = [("nb201_equal_gpu_h", "NB201", C["arch"], "o"), ("nb101_equal_gpu_h", "NB101", C["re"], "s"), ("natssss_equal_gpu_h", "NATS-SSS", C["ls"], "^")]
order = pi[pi.source == "nb201_equal_gpu_h"].sort_values("P_archtree_better").other.tolist()
ax = axs[1]
for k, (src, lab, c, mk) in enumerate(srcs):
    t = pi[pi.source == src].set_index("other")
    for i, o in enumerate(order):
        if o not in t.index: continue
        yy = i + (k - 1) * 0.24
        ax.hlines(yy, t.loc[o, "ci95_lo"], t.loc[o, "ci95_hi"], color=c, lw=1.0, alpha=0.6)
        ax.scatter([t.loc[o, "P_archtree_better"]], [yy], color=c, marker=mk, s=12, zorder=3, edgecolors="white", lw=0.3,
                   label=lab if i == len(order) - 1 or (src == "nb101_equal_gpu_h" and o == "re_zcwarm") else None)
ax.axvline(0.5, color=INK, lw=0.7, ls="--"); ax.set_yticks(range(len(order))); ax.set_yticklabels([arm[o] for o in order], fontsize=6.0)
ax.set_xlim(0.0, 1.0); ax.set_xlabel("P(ArchTree-RMG better)"); ax.grid(axis="y", lw=0); ax.grid(axis="x", lw=0.4, color=GRID)
h_ = [Line2D([], [], marker=mk, ls="", color=c, ms=3.8) for _, _, c, mk in srcs]
ax.legend(h_, [s_[1] for s_ in srcs], fontsize=5.8, loc="upper left", handletextpad=0.1)
plabel(ax, "(b) equal GPU-hours (95% bootstrap CI)")
ax = axs[2]
lbl = [("nb201_full_budget", "NB201\nfixed"), ("nb201_equal_gpu_h", "NB201\nequal GPU-h"), ("nb101_equal_gpu_h", "NB101\nequal GPU-h"),
       ("nb101_same_budget", "NB101\nfixed"), ("natssss_equal_gpu_h", "NATS\nequal GPU-h")]
yb = np.arange(len(lbl))[::-1]
for j, (dl, c, a_) in enumerate([(0.5, C["arch"], 0.9), (0.25, C["s1"], 0.9)]):
    v = [100 * to[(to.source == s_) & (to.delta == dl)].equivalent.mean() for s_, _ in lbl]
    ax.barh(yb + (0.18 if j == 0 else -0.18), v, height=0.34, color=c, label=f"$\\pm${dl:g} pt", lw=0)
    for yy, vv in zip(yb + (0.18 if j == 0 else -0.18), v):
        ax.text(vv + 1, yy, f"{vv:.0f}%", va="center", fontsize=5.6, color=MUTED)
ax.set_yticks(yb); ax.set_yticklabels([l for _, l in lbl], fontsize=6.0); ax.set_xlim(0, 100)
ax.set_xlabel("cells shown equivalent (TOST, %)"); ax.grid(axis="y", lw=0); ax.grid(axis="x", lw=0.4, color=GRID)
ax.legend(fontsize=5.8, loc="lower right"); plabel(ax, "(c) equivalence margins")
fig.tight_layout(w_pad=0.6)
save(fig, "fig16_stats")
# =============================================================== F17: method scorecard
tc = rd("final/results/train_cost_summary.csv"); bs = rd("phase0/results/baseline_summary.csv")
rcv = rd("final/results/regret_curves.csv"); rcv = rcv[(rcv.budget == 200) & (rcv.dataset == "cifar10")]
dup = rd("final/results/duplicate_evals_summary.csv"); dup = dup[dup.budget == 200]
rk = rd("final/results/stats_avg_ranks.csv"); rk = rk[rk.source == "nb201_full_budget"].set_index("method").avg_rank
pib = rd("final/results/stats_prob_improvement.csv"); pib = pib[pib.source == "nb201_full_budget"].set_index("other").P_archtree_better
st = rd("final/results/statistical_tests.csv"); st = st[st.comparator_type == "external"]
def holm(p):
    p = np.asarray(p); o = np.argsort(p); mm = len(p); adj = np.empty(mm); run = 0
    for r_, i in enumerate(o): run = max(run, (mm - r_) * p[i]); adj[i] = min(1, run)
    return adj
wtl = {}
for b_, g_ in st.groupby("method_b"):
    ph = holm(g_.p_value.values); w = int(((ph < .05) & (g_.mean_diff > 0)).sum()); l = int(((ph < .05) & (g_.mean_diff < 0)).sum())
    wtl[b_] = f"{w}/{len(g_) - w - l}/{l}"
rows = [("random_search", "Random search", "phase0", "random_search", "none"), ("mcts_nas", "MCTS-style", "phase0", "mcts_nas", "none"),
        ("enas", "ENAS-style", "phase0", "enas", "none"), ("llm_nas", "LLM-NAS", "phase0", "llm_nas", "none"),
        ("zico", "ZiCo-style", "phase0", "zico", "none"), ("rz_nas", "RZ-style", "phase0", "rz_nas", "none"),
        ("re", "RE", "phase0", "re", "none"), ("re_zc_warmup", "RE + ZC warm-up", "phase0", "re_zc_warmup", "none"),
        ("local_search", "Local search", "phase0", "local_search", "none"),
        ("archtree_v1", "S1 ArchTree-base", "phase1", "archtree_trie_only", "none"),
        ("archtree_none", "S3 ArchTree-RM", "gated", "archtree", "none"),
        ("archtree_llm", "S4 ArchTree-RMG", "phase3", "archtree_headline", "unsupervised")]
cols = [("C10", "ACCURACY", 1), ("C100", "ACCURACY", 1), ("IN16", "ACCURACY", 1),
        ("Real evals", "COST", -1), ("GPU-h (3 sets)", "COST", -1),
        ("Evals to\nregret$\\leq$0.5", "EFFICIENCY", -1), ("Repeated\nevals %", "EFFICIENCY", -1),
        ("Friedman\nrank", "STATISTICS vs RMG", -1), ("P(RMG\nbetter)", "STATISTICS vs RMG", -1), ("Holm\nW/T/L", "STATISTICS vs RMG", 0)]
V = np.full((len(rows), len(cols)), np.nan); TXT = [["–"] * len(cols) for _ in rows]
for i, (key, lab, src, m, g) in enumerate(rows):
    t = tc[(tc.source == src) & (tc.method == m) & (tc.gate == g) & (tc.budget == 200)].set_index("dataset")
    for j, d in enumerate(DS):
        if src == "phase0": v = bs[(bs.method == m) & (bs.budget == 200) & (bs.dataset == d)].best_test_acc.mean()
        else: v = t.loc[d, "test"]
        V[i, j] = v; TXT[i][j] = f"{v:.2f}"
    V[i, 3] = t.real_evals.mean(); TXT[i][3] = f"{V[i, 3]:.0f}"
    V[i, 4] = t.train_hours.sum(); TXT[i][4] = f"{V[i, 4]:.0f}"
    g_ = rcv[rcv.method == key].sort_values("t")
    if len(g_):
        hit = g_[g_.mean_regret <= 0.5]
        V[i, 5] = hit.t.iloc[0] if len(hit) else 999; TXT[i][5] = f"{hit.t.iloc[0]:.0f}" if len(hit) else ">200"
    d_ = dup[dup.method == key]
    if len(d_): V[i, 6] = 100 * d_.dup_rate_mean.mean(); TXT[i][6] = f"{V[i, 6]:.1f}"
    rkey = "archtree_v2" if key == "archtree_llm" else key
    if rkey in rk.index: V[i, 7] = rk[rkey]; TXT[i][7] = f"{rk[rkey]:.2f}"
    if key in pib.index: V[i, 8] = pib[key]; TXT[i][8] = f"{pib[key]:.2f}"
    if key in wtl: TXT[i][9] = wtl[key]
    if key == "archtree_llm": TXT[i][8] = "ref."; TXT[i][9] = "ref."
# colour: rank within column (darker = better)
S = np.full(V.shape, np.nan)
for j, (_, _, sgn) in enumerate(cols):
    if sgn == 0: continue
    col = V[:, j]; ok = ~np.isnan(col)
    if j == 8:   # P(RMG better): lower = the baseline is stronger -> shade by strength of the baseline
        sc = -col
    else: sc = sgn * col
    r_ = pd.Series(sc[ok]).rank(method="average").values
    S[ok, j] = (r_ - 1) / max(r_.max() - 1, 1)
cm = LinearSegmentedColormap.from_list("sc", ["#f2f6fc", "#9ec3ec", "#2a78d6", "#0b2f66"])
fig, ax = plt.subplots(figsize=(FULL, 2.9))
nR, nC = V.shape
for i in range(nR):
    for j in range(nC):
        if np.isnan(S[i, j]):
            fc = "#ededeb" if TXT[i][j] in ("–", "ref.") else "#ffffff"
        else: fc = cm(0.06 + 0.94 * S[i, j])
        ax.add_patch(plt.Rectangle((j, nR - 1 - i), 0.97, 0.9, fc=fc, ec="none"))
        dark = (not np.isnan(S[i, j])) and S[i, j] > 0.55
        best = (not np.isnan(S[i, j])) and S[i, j] == np.nanmax(S[:, j])
        ax.text(j + 0.485, nR - 1 - i + 0.45, TXT[i][j], ha="center", va="center", fontsize=6.3,
                color="white" if dark else INK, fontweight="bold" if best else "normal")
ax.set_xlim(-0.02, nC); ax.set_ylim(-0.15, nR + 1.55); ax.axis("off")
for i, (_, lab, *_r) in enumerate(rows):
    ax.text(-0.08, nR - 1 - i + 0.45, lab, ha="right", va="center", fontsize=6.6, fontweight="bold" if "RMG" in lab else "normal")
arrow = {1: r" $\uparrow$", -1: r" $\downarrow$", 0: ""}
for j, (cl, grp, sgn) in enumerate(cols):
    ax.text(j + 0.485, nR + 0.12, cl + (arrow[sgn] if j != 8 else ""), ha="center", va="bottom", fontsize=5.9, color=MUTED, linespacing=1.0)
groups = {}
for j, (_, grp, _s) in enumerate(cols): groups.setdefault(grp, []).append(j)
for grp, js in groups.items():
    ax.text((min(js) + max(js) + 1) / 2, nR + 1.25, grp, ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    ax.plot([min(js) + 0.05, max(js) + 0.92], [nR + 1.17] * 2, color=INK, lw=0.6)
for yline in [nR - 9]:
    ax.plot([-2.9, nC], [yline - 0.05] * 2, color=MUTED, lw=0.5, ls=":", clip_on=False)
ax.text(nC + 0.08, nR - 4.5, "baselines", rotation=90, va="center", fontsize=6, color=MUTED, style="italic")
ax.text(nC + 0.08, nR - 10.5, "ArchTree stages", rotation=90, va="center", fontsize=6, color=MUTED, style="italic")
save(fig, "fig17_scorecard")
print("figures written:", sorted(f for f in os.listdir(OUT) if f.endswith(".pdf")))
