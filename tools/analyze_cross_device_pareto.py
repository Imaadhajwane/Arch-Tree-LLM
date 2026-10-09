# tools/analyze_cross_device_pareto.py
# isort: skip_file
"""
§3.2 — Verify Pareto-optimality across devices (NB201 + HW-NAS-Bench, no GPU, no search).

Part 1, ground truth (exact lookup over all 15,625 archs; HWMetricUnavailable archs excluded):
  truth_overlap       pairwise Jaccard of true fronts + Spearman of raw latency
  truth_multiplicity  number of archs Pareto-optimal on >= k devices
  truth_transfer      device A's true front re-measured with device B's latency, then compared
                      with B's true front (HV ratio, IGD; objectives min-max normalised by B's front)

Part 2, Phase 4 results (needs outputs/phase4/results/phase4_pareto_results.csv):
  The final front is rebuilt from every evaluated row of a run. Do NOT use the CSV's
  on_pareto_front column: it records membership at insertion time, not final membership.
  found_vs_truth      per (dataset, hardware, seed, budget): precision, recall, HV ratio vs true front
  found_overlap       per device pair: Jaccard of found fronts, transfer precision A->B

Ties: archs with identical (accuracy, latency) -> only the first is kept (strict improvement),
so front sizes are slightly conservative.

Outputs: outputs/final/results/cross_device_*.csv
Figure : outputs/final/plots/figures/cross_device_truth_jaccard.png
Cache  : data/hw_table__{dataset}.npz (safe to delete)

Usage:
    python tools/analyze_cross_device_pareto.py
    python tools/analyze_cross_device_pareto.py --datasets cifar10 --skip-found
"""
import argparse
import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402
import yaml           # noqa: E402
from loguru import logger  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import archtree.storage as storage  # noqa: E402
from archtree.encoder import OPS_NB201, index_from_tokens_nb201  # noqa: E402
from benchmarks.hwnb import HWMetricUnavailable, HWNASBench  # noqa: E402

with open(PROJECT_ROOT / "config.yaml") as _fh:
    cfg = yaml.safe_load(_fh)
N = 5 ** 6  # NB201 search space size (6 edges, 5 ops)

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>")


# ── data ────────────────────────────────────────────────────────────────────
def load_tables(hw: HWNASBench, hardware, dataset):
    """Return (acc[N], {device: latency[N]}). NaN latency = metric unavailable. Cached as .npz."""
    cache = PROJECT_ROOT / "data" / f"hw_table__{dataset}.npz"
    if cache.exists():
        z = np.load(cache)
        # "v2" marks tables built with the Oct-2026 index-mapped hwnb.py; older tables are invalid
        if "v2" in z.files and all(h in z.files for h in hardware) and "acc" in z.files:
            return z["acc"], {h: z[h] for h in hardware}

    acc = np.full(N, np.nan)
    lat = {h: np.full(N, np.nan) for h in hardware}
    for i in range(N):
        for h in hardware:
            try:
                a, lt = hw.hw_query(i, hardware=h, dataset=dataset)
                acc[i], lat[h][i] = a, lt
            except HWMetricUnavailable:
                # No latency on this device, but accuracy still comes from plain NB201
                if np.isnan(acc[i]):
                    acc[i] = hw.query(i, dataset)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, acc=acc, v2=np.array([1]), **lat)
    return acc, lat


# ── front / metric helpers ──────────────────────────────────────────────────
def pareto_idx(acc, lat, subset=None):
    """Indices on the max-acc / min-lat front (strict-improvement rule)."""
    ok = np.where(~np.isnan(lat) & ~np.isnan(acc))[0] if subset is None else np.asarray(subset)
    ok = ok[~np.isnan(lat[ok]) & ~np.isnan(acc[ok])]
    order = ok[np.lexsort((-acc[ok], lat[ok]))]  # latency asc, then accuracy desc
    keep, best = [], -np.inf
    for i in order:
        if acc[i] > best:  # sweep by latency; keep only points that raise the best accuracy
            keep.append(int(i))
            best = acc[i]
    return keep


def hv(points, lat_ref):
    """Hypervolume of a max-acc / min-lat front vs reference point (acc=0, lat=lat_ref)."""
    if not points:
        return 0.0
    pts = sorted(points, key=lambda p: p[1])  # latency asc
    area, prev_acc = 0.0, 0.0
    for a, lt in pts:
        if a > prev_acc:
            area += (a - prev_acc) * max(lat_ref - lt, 0.0)
            prev_acc = a
    return area


def igd_norm(found, truth):
    """Inverted generational distance; both objectives min-max normalised by the TRUE front's range."""
    if not found:
        return float("inf")
    t, f = np.array(truth, float), np.array(found, float)
    lo, hi = t.min(0), t.max(0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)  # avoid /0 on a degenerate axis
    t, f = (t - lo) / span, (f - lo) / span
    d = np.sqrt(((t[:, None, :] - f[None, :, :]) ** 2).sum(-1)).min(1)  # nearest found point per true point
    return float(d.mean())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan


# ── part 1: ground truth ────────────────────────────────────────────────────
def part1(hw, hardware, datasets, res):
    overlap, transfer, mult = [], [], []
    truth = {}  # dataset -> (acc, lat, true fronts); reused by part 2
    for ds in datasets:
        acc, lat = load_tables(hw, hardware, ds)
        fronts = {h: pareto_idx(acc, lat[h]) for h in hardware}
        truth[ds] = (acc, lat, fronts)
        for h in hardware:
            logger.info(f"  {ds:15s} {h:8s} true front: {len(fronts[h]):4d} archs")

        # How many archs are Pareto-optimal on >= k devices (k = n_devices -> "universal")
        counts = np.zeros(N, int)
        for h in hardware:
            counts[fronts[h]] += 1
        for k in range(1, len(hardware) + 1):
            mult.append({"dataset": ds, "on_at_least_k_devices": k, "n_archs": int((counts >= k).sum())})

        # Symmetric pairwise overlap
        for a, b in itertools.combinations(hardware, 2):
            both = ~np.isnan(lat[a]) & ~np.isnan(lat[b])
            rho = spearmanr(lat[a][both], lat[b][both])[0] if both.sum() > 3 else np.nan
            overlap.append({"dataset": ds, "hw_a": a, "hw_b": b,
                            "front_a": len(fronts[a]), "front_b": len(fronts[b]),
                            "jaccard": jaccard(fronts[a], fronts[b]),
                            "n_shared": len(set(fronts[a]) & set(fronts[b])),
                            "latency_spearman": rho})

        # Directed transfer: A's front re-measured on B, compared with B's true front
        for a, b in itertools.permutations(hardware, 2):
            tf = [(acc[i], lat[b][i]) for i in fronts[b]]
            lat_ref = float(np.nanmax(lat[b]) * 1.1)  # shared reference so HVs are comparable
            sub = [i for i in fronts[a] if not np.isnan(lat[b][i])]
            tr_idx = pareto_idx(acc, lat[b], subset=sub)  # non-dominated subset under B's latency
            tr_pts = [(acc[i], lat[b][i]) for i in tr_idx]
            hv_true = hv(tf, lat_ref)
            transfer.append({"dataset": ds, "from_hw": a, "to_hw": b,
                             "hv_ratio": hv(tr_pts, lat_ref) / hv_true if hv_true else np.nan,
                             "igd_norm": igd_norm(tr_pts, tf),
                             "n_transferred_front": len(tr_idx), "n_true_front_b": len(tf)})

    pd.DataFrame(overlap).to_csv(res / "cross_device_truth_overlap.csv", index=False)
    pd.DataFrame(transfer).to_csv(res / "cross_device_truth_transfer.csv", index=False)
    pd.DataFrame(mult).to_csv(res / "cross_device_truth_multiplicity.csv", index=False)

    ov, tr, mu = pd.DataFrame(overlap), pd.DataFrame(transfer), pd.DataFrame(mult)
    logger.info("=" * 78)
    logger.info("Part 1 (ground truth)")
    for ds in datasets:
        o, t, m = ov[ov.dataset == ds], tr[tr.dataset == ds], mu[mu.dataset == ds]
        logger.info(f"  {ds}: mean front Jaccard between devices = {o['jaccard'].mean():.2f} "
                    f"(min {o['jaccard'].min():.2f}, max {o['jaccard'].max():.2f}); "
                    f"mean latency Spearman = {o['latency_spearman'].mean():.2f}")
        logger.info(f"  {ds}: archs Pareto-optimal on >=k devices: " +
                    ", ".join(f"k={r.on_at_least_k_devices}:{r.n_archs}" for r in m.itertuples()))
        logger.info(f"  {ds}: transfer A->B  mean HV ratio = {t['hv_ratio'].mean():.2f} "
                    f"(worst {t['hv_ratio'].min():.2f}), mean normalised IGD = {t['igd_norm'].mean():.3f}")
    _plot(ov, hardware, res)
    return truth


# ── part 2: Phase 4 found fronts ────────────────────────────────────────────
def _tokens_to_idx(s: str):
    """'op|op|op|op|op|op' -> NB201 index, or None if the string is not a valid NB201 arch."""
    toks = str(s).split("|")
    return index_from_tokens_nb201(toks) if len(toks) == 6 and all(t in OPS_NB201 for t in toks) else None


def part2(truth, hardware, res):
    p = PROJECT_ROOT / "outputs" / "phase4" / "results" / "phase4_pareto_results.csv"
    if not p.exists():
        logger.warning(f"{p} not found - skipping Part 2.")
        return
    df = pd.read_csv(p)
    df["idx"] = df["arch_tokens"].map(_tokens_to_idx)
    df = df.dropna(subset=["idx"]).astype({"idx": int})

    fvt, fov = [], []
    for (ds, seed, budget), g in df.groupby(["dataset", "seed", "budget"]):
        if ds not in truth:
            continue
        acc, lat, tfronts = truth[ds]
        found = {}
        for h, gh in g.groupby("hardware"):
            if h not in tfronts:
                continue
            gh = gh.drop_duplicates("idx")
            # Scatter this run's evaluated points into full-size arrays so pareto_idx can be reused
            acc_f = np.full(N, np.nan)
            lat_f = np.full(N, np.nan)
            acc_f[gh["idx"].values], lat_f[gh["idx"].values] = gh["accuracy"].values, gh["latency_ms"].values
            found[h] = pareto_idx(acc_f, lat_f)

            tf = tfronts[h]
            lat_ref = float(np.nanmax(lat[h]) * 1.1)
            hv_true = hv([(acc[i], lat[h][i]) for i in tf], lat_ref)
            hv_found = hv([(acc_f[i], lat_f[i]) for i in found[h]], lat_ref)
            inter = len(set(found[h]) & set(tf))
            fvt.append({"dataset": ds, "hardware": h, "seed": seed, "budget": budget,
                        "found_front": len(found[h]), "true_front": len(tf),
                        "precision": inter / len(found[h]) if found[h] else np.nan,
                        "recall": inter / len(tf),
                        "hv_ratio": hv_found / hv_true if hv_true else np.nan})

        # Directed device pairs within the same run
        for x, y in itertools.permutations(sorted(found), 2):
            ty = set(tfronts[y])
            fov.append({"dataset": ds, "seed": seed, "budget": budget, "hw_a": x, "hw_b": y,
                        "jaccard_found": jaccard(found[x], found[y]),
                        "transfer_precision_a_to_b":
                            (len(set(found[x]) & ty) / len(found[x])) if found[x] else np.nan})

    if not fvt:
        logger.warning("Part 2: no Phase 4 rows matched the requested datasets/hardware - nothing written.")
        return
    fvt, fov = pd.DataFrame(fvt), pd.DataFrame(fov)
    fvt.to_csv(res / "cross_device_found_vs_truth.csv", index=False)
    fov.to_csv(res / "cross_device_found_overlap.csv", index=False)

    logger.info("=" * 78)
    logger.info("Part 2 (Phase 4 found fronts, rebuilt from all evaluated rows)")
    for (ds, b), g in fvt.groupby(["dataset", "budget"]):
        logger.info(f"  {ds:15s} b={int(b):<4d} precision={g['precision'].mean():.2f} "
                    f"recall={g['recall'].mean():.2f} HV ratio={g['hv_ratio'].mean():.2f}")
    if len(fov):
        logger.info(f"  found fronts: mean Jaccard between devices = {fov['jaccard_found'].mean():.2f}; "
                    f"mean transfer precision A->B = {fov['transfer_precision_a_to_b'].mean():.2f}")
        # Compare transfer precision with the own-device precision printed above
        logger.info("  (transfer precision << own-device precision => a front found for one device "
                    "is NOT optimal on another)")


# ── plot ────────────────────────────────────────────────────────────────────
def _plot(ov, hardware, res):
    """Heatmap of mean true-front Jaccard between devices."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from plots.figure_export import install as _gallery; _gallery()   # copy + data CSV to plots/figures
    except ImportError:
        return
    M = np.eye(len(hardware))
    for (a, b), v in ov.groupby(["hw_a", "hw_b"])["jaccard"].mean().items():
        i, j = hardware.index(a), hardware.index(b)
        M[i, j] = M[j, i] = v  # symmetric
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(hardware)), hardware, rotation=45, ha="right")
    ax.set_yticks(range(len(hardware)), hardware)
    for i in range(len(hardware)):
        for j in range(len(hardware)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 else "black")
    fig.colorbar(im, fraction=0.046)
    ax.set_title("true-front Jaccard between devices")
    fig.tight_layout()
    fig.savefig(storage.get_figure_path("final", "cross_device_truth_jaccard"), dpi=180)
    plt.close(fig)


def main(args):
    hardware = args.hardware
    res = storage.get_output_dir("final", "results")
    hw = HWNASBench(cfg)
    hw._load()
    logger.info(f"hardware={hardware} datasets={args.datasets}")
    truth = part1(hw, hardware, args.datasets, res)
    if not args.skip_found:
        part2(truth, hardware, res)
    logger.success("cross-device analysis done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", nargs="+", default=cfg["benchmarks"]["hw_targets"])
    ap.add_argument("--datasets", nargs="+", default=["cifar10"])
    ap.add_argument("--skip-found", action="store_true", help="run Part 1 only")
    main(ap.parse_args())