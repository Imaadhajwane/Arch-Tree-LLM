# tools/check_phase1_determinism.py
"""
Is search_phase1 deterministic, and does the CURRENT code reproduce an OLD run?

Runs ONE Phase-1 combination in two fresh Python processes with different PYTHONHASHSEED values
(0 and 1) and compares the evaluated-architecture sequences. Then compares against:
  * the log currently in outputs/phase1/logs/ (same combo), and
  * optionally an older log (--old-log).

Reading the verdict
  A vs B identical, both differ from old  -> code/config changed since the old run (not nondeterminism)
  A vs B differ                           -> nondeterminism (e.g. iteration order of a set / hash seed)
  everything identical                    -> fully reproducible

Writes only to outputs/p1_det_hs0/ and outputs/p1_det_hs1/ (safe to delete afterwards).
Reads only; touches no existing results.

Usage:
    python tools/check_phase1_determinism.py
    python tools/check_phase1_determinism.py --dataset cifar10 --seed 42 --budget 50 --old-log "outputs_backup_XXXX/_moved/phase1/logs/archtree_v1__nb201__cifar10__seed42__budget50.json"
"""
import argparse
import glob
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def child(a) -> None:
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import yaml
    from benchmarks.nb201 import NB201
    from archtree.search import search_phase1

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    bench = NB201(cfg)
    bench._load()
    bm = cfg.get("ucb", {}).get("backup_mode", "mean")
    r = search_phase1(bench, a.budget, a.dataset, a.seed, phase=f"p1_det_hs{a.hs}", backup_mode=bm)
    sig = inspect.signature(search_phase1).parameters
    print("CHILD_JSON " + json.dumps({"best_acc": r["best_acc"], "backup_mode": bm,
                                       "C_default": sig["C"].default if "C" in sig else None,
                                       "prune_default": sig["prune_percentile"].default
                                       if "prune_percentile" in sig else None}))


def load_archs(path):
    j = json.load(open(path))
    return [e["arch"] for e in j["eval_history"]], j.get("best_acc")


def first_div(x, y):
    return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None if len(x) == len(y) else min(len(x), len(y)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--old-log", default=None)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--hs", default="0")
    a = ap.parse_args()
    if a.child:
        child(a)
        return

    info = {}
    for hs in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        out = subprocess.run([sys.executable, __file__, "--child", "--hs", hs, "--dataset", a.dataset,
                              "--seed", str(a.seed), "--budget", str(a.budget)],
                             env=env, capture_output=True, text=True, cwd=ROOT)
        line = [l for l in out.stdout.splitlines() if l.startswith("CHILD_JSON ")]
        if out.returncode != 0 or not line:
            print(f"child hs={hs} failed:\n{out.stderr[-1500:]}")
            sys.exit(1)
        info[hs] = json.loads(line[0][len("CHILD_JSON "):])

    name = f"archtree_v1__nb201__{a.dataset}__seed{a.seed}__budget{a.budget}.json"
    A, accA = load_archs(ROOT / "outputs" / "p1_det_hs0" / "logs" / name)
    B, accB = load_archs(ROOT / "outputs" / "p1_det_hs1" / "logs" / name)
    print(f"settings used now: backup_mode={info['0']['backup_mode']}  C default={info['0']['C_default']}  "
          f"prune default={info['0']['prune_default']}")
    print(f"hash-seed 0 vs 1 : first divergence = {first_div(A, B)}   best {accA} vs {accB}")

    cur = ROOT / "outputs" / "phase1" / "logs" / name
    if cur.exists():
        C, accC = load_archs(cur)
        print(f"current phase1 log vs hs0 run: first divergence = {first_div(A, C)}   best {accA} vs {accC}")
    old = a.old_log or (sorted(glob.glob(str(ROOT / "outputs_backup_*" / "_moved" / "phase1" / "logs" / name))) or [None])[-1]
    if old and Path(old).exists():
        O, accO = load_archs(old)
        print(f"OLD log vs hs0 run: first divergence = {first_div(A, O)}   best {accO} vs {accA}")
        print(f"OLD log file date: {Path(old).stat().st_mtime}")
    print("\nverdict: " + ("NONDETERMINISTIC across processes (hash seed matters)" if first_div(A, B) is not None
                           else "deterministic across hash seeds"))


if __name__ == "__main__":
    main()