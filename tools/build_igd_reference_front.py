# tools/build_igd_reference_front.py
"""
Run ONCE per hardware target: exhaustively queries every NB201 architecture's
(accuracy, latency) pair from HW-NAS-Bench, computes the TRUE non-dominated
front, saves it for IGD comparison in Phase 4 / Phase 6.

This is a lookup-table sweep (HW-NAS-Bench is precomputed data, not live
inference) — no GPU/torch work involved, pure dict/pickle reads. Fast:
15,625 archs x 6 hw targets = 93,750 lookups total.

Oct 2026: fronts built before the hwnb.py index fix paired each accuracy with ANOTHER
architecture's latency and are invalid. This script now OVERWRITES existing fronts by default
(use --skip-existing to keep them). Accuracy = the search metric of the v2 NB201 cache.

Usage (from archtree-llm/ root):
    python tools/build_igd_reference_front.py
"""
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from benchmarks.hwnb import HWNASBench

cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

print("[igd-front] Loading HW-NAS-Bench...")
bench = HWNASBench(cfg)
bench._load()
print(f"[igd-front] Loaded. {bench.total_archs} archs, "
      f"{len(cfg['benchmarks']['hw_targets'])} hw targets to sweep.\n")

out_dir = PROJECT_ROOT / "data"
out_dir.mkdir(parents=True, exist_ok=True)

SKIP_EXISTING = "--skip-existing" in sys.argv[1:]
print(f"[igd-front] alignment check (Spearman latency vs #conv, expect ~0.6): "
      f"{bench.check_alignment('cifar10', 'edgegpu'):.3f}")

for hardware in cfg["benchmarks"]["hw_targets"]:
    out_path = out_dir / f"true_pareto_front__{hardware}.json"
    if out_path.exists() and SKIP_EXISTING:
        print(f"[igd-front] SKIP {hardware} — {out_path} already exists")
        continue

    t0 = time.time()
    pts = []
    errors = 0
    for idx in range(bench.total_archs):
        try:
            acc, lat = bench.hw_query(idx, hardware=hardware, dataset="cifar10")
            pts.append((acc, lat))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    WARNING idx={idx}: {e}")
            continue

    # non-dominated front: sort by acc desc, keep strictly improving latency
    pts.sort(key=lambda p: -p[0])
    front, best_lat = [], float("inf")
    for acc, lat in pts:
        if lat < best_lat:
            front.append((acc, lat))
            best_lat = lat

    with open(out_path, "w") as f:
        json.dump(front, f)

    elapsed = time.time() - t0
    print(f"[igd-front] {hardware}: {len(pts)}/{bench.total_archs} queried "
          f"({errors} errors), {len(front)} front points, "
          f"{elapsed:.1f}s -> {out_path}")

print("\n[igd-front] DONE — all reference fronts written to data/true_pareto_front__*.json")