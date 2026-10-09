# tools/preflight.py
"""
Checks everything the requested pipeline steps need BEFORE hours of compute start.
Called by run_full_pipeline.ps1 (step 0 and before every plan); also usable by hand:

    python tools/preflight.py                    # check everything, report only
    python tools/preflight.py --steps 4 6 7 8    # exit 1 if something those steps need is missing

Never prints API keys (only how many are configured).
"""
import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

ALL_STEPS = range(0, 26)              # steps 0-25 of run_full_pipeline.ps1 (Oct 4, 2026)
GPU_STEPS = {4, 7, 12, 13, 18, 21, 22, 25}
GROQ_STEPS = {5, 14, 15, 16, 24}      # 8 only if ARCHTREE_SKIP_A4 != 1
HW_FRONTS = ["edgegpu", "raspi4", "edgetpu", "pixel3", "eyeriss", "fpga"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*", type=int, default=None)
    ap.add_argument("--skip-a4", action="store_true")
    a = ap.parse_args()
    steps = set(a.steps) if a.steps is not None else set(ALL_STEPS)
    skip_a4 = a.skip_a4 or os.environ.get("ARCHTREE_SKIP_A4") == "1"
    groq_needed = bool(steps & GROQ_STEPS) or (8 in steps and not skip_a4)

    rows = []          # (item, ok, detail, blocking)

    def add(item, ok, detail, needed_by, produced_by=()):
        # blocking only if a requested step needs it and no requested step builds it first
        blocking = bool(steps & set(needed_by)) and not (steps & set(produced_by))
        rows.append((item, ok, detail, blocking))

    # python packages
    for mod in ["numpy", "pandas", "scipy", "yaml", "loguru", "dotenv", "torch"]:
        try:
            importlib.import_module(mod)
            add(f"python package {mod}", True, "", ALL_STEPS)
        except ImportError:
            add(f"python package {mod}", False, "pip install -r requirements.txt", ALL_STEPS)
    for mod, needed, why in [("matplotlib", [3, 8, 10, 11, 12, 13, 15, 17, 18], "figures"),
                             ("groq", sorted(GROQ_STEPS | {8}), "Groq LLM steps"),
                             ("nats_bench", [23], "builds the NATS-SSS cache (pip install nats_bench)")]:
        try:
            importlib.import_module(mod)
            add(f"python package {mod}", True, "", needed)
        except ImportError:
            add(f"python package {mod}", False, why, needed)
    try:
        importlib.import_module("torchvision")
        add("python package torchvision", True, "", GPU_STEPS)
    except ImportError:
        add("python package torchvision", False, "needed for the CIFAR-10 proxy batch", GPU_STEPS)

    # GPU
    try:
        import torch
        cuda = torch.cuda.is_available()
        add("CUDA GPU", cuda, torch.cuda.get_device_name(0) if cuda else "torch.cuda.is_available() is False",
            GPU_STEPS)
    except Exception as e:      # noqa: BLE001
        add("CUDA GPU", False, str(e), GPU_STEPS)

    # data
    import yaml
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    p = cfg["paths"]
    cache = ROOT / p.get("nb201_cache", "data/nb201_cache_v2.json")
    add("NB201 v2 cache", cache.exists(), str(cache.relative_to(ROOT)),
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], produced_by=[2])
    if not cache.exists():
        add("NB201 .pth (to build the cache)", (ROOT / p["nb201"]).exists(), p["nb201"], [2])
    fronts = [f for f in HW_FRONTS if not (ROOT / "data" / f"true_pareto_front__{f}.json").exists()]
    add("HW true Pareto fronts (6)", not fronts, "missing: " + ", ".join(fronts) if fronts else "", [9, 18],
        produced_by=[3])
    hw_repo = ROOT / p.get("hwnb_repo", "third_party/HW-NAS-Bench")
    add("HW-NAS-Bench repo + pickle", (hw_repo / "HW-NAS-Bench-v1_0.pickle").exists(), str(hw_repo), [3, 9, 18])
    add("CIFAR-10 images (data/cifar10)", (ROOT / "data" / "cifar10").exists(),
        "downloaded automatically by torchvision if missing", [])
    npz = ROOT / p.get("nb101_cache", "data/nb101_cache.npz")
    tfr = ROOT / p.get("nb101", "data/nasbench_only108.tfrecord")
    add("NB101 cache (npz)", npz.exists(), "built by step 20" if not npz.exists() else "", [21, 22, 25],
        produced_by=[20])
    if not npz.exists():
        add("NB101 tfrecord (to build the cache)", tfr.exists(), p.get("nb101"), [20])

    ss = ROOT / p.get("natssss_cache", "data/natssss_cache.json")
    add("NATS-SSS cache", ss.exists(), str(ss.relative_to(ROOT)) if ss.exists() else "built by step 23",
        [22, 25], produced_by=[23])
    if not ss.exists():
        src = ROOT / p.get("natssss", "data/NATS-sss-v1_0-50262.pickle.pbz2")
        alt = src.parent / "NATS-sss-v1_0-50262-simple"
        add("NATS-SSS source (to build the cache)", src.exists() or alt.is_dir(),
            "extract NATS-sss-v1_0-50262-simple.tar into data\\ (data/README.md)", [23])
    zc = ROOT / "outputs" / "_zc_raw_cache.json"
    add("zero-cost proxy cache (step 25)", zc.exists(),
        f"{zc.stat().st_size / 1e6:.0f} MB" if zc.exists() else "optional - step 22 computes missing scores itself", [])

    # upstream outputs some steps read
    cal = ROOT / "outputs" / "phase2" / "weights"
    has_cal = cal.exists() and any(cal.glob("zc_calibration_weights__nb201__*.pt"))
    add("Phase 2 calibration (from step 7)", has_cal, "run step 7 first", [8, 10, 11, 12, 13, 14, 15, 16],
        produced_by=[7])

    # LLM backend (keys never printed)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:           # noqa: BLE001
        pass
    pool = [k for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
    n_keys = len(pool) or (1 if os.environ.get("GROQ_API_KEY", "").strip() else 0)
    if os.environ.get("GROQ_API_KEYS_FILE", "").strip():
        n_keys = max(n_keys, 1)
    try:
        importlib.import_module("ollama")
        has_ollama = True
    except ImportError:
        has_ollama = False
    rows.append(("Groq keys configured", n_keys > 0, f"{n_keys} key(s)", groq_needed))
    # LLM-free steps still construct LLMPrior: needs a key OR the ollama package
    add("LLMPrior constructible (key or ollama pkg)", n_keys > 0 or has_ollama, "", [6, 8, 9, 10, 11, 12, 21])

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    add("free disk >= 5 GB", free_gb >= 5, f"{free_gb:.0f} GB free", ALL_STEPS)

    bad = [r for r in rows if not r[1] and r[3]]
    print(f"\nPREFLIGHT for steps {sorted(steps)}" + ("  (A4 skipped)" if skip_a4 else ""))
    for item, ok, detail, blocking in rows:
        tag = "ok  " if ok else ("MISS" if blocking else "warn")
        print(f"  [{tag}] {item:45s} {detail}")
    if bad:
        print(f"\nPREFLIGHT FAILED: {len(bad)} blocking item(s) above (MISS). Fix them, then rerun.")
        sys.exit(1)
    print("\nPREFLIGHT OK")


if __name__ == "__main__":
    main()
