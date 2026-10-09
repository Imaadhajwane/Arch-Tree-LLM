# tools/build_repro_appendix.py
# isort: skip_file
"""
§3.4 — reproducibility appendix, pinned versions, release bundle.

Builds, from the LIVE project (not from memory), into outputs/final/repro/:
  REPRODUCIBILITY.md        environment, seeds, budgets, effective hyper-parameters, run order, checksums
  repro_appendix.tex        the same two tables as booktabs LaTeX
  environment.json          machine-readable
  requirements.lock.txt     `pip freeze` of the interpreter running this script
  data_checksums.json       sha256 of benchmark files (cached by size+mtime; --skip-hash to skip)
With --bundle also writes release/archtree_repro_bundle.zip (config, seeds, lock file, appendix,
cache builders, checksums). Benchmark files themselves are NOT bundled (licences) — the
README inside the zip says where to download them.

CONFIG vs CODE audit (important): several config.yaml keys are never read by any script, and
some code constants silently differ from the config comment. The appendix must state what the
code EFFECTIVELY uses, so this script
  * scans every .py for each config leaf key and lists keys that nothing reads, and
  * extracts the effective constants straight from the source files by regex
    (search.py, ucb.py, zc_proxy.py, llm_prior.py).
Known example at the time of writing: config ucb.C = 2.0 is never passed to any search function,
so every phase uses the code default C = 1.414; phase 1/2 read only ucb.backup_mode.

Usage (from archtree-llm/ root):
    python tools/build_repro_appendix.py
    python tools/build_repro_appendix.py --skip-hash --bundle
"""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
OUT = PROJECT_ROOT / "outputs" / "final" / "repro"

PACKAGES = [("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy"), ("scipy", "scipy"),
            ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("pyyaml", "yaml"), ("loguru", "loguru"),
            ("python-dotenv", "dotenv"), ("groq", "groq"), ("ollama", "ollama"),
            ("nas-bench-201", "nas_201_api"), ("psutil", "psutil")]

RUN_ORDER = [
    "python tools/build_nb201_cache.py",
    "python tools/build_igd_reference_front.py",
    "python experiments/run_phase0.py            # baselines (+ run_phase0_llm_nas_only.py)",
    "python experiments/run_phase1.py",
    "python experiments/run_phase2.py            # ZC calibration per dataset",
    "python experiments/run_phase3.py",
    "python experiments/run_phase4.py",
    "python experiments/run_ablations.py",
    "python experiments/run_trie_ablation.py     # A7",
    "python experiments/run_proxy_ablation.py    # §3.3",
    "python experiments/run_llm_size_sensitivity.py",
    "python experiments/run_convergence_study.py",
    "python experiments/run_phase6_aggregate.py",
    "python tools/analyze_duplicate_evals.py && python tools/analyze_tcr_h.py && "
    "python tools/analyze_regret.py && python tools/analyze_cross_device_pareto.py",
]


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def pkg_version(dist, mod):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        try:
            return getattr(__import__(mod), "__version__", "not installed")
        except Exception:
            return "not installed"


def git_info(path: Path):
    if not (path / ".git").exists() and not sh(["git", "rev-parse", "--git-dir"], cwd=path):
        return {"commit": None, "dirty": None}
    return {"commit": sh(["git", "rev-parse", "HEAD"], cwd=path) or None,
            "dirty": bool(sh(["git", "status", "--porcelain"], cwd=path))}


def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return {"name": p.name, "capability": f"{p.major}.{p.minor}", "memory_gb": round(p.total_memory / 1e9, 1),
                    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "cudnn_benchmark_note": "zc_proxy sets cudnn.benchmark=True (non-deterministic kernel choice)"}
        return {"name": None}
    except Exception:
        return {"name": None}


def sha256(path: Path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def checksums(skip: bool):
    cache_p = OUT / "data_checksums.json"
    old = json.load(open(cache_p)) if cache_p.exists() else {}
    res = {}
    for key in ("nb201", "nb101", "nb201_cache", "nb101_cache", "hwnb"):
        rel = cfg["paths"].get(key)
        if not rel:
            continue
        p = PROJECT_ROOT / rel
        if not p.exists():
            res[key] = {"path": rel, "sha256": None, "note": "missing"}
            continue
        st = p.stat()
        prev = old.get(key, {})
        if prev.get("size") == st.st_size and prev.get("mtime") == st.st_mtime and prev.get("sha256"):
            res[key] = prev
        elif skip:
            res[key] = {"path": rel, "size": st.st_size, "mtime": st.st_mtime, "sha256": None, "note": "skipped"}
        else:
            print(f"[repro] hashing {rel} ({st.st_size / 1e9:.2f} GB) ...")
            res[key] = {"path": rel, "size": st.st_size, "mtime": st.st_mtime, "sha256": sha256(p)}
    return res


def leaves(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from leaves(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", k, v


def unused_config_keys():
    srcs = {p: p.read_text(errors="ignore") for p in PROJECT_ROOT.rglob("*.py")
            if "outputs" not in p.parts and p.name != "build_repro_appendix.py"}
    out = []
    for full, leaf, val in leaves(cfg):
        if not any(re.search(rf"[\"']{re.escape(leaf)}[\"']", t) for t in srcs.values()):
            out.append((full, val))
    return out


def grab(path, pattern, cast=str):
    p = PROJECT_ROOT / path
    if not p.exists():
        return None
    m = re.search(pattern, p.read_text(errors="ignore"), re.M | re.S)
    return cast(m.group(1)) if m else None


def effective_constants():
    """Since Oct 2026 the search constants come from config.yaml through archtree/settings.py, so
    they are read from the live module (what the code actually uses), plus a few literals."""
    from archtree import settings as S
    return {
        "UCB exploration C": S.C,
        "backup_mode phases 1/2": S.BACKUP_MODE_P12,
        "backup_mode phase 3/4 + studies": S.BACKUP_MODE_P34,
        "PRUNE_EVERY_N": S.PRUNE_EVERY_N,
        "PRUNE_PERCENTILE": S.PRUNE_PERCENTILE,
        "PRUNE_RULE": S.PRUNE_RULE,
        "PRUNE_MIN_VISITS": S.PRUNE_MIN_VISITS,
        "WARMUP_REAL_EVALS": S.WARMUP_REAL_EVALS,
        "PROXY_THRESHOLD_PERCENTILE": S.PROXY_THRESHOLD_PERCENTILE,
        "ZC calibration mode": S.CALIBRATION_MODE,
        "ZC calibration archs (labels if supervised)": S.CALIBRATION_ARCHS,
        "LLM alpha_init": S.ALPHA_INIT,
        "LLM max calls / run": S.MAX_LLM_CALLS,
        "LATENCY_PENALTY": S.LATENCY_PENALTY,
        "LATENCY_NORM_MS": S.LATENCY_NORM_MS,
        "NB201 search split / report metric": f"{S.NB201_SEARCH_SPLIT} / {S.REPORT_METRIC}",
        "ucb.alpha_at reference_budget": grab("archtree/ucb.py", r"reference_budget: int = (\d+)", int),
        "LLMPrior alpha_floor": grab("archtree/llm_prior.py", r"alpha_floor: float = ([\d.]+)", float),
        "LLM temperature": grab("archtree/llm_prior.py", r"temperature=([\d.]+)", float),
        "LLM groq max_tokens": grab("archtree/llm_prior.py", r"max_tokens=(\d+)", int),
        "ZC proxy fixed init seed": grab("archtree/zc_proxy.py", r"_init_seed = (\d+)", int),
        "ZC NASWOT batches": grab("archtree/zc_proxy.py", r"^N_NASWOT_BATCHES = (\d+)", int),
        "ZC NASWOT batch size": grab("archtree/zc_proxy.py", r"^NASWOT_BATCH_SIZE = (\d+)", int),
        "ZC proxy channels": grab("archtree/zc_proxy.py", r"^PROXY_CHANNELS = (\d+)", int),
    }


def build(args):
    OUT.mkdir(parents=True, exist_ok=True)
    env = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "gpu": gpu_info(),
        "packages": {d: pkg_version(d, m) for d, m in PACKAGES},
        "git": {"archtree": git_info(PROJECT_ROOT),
                "HW-NAS-Bench": git_info(PROJECT_ROOT / cfg["paths"].get("hwnb_repo", "third_party/HW-NAS-Bench"))},
    }
    freeze = sh([sys.executable, "-m", "pip", "freeze"])
    (OUT / "requirements.lock.txt").write_text(freeze + "\n")
    chk = checksums(args.skip_hash)
    (OUT / "data_checksums.json").write_text(json.dumps(chk, indent=2))
    eff = effective_constants()
    unused = unused_config_keys()
    env.update({"config": cfg, "effective_constants": eff,
                "unused_config_keys": [k for k, _ in unused], "data_checksums": chk})
    (OUT / "environment.json").write_text(json.dumps(env, indent=2, default=str))

    S = cfg["search"]
    md = ["# Reproducibility appendix", f"_generated {env['generated_utc']} from the live project_", "",
          "## Environment", "", "| item | value |", "|---|---|",
          f"| Python | {env['python']} |", f"| OS | {env['platform']} |",
          f"| GPU | {env['gpu'].get('name')} (CC {env['gpu'].get('capability')}, CUDA {env['gpu'].get('cuda')}, "
          f"cuDNN {env['gpu'].get('cudnn')}) |" if env["gpu"].get("name") else "| GPU | none detected on this machine |"]
    md += [f"| {d} | {v} |" for d, v in env["packages"].items()]
    for name, g in env["git"].items():
        md.append(f"| {name} commit | {g['commit']}{' (DIRTY)' if g['dirty'] else ''} |")
    md += ["", "Full pinned list: `requirements.lock.txt`.", "",
           "## Protocol", "",
           f"* Benchmark: NAS-Bench-201 (5^6 = 15,625 architectures); datasets {cfg['benchmarks']['datasets']}; "
           f"hardware targets (HW-NAS-Bench) {cfg['benchmarks']['hw_targets']}",
           f"* Budgets (proposals per run): {S['budgets']}; seeds ({len(S['seeds'])}): {S['seeds']}",
           "* Statistics: paired Wilcoxon signed-rank across seeds at fixed budget, paired Cohen's d (run_phase6_aggregate.py)",
           "* Proxy gate: top `zc_proxy.proxy_threshold_percentile` = "
           f"{cfg['zc_proxy']['proxy_threshold_percentile']}% of scores seen so far, after {eff['WARMUP_REAL_EVALS']} warm-up real evals",
           f"* LLM: config backend model `{cfg['llm']['groq']['model']}` (Groq) / `{cfg['llm']['model']}` (Ollama); "
           f"alpha_init = {cfg['llm']['alpha_init']} (locked), max_calls/run = {cfg['llm']['max_calls']}", "",
           "## Effective hyper-parameters (read from the source, not from config.yaml)", "",
           "| parameter | value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in eff.items()]
    md += ["", "## config.yaml keys that NO script reads (do not report these as settings)", ""]
    md += [f"* `{k}` = {v}" for k, v in unused] or ["* none"]
    md += ["", "Since Oct 2026 these values come from config.yaml via archtree/settings.py (one source of truth).", "",
           "## Determinism notes", "",
           f"* Tabular benchmarks: accuracy/latency lookups are deterministic; run-to-run variation comes from the search seed "
           f"(`random.seed`, `torch.manual_seed`).",
           f"* ZC proxy: fixed init seed {eff['ZC proxy fixed init seed']}, cached CIFAR-10 mini-batches drawn once per process "
           "(unseeded DataLoader shuffle) — raw scores are cached to `outputs/_zc_raw_cache.json`; ship that file (or its hash) "
           "to make proxy scores identical across machines. cuDNN benchmark mode is on (kernel choice can vary).",
           "* LLM prior: hosted models drift and are rate-limited; answers are cached per trie prefix. Report the model string, "
           f"temperature ({eff['LLM temperature']}), and ship the cache files (`outputs/*/logs/llm_cache__*.json`) for exact replay.",
           "", "## Run order", "", "```"] + RUN_ORDER + ["```", "", "## Data checksums (sha256)", ""]
    md += [f"* `{v['path']}`: `{v.get('sha256')}`" + (f" ({v['note']})" if v.get("note") else "") for v in chk.values()]
    (OUT / "REPRODUCIBILITY.md").write_text("\n".join(md) + "\n")

    def tex(s):
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")
    rows_env = "\n".join(f"{tex(d)} & {tex(v)} \\\\" for d, v in env["packages"].items())
    rows_eff = "\n".join(f"{tex(k)} & {tex(v)} \\\\" for k, v in eff.items())
    (OUT / "repro_appendix.tex").write_text(
        "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{ll}\\toprule\nPackage & Version \\\\\\midrule\n"
        f"Python & {tex(env['python'])} \\\\\n{rows_env}\n\\bottomrule\\end{{tabular}}\n\\caption{{Software environment.}}\\end{{table}}\n\n"
        "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{ll}\\toprule\nParameter & Value \\\\\\midrule\n"
        f"{rows_eff}\n\\bottomrule\\end{{tabular}}\n\\caption{{Effective search hyper-parameters.}}\\end{{table}}\n")

    print(f"[repro] wrote {OUT}/REPRODUCIBILITY.md, repro_appendix.tex, environment.json, "
          f"requirements.lock.txt, data_checksums.json")
    if unused:
        print(f"[repro] WARNING: {len(unused)} config key(s) read by no script: {[k for k, _ in unused]}")
    if args.bundle:
        bundle(chk)


def bundle(chk):
    rel = PROJECT_ROOT / "release"
    rel.mkdir(exist_ok=True)
    readme = ("# ArchTree-LLM reproduction bundle\n\n"
              "Contains config, seeds, pinned environment, appendix, checksums and the full source code.\n"
              "NOT included (licences/size): NAS-Bench-201 (.pth), NAS-Bench-101, HW-NAS-Bench.\n"
              "Download them, verify against data_checksums.json, then follow the run order in REPRODUCIBILITY.md.\n"
              "Start with `python tools/build_nb201_cache.py` (fast JSON cache) and "
              "`python tools/build_igd_reference_front.py`.\n")
    with zipfile.ZipFile(rel / "archtree_repro_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README_RELEASE.md", readme)
        z.writestr("SEEDS.txt", " ".join(map(str, cfg["search"]["seeds"])) + "\n")
        z.write(PROJECT_ROOT / "config.yaml", "config.yaml")
        for f in ("REPRODUCIBILITY.md", "repro_appendix.tex", "environment.json", "requirements.lock.txt",
                  "data_checksums.json"):
            z.write(OUT / f, f)
        # Full source (Oct 2026: the old bundle shipped only the two cache builders). Never .env.
        for d in ("archtree", "benchmarks", "baselines", "experiments", "tools", "utils", "plots"):
            for f in sorted((PROJECT_ROOT / d).glob("*.py")):
                z.write(f, f"{d}/{f.name}")
        for f in ("requirements.txt", "run_full_pipeline.ps1", "setup_env.ps1", "setup_env.sh", "README.md",
                  "ReadMe/FIXES_OCT2026.md", "ReadMe/RUN_PLAN_OCT2026.md"):
            if (PROJECT_ROOT / f).exists():
                z.write(PROJECT_ROOT / f, f)
    print(f"[repro] bundle -> {rel / 'archtree_repro_bundle.zip'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hash", action="store_true")
    ap.add_argument("--bundle", action="store_true")
    build(ap.parse_args())