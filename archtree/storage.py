# archtree/storage.py
"""
File I/O for all phases: results CSVs, run logs, weight banks, ZC calibration,
LLM cache, and Groq call counters.

- Small JSON side files are written atomically (temp file + os.replace), and
  loaded tolerantly (missing/empty/corrupt -> default), so an interrupted write
  can't permanently break later runs.
- ZC calibration is scoped per (bench, dataset). A proxy fit on cifar10 is not
  guaranteed to rank cifar100 / ImageNet16-120 correctly.
- The LLM cache is scoped per dataset, so one dataset can't replay another's
  cached scores as fake cache hits.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Atomic / tolerant JSON helpers ────────────────────────────────────────

def _atomic_write_json(path: Path, obj) -> None:
    """Write to a sibling .tmp file, then os.replace() it into place, so an
    interrupted write never leaves a truncated target. The replace is retried
    briefly because Windows (antivirus, OneDrive sync) can transiently lock it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    last_err = None
    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except OSError as e:
            last_err = e
            time.sleep(0.05 * (attempt + 1))
    raise last_err


def _load_json_tolerant(path: Path, default):
    """Read JSON; return `default` if the file is missing, empty, or corrupt.
    A bad file self-heals on the next save instead of crashing every run."""
    if not path.exists():
        return default
    try:
        with open(path) as f:
            content = f.read()
        if not content.strip():
            print(f"[storage] WARNING: {path} is empty — treating as no data. "
                  f"It will be rewritten correctly on the next save.")
            return default
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"[storage] WARNING: {path} is corrupt (not valid JSON) — treating as no data. "
              f"It will be rewritten correctly on the next save.")
        return default


# ── Paths and naming ──────────────────────────────────────────────────────

def get_output_dir(phase: str, subdir: str) -> Path:
    root = PROJECT_ROOT / "outputs" / phase / subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_filename(method: str, bench: str, dataset: str, seed: int, ext: str) -> str:
    return f"{method}__{bench}__{dataset}__seed{seed}.{ext}"


# ── Phase dependency check ────────────────────────────────────────────────

PHASE_REQUIRED_INPUTS = {
    "phase1": ["outputs/phase0/results/baseline_summary.csv"],
    "phase2": [
        "outputs/phase1/results/phase1_main_results.csv",
        "outputs/phase1/results/phase1_tcr_per_iter.csv",
    ],
    "phase3": ["outputs/phase2/results/phase2_main_results.csv"],
    "phase4": ["outputs/phase3/results/phase3_main_results.csv"],
    "ablations": [
        "outputs/phase3/results/phase3_main_results.csv",
        "outputs/phase4/results/phase4_pareto_results.csv",
        "outputs/phase4/results/phase4_hv_igd_summary.csv",
    ],
    "final": [
        "outputs/phase1/results/phase1_main_results.csv",
        "outputs/phase2/results/phase2_main_results.csv",
        "outputs/phase3/results/phase3_main_results.csv",
        "outputs/phase4/results/phase4_pareto_results.csv",
        "outputs/ablations/results/ablation_A1_no_llm.csv",
    ],
}


def verify_phase_inputs(phase: str) -> None:
    """Exit with a clear message if upstream outputs for `phase` are missing."""
    required = PHASE_REQUIRED_INPUTS.get(phase, [])
    missing = [p for p in required if not (PROJECT_ROOT / p).exists()]
    if missing:
        print(f"\n[storage] ERROR: Cannot start {phase}. Missing upstream outputs:")
        for m in missing:
            print(f"  x  {m}")
        print("\nRun preceding phase master script first, then retry.")
        sys.exit(1)
    print(f"[storage] All upstream inputs verified for {phase}. Starting...")


# ── Run logs and result CSVs ──────────────────────────────────────────────

def save_run_log(phase, method, bench, dataset, seed, log, budget=None) -> None:
    fname = build_filename(method, bench, dataset, seed, "json")
    if budget is not None:
        fname = fname.replace(".json", f"__budget{budget}.json")
    path = get_output_dir(phase, "logs") / fname
    _atomic_write_json(path, log)
    print(f"[storage] log -> {path}")


def clear_result_file(phase: str, filename: str) -> None:
    """Delete a results CSV if present. Call once at the top of scripts that
    rebuild a file from scratch, so append_result_row() doesn't stack
    duplicate rows on a stale earlier run."""
    path = get_output_dir(phase, "results") / filename
    if path.exists():
        path.unlink()


def append_result_row(phase: str, filename: str, row: dict) -> None:
    # The header comes from the first row's keys. Adding a key to rows of an
    # existing file raises ValueError, so new metrics need a new file.
    path = get_output_dir(phase, "results") / filename
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_result_rows(phase: str, filename: str, rows) -> None:
    """Append many rows with ONE file open (same CSV as calling append_result_row per row)."""
    rows = list(rows)
    if not rows:
        return
    path = get_output_dir(phase, "results") / filename
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# Per-iteration logs (TCR curves) used to open/close their CSV once per search ITERATION -
# hundreds of thousands of tiny appends, each scanned by the antivirus on Windows. They are now
# buffered in memory and written in blocks (Oct 4, 2026). The files are byte-identical; call
# flush_row_buffers() before reading such a file in the same process (done at the end of every
# search function and at interpreter exit).
_ROW_BUFFER = {}          # (phase, filename) -> list of rows
_ROW_BUFFER_MAX = 5000


def append_result_row_buffered(phase: str, filename: str, row: dict) -> None:
    buf = _ROW_BUFFER.setdefault((phase, filename), [])
    buf.append(row)
    if len(buf) >= _ROW_BUFFER_MAX:
        append_result_rows(phase, filename, buf)
        buf.clear()


def flush_row_buffers() -> None:
    for (phase, filename), buf in list(_ROW_BUFFER.items()):
        if buf:
            append_result_rows(phase, filename, buf)
            buf.clear()


import atexit as _atexit  # noqa: E402
_atexit.register(flush_row_buffers)


def load_results_csv(phase: str, filename: str) -> pd.DataFrame:
    path = get_output_dir(phase, "results") / filename
    return pd.read_csv(path)


def save_plot_data(phase: str, plot_name: str, df: pd.DataFrame) -> None:
    path = get_output_dir(phase, "plots/data") / f"{plot_name}.csv"
    df.to_csv(path, index=False)
    print(f"[storage] plot data -> {path}")


def load_plot_data(phase: str, plot_name: str) -> pd.DataFrame:
    path = get_output_dir(phase, "plots/data") / f"{plot_name}.csv"
    return pd.read_csv(path)


def get_figure_path(phase: str, plot_name: str, ext: str = "png") -> Path:
    return get_output_dir(phase, "plots/figures") / f"{plot_name}.{ext}"


# ── Weight bank ───────────────────────────────────────────────────────────

def _weight_bank_path(phase, bench, dataset, seed, namespace: str = "") -> Path:
    tag = f"__{namespace}" if namespace else ""   # namespace keeps ablation arms apart
    fname = build_filename("weight_bank", bench, dataset, seed, "pt").replace(".pt", f"{tag}.pt")
    return get_output_dir(phase, "weights") / fname


def save_weight_bank(phase, bench, dataset, seed, state_dict: dict, namespace: str = "") -> None:
    torch.save(state_dict, _weight_bank_path(phase, bench, dataset, seed, namespace))


def load_weight_bank(phase, bench, dataset, seed, namespace: str = "") -> dict:
    path = _weight_bank_path(phase, bench, dataset, seed, namespace)
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)


# ── ZC proxy calibration (scoped per bench + dataset) ─────────────────────

def _zc_cal_path(phase: str, bench: str, dataset: str, mode: str = "supervised") -> Path:
    tag = "" if mode == "supervised" else f"__{mode}"
    return get_output_dir(phase, "weights") / f"zc_calibration_weights__{bench}__{dataset}{tag}.pt"


def save_zc_calibration(phase: str, bench: str, dataset: str, blob: dict,
                        mode: str = "supervised") -> None:
    path = _zc_cal_path(phase, bench, dataset, mode)
    torch.save(blob, path)
    print(f"[storage] zc calibration ({mode}) -> {path}")


def load_zc_calibration(phase: str, bench: str, dataset: str, mode: str = "supervised") -> dict:
    path = _zc_cal_path(phase, bench, dataset, mode)
    if path.exists():
        return torch.load(path, weights_only=False)
    raise FileNotFoundError(
        f"[storage] no ZC calibration ({mode}) for phase={phase} bench={bench} dataset={dataset} "
        f"at {path}. Run experiments/run_phase2.py first. (The old bench-only legacy fallback "
        f"was removed in Oct 2026: it silently gave every dataset a cifar10-fit proxy.)")


# ── LLM prior cache (Phase 3) ─────────────────────────────────────────────

def _llm_cache_path(phase: str, bench: str, dataset: str, seed: int, namespace: str = "") -> Path:
    tag = f"__{namespace}" if namespace else ""
    return get_output_dir(phase, "logs") / f"llm_cache__{bench}__{dataset}__seed{seed}{tag}.json"


def load_llm_cache(phase: str, bench: str, dataset: str, seed: int, namespace: str = "") -> dict:
    path = _llm_cache_path(phase, bench, dataset, seed, namespace)
    raw = _load_json_tolerant(path, {})
    # JSON keys are "tok1|tok2|..."; the empty string is the root (empty tuple)
    return {tuple(k.split("|")) if k else tuple(): v for k, v in raw.items()}


def save_llm_cache(phase: str, bench: str, dataset: str, seed: int, cache: dict, namespace: str = "") -> None:
    path = _llm_cache_path(phase, bench, dataset, seed, namespace)
    serializable = {"|".join(k): v for k, v in cache.items()}
    _atomic_write_json(path, serializable)


# ── Groq call counter ─────────────────────────────────────────────────────

def _groq_counter_path(phase, bench, dataset, seed, namespace="") -> Path:
    tag = f"__{namespace}" if namespace else ""
    return get_output_dir(phase, "logs") / f"groq_calls__{bench}__{dataset}__seed{seed}{tag}.json"


def load_groq_counter(phase, bench, dataset, seed, namespace="") -> int:
    path = _groq_counter_path(phase, bench, dataset, seed, namespace)
    return _load_json_tolerant(path, {}).get("count", 0)


def save_groq_counter(phase, bench, dataset, seed, count, namespace="") -> None:
    path = _groq_counter_path(phase, bench, dataset, seed, namespace)
    _atomic_write_json(path, {"count": count})

# ── Report metric (Oct 2026) ──────────────────────────────────────────────
def to_report_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where `best_acc` holds the REPORT metric (config benchmarks.report_metric,
    default best_test_acc = test accuracy of the architecture chosen by validation accuracy).
    The search-metric value is kept as `best_acc_search`. Files without the report column
    (pre-Oct-2026 runs) are returned unchanged, so old outputs still load."""
    from archtree import settings
    col = settings.REPORT_METRIC
    if df is None or col not in getattr(df, "columns", []) or col == "best_acc":
        return df
    if df[col].isna().all():
        return df
    out = df.copy()
    out["best_acc_search"] = out["best_acc"]
    out["best_acc"] = out[col].where(out[col].notna(), out["best_acc"])
    return out


def load_report_csv(phase: str, filename: str) -> pd.DataFrame:
    """load_results_csv + to_report_metric. Use for tables, tests and figures."""
    return to_report_metric(load_results_csv(phase, filename))
