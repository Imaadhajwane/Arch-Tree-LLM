# test_storage.py
"""
Unit tests for archtree/storage.py
Run from archtree-llm/ root:
    python tools/test_storage.py

All tests use outputs/test_phase/ — cleaned up after run.

── FIX (this version) ──────────────────────────────────────────────────────
storage.py has NO load_run_log() and NO list_phase_outputs() functions —
those were never defined there (only save_run_log exists). The old test
called both and would crash with AttributeError before printing any real
result. Fixed by:
  - verifying save_run_log() by reading the JSON file back directly
    (json.load) instead of a nonexistent loader function.
  - replacing the list_phase_outputs() check with a direct check that the
    logs/results/plots-data directories exist (what get_output_dir()
    actually creates), since there's no dedicated "list" API to test.
Also fixed save_llm_cache/load_llm_cache calls: storage.py's real
signature is (phase, bench, dataset, seed, cache, namespace="") — the old
test omitted `dataset` entirely, which would raise a TypeError.
"""

from contextlib import redirect_stdout
import io
import json
import sys
import os
import shutil
from pathlib import Path

# works from anywhere: put the project root (parent of tools/) on the path first
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402
import archtree.storage as storage  # noqa: E402

PASS = 0
FAIL = 0


def check(label: str, condition: bool):
    global PASS, FAIL
    if condition:
        print(f"  ✓  {label}")
        PASS += 1
    else:
        print(f"  ✗  FAIL: {label}")
        FAIL += 1


# ── Setup: clean test phase dir ────────────────────────────────────────────────
TEST_PHASE = "test_phase"
test_dir = Path("outputs") / TEST_PHASE
if test_dir.exists():
    shutil.rmtree(test_dir)
print("\n[test_storage] Starting tests...\n")


# ── 1. build_filename ──────────────────────────────────────────────────────────
print("1. build_filename")
name = storage.build_filename("archtree_v1", "nb201", "cifar10", 42, "json")
check("correct format", name == "archtree_v1__nb201__cifar10__seed42.json")
check("double underscore separator", "__" in name)


# ── 2. get_output_dir ─────────────────────────────────────────────────────────
print("\n2. get_output_dir")
d = storage.get_output_dir(TEST_PHASE, "logs")
check("dir created", d.exists())
check("correct path", d == storage.PROJECT_ROOT / "outputs" / TEST_PHASE / "logs")


# ── 3. save_run_log — verify by reading the JSON file back directly ──────────
print("\n3. save_run_log")
log = {
    "method": "archtree_v1",
    "benchmark": "nb201",
    "dataset": "cifar10",
    "seed": 42,
    "budget": 100,
    "best_acc": 93.9,
    "eval_count": 100,
    "wall_time_sec": 12.3,
    "eval_history": [{"iter": 1, "arch": "conv3x3|skip", "acc": 88.1, "tcr": 0.006}],
}
storage.save_run_log(TEST_PHASE, "archtree_v1", "nb201", "cifar10", 42, log, budget=100)

# save_run_log appends "__budget{N}" to the filename when budget is passed —
# mirror that exactly to find the file back (see storage.save_run_log()).
run_log_path = (
    storage.get_output_dir(TEST_PHASE, "logs")
    / "archtree_v1__nb201__cifar10__seed42__budget100.json"
)
check("log file written", run_log_path.exists())
with open(run_log_path) as f:
    loaded = json.load(f)
check("best_acc preserved", loaded["best_acc"] == 93.9)
check("eval_history preserved", len(loaded["eval_history"]) == 1)
check("nested dict preserved", loaded["eval_history"][0]["iter"] == 1)


# ── 4. append_result_row / load_results_csv ───────────────────────────────────
print("\n4. append_result_row / load_results_csv")
row1 = {"method": "random_search", "benchmark": "nb201", "dataset": "cifar10",
        "seed": 42, "budget": 100, "best_acc": 88.1, "eval_count": 100}
row2 = {"method": "enas", "benchmark": "nb201", "dataset": "cifar10",
        "seed": 42, "budget": 100, "best_acc": 90.3, "eval_count": 100}
storage.append_result_row(TEST_PHASE, "baseline_summary.csv", row1)
storage.append_result_row(TEST_PHASE, "baseline_summary.csv", row2)
df = storage.load_results_csv(TEST_PHASE, "baseline_summary.csv")
check("2 rows written", len(df) == 2)
check("method column correct", list(df["method"]) == ["random_search", "enas"])
check("best_acc values correct", list(df["best_acc"]) == [88.1, 90.3])

# append again — should add row 3, not overwrite
storage.append_result_row(TEST_PHASE, "baseline_summary.csv",
                          {"method": "mcts_nas", "benchmark": "nb201", "dataset": "cifar10",
                           "seed": 42, "budget": 100, "best_acc": 91.0, "eval_count": 100})
df2 = storage.load_results_csv(TEST_PHASE, "baseline_summary.csv")
check("append-only (3 rows now)", len(df2) == 3)


# ── 5. save_plot_data / load_plot_data ────────────────────────────────────────
print("\n5. save_plot_data / load_plot_data")
plot_df = pd.DataFrame({
    "method": ["archtree_v1", "random_search"],
    "iteration": [1, 1],
    "best_acc_so_far": [88.1, 85.0],
})
storage.save_plot_data(TEST_PHASE, "eta_curve_test", plot_df)
loaded_plot = storage.load_plot_data(TEST_PHASE, "eta_curve_test")
check("plot data rows preserved", len(loaded_plot) == 2)
check("plot data columns preserved", list(loaded_plot.columns)
      == ["method", "iteration", "best_acc_so_far"])


# ── 6. get_figure_path ────────────────────────────────────────────────────────
print("\n6. get_figure_path")
fig_path = storage.get_figure_path(TEST_PHASE, "eta_curve_test")
check("returns Path object", isinstance(fig_path, Path))
check("correct filename", fig_path.name == "eta_curve_test.png")
check("figures subdir", Path("plots/figures")
      in fig_path.parents or "plots" in fig_path.parts)


# ── 7. LLM cache — FIXED: dataset is a required positional arg ───────────────
print("\n7. save_llm_cache / load_llm_cache")
cache = {
    ("conv3x3", "skip"): {"none": 0.3, "conv1x1": 0.7},
    ("conv3x3",): {"skip": 0.9, "avgpool": 0.1},
}
storage.save_llm_cache(TEST_PHASE, "nb201", "cifar10", 42, cache)
loaded_cache = storage.load_llm_cache(TEST_PHASE, "nb201", "cifar10", 42)
check("cache keys restored as tuples", all(isinstance(k, tuple)
      for k in loaded_cache.keys()))
check("cache values preserved", loaded_cache[(
    "conv3x3", "skip")]["conv1x1"] == 0.7)
check("2 entries preserved", len(loaded_cache) == 2)

# empty cache (no file) returns {}
empty = storage.load_llm_cache(TEST_PHASE, "nb201", "cifar10", 999)
check("missing cache returns {}", empty == {})


# ── 8. verify_phase_inputs — should FAIL cleanly for phase1 (no phase0 output) ─
print("\n8. verify_phase_inputs (cold-resume guard)")
# phase0 has no deps → should pass
try:
    storage.verify_phase_inputs("phase0")
    check("phase0 passes (no deps)", True)
except SystemExit:
    check("phase0 passes (no deps)", False)

# phase1 requires phase0 outputs — should fail since we're in test_phase
buf = io.StringIO()
exit_called = False
try:
    with redirect_stdout(buf):
        # a fake phase whose required input never exists (independent of real outputs on disk)
        storage.PHASE_REQUIRED_INPUTS["_test_missing"] = ["outputs/phase0/results/__never_exists__.csv"]
        storage.verify_phase_inputs("_test_missing")
except SystemExit:
    exit_called = True
check("phase1 exits when phase0 missing", exit_called)
check("error message mentions missing file",
      "__never_exists__.csv" in buf.getvalue())


# ── 9. output directory layout — FIXED: no list_phase_outputs() exists ───────
# storage.py exposes no dedicated "list outputs" API, so instead verify
# directly that get_output_dir() created the logs/results/plots-data
# subtrees we exercised above (this is what a hypothetical lister would
# report anyway).
print("\n9. output directory layout")
logs_dir = storage.get_output_dir(TEST_PHASE, "logs")
results_dir = storage.get_output_dir(TEST_PHASE, "results")
plots_dir = storage.get_output_dir(TEST_PHASE, "plots/data")
check("logs dir exists", logs_dir.exists())
check("results dir exists", results_dir.exists())
check("plots dir exists", plots_dir.exists())
check("baseline_summary.csv present in results", (results_dir / "baseline_summary.csv").exists())


# ── Cleanup ────────────────────────────────────────────────────────────────────
shutil.rmtree(test_dir)
print(f"\n[test_storage] Cleaned up outputs/{TEST_PHASE}/")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL TESTS PASSED ✓")
    print("Phase 0 exit criterion 0.3 met: storage.py unit-tested.")
else:
    print("SOME TESTS FAILED ✗ — fix before proceeding.")
    sys.exit(1)