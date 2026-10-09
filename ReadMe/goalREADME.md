# ArchTree-LLM — project goals and where each one stands

> **Working title:** *Neural Architecture Search via Trie-Based Deduplication and Zero-Cost Proxies, with
> Hardware-Aware Coverage: An Empirical Study of LLM-Guided Warm-Start*
>
> **Title that matches the final results (suggested, Oct 4, 2026):** *A Label-Free Zero-Cost Gate for
> Sample-Efficient NAS: Equal-GPU-Hour Evidence, a Gate-Cost Criterion, and What an LLM Prior Really Adds*
> (the final choice is the author's; the old original title was "ArchTree-LLM: A Hierarchical Trie-Based
> Search Space with LLM-Guided UCB Traversal and Zero-Cost Proxy Ensemble for Sample-Efficient
> Hardware-Aware NAS").

M.Tech Research Project — AY 2025-26 / 2026-27
**Author:** Imaad Hajwane | MIT-WPU
**Guide:** Dr. Jayshree Aher
**Target venue (updated Oct 4, 2026):** **TMLR** first (judges correctness and evidence, not novelty alone),
AutoML-Conf as the alternative, or a journal (Neural Networks, Applied Soft Computing, IEEE Access). Earlier
targets: AAAI / IJCAI / ECAI 2026 → master's seminar paper (Sep 2026) → TMLR / AutoML-Conf after the Oct 2026
v2 results.

*This file is about the GOALS: what the project set out to show, and whether each goal was met. All numbers,
tables and run details are in the master `README.md` (§9) and `ReadMe/RESULTS_LOG_OCT2026.md` (§5, §5.9), plus `ReadMe/FIXES_OCT2026.md`.*

**Status, Oct 4, 2026 (13:04): all experiments are done. The next step is writing the paper.**

---

## 1. Aim

Make neural architecture search cheaper by never evaluating the same architecture twice (a trie) and by
letting a cheap **label-free zero-cost proxy gate** decide which proposals deserve a real (expensive)
evaluation. Then test, honestly and under fair compute, whether an **LLM prior** and the **tree structure**
add anything on top.

The original plan combined five ideas (trie dedup, UCB traversal, LLM warm-start, zero-cost proxy ensemble,
hierarchical weight inheritance). The measured result is narrower, and it is what the paper claims:

> **Final core claim.** A label-free zero-cost gate (0 labels) inside a deduplicating trie / UCB search
> (ArchTree v2) reaches the same accuracy as the strongest baselines with ~40% of their real evaluations and
> 19-23% fewer GPU-hours. At **equal GPU-hours** it has **0 losses against 13 methods on 3 benchmarks** and
> is within the critical difference of the best method on all three (tied-best on NB201). The gate helps every
> search strategy it is added to. A closed-form criterion predicts **in advance** that correcting the gate
> for training cost hurts, and it does, on all 3 benchmarks. An LLM prior speeds up early search on
> cifar10 / cifar100 using general architecture knowledge (not memorisation, not a size heuristic), but never
> changes final accuracy.

---

## 2. Goals and outcomes (C1-C9 + the Oct 2026 additions)

| ID | Goal (as planned) | Outcome (Oct 4, 2026) | Status |
|---|---|---|---|
| C1 | Compressed trie with O(L) deduplication | 0 duplicate real evaluations in every ArchTree run (vs 30% for LLM-NAS, 67% for a bandit without dedup, averaged over budgets; 55% / 74% at budget 200) | **met** |
| C2 | UCB1 traversal with a regret bound | Informal argument only (`ReadMe/REGRET_ARGUMENT.md`). v2 fixes a reward-scale flaw (rank reward, C = 0.3) and is robust to C (0 of 78 tests) | **met, as an empirical finding; no formal proof** |
| C3 | Hierarchical weight inheritance (weight bank) | Implemented, but tabular benchmarks train nothing, so it is simulated and **not claimed** | **dropped from the claims** |
| C4 | Formal proofs (insert, pruning, weight consistency) | T1, T2, T5 hold; T4 (pruning keeps the optimum) is false as first stated and restated as a heuristic | **partly met** |
| C5 | Evaluation on NB201 (3 datasets) + NB101 + HW-NAS-Bench | NB201 ✓, **NB101 ✓** (unblocked: TF-free reader), HW-NAS-Bench ✓, **plus NATS-Bench SSS** (3rd benchmark, Oct 3) | **exceeded** |
| C6 | TCR coverage metric | Implemented and reported (TCR-H vs accuracy analysis, step 18); not a headline result | **met (minor)** |
| **C7** | **LLM warm-start as a semantic UCB prior** | Re-tested after the prior-bias fix: **2-3× faster early convergence on cifar10 / cifar100** (Holm-significant), none on ImageNet16-120, **same final accuracy**. Knowledge controls: near chance when op names are hidden, **better with op descriptions than with NB201 names** (not memorisation), and a size prior ranks ops better but searches worse (the LLM is more than "bigger is better") | **partly met: early speed-up, no final gain** |
| **C8** | **Zero-cost proxy ensemble (4 proxies, calibrated)** | **Label-free** ensemble (0 labels, τ 0.63-0.68) as good as the supervised one; saves 53% GPU-hours at equal accuracy; at equal GPU-hours it helps every proposer | **met: the main contribution** |
| **C9** | **Hardware-aware search + TCR-H** | After the HW-NAS-Bench index fix: device fronts barely overlap (Jaccard ~0.13; 1 of 15,625 archs Pareto-optimal on all 6 devices), but a front found for one device keeps ~98% of the hypervolume on another, and HW-aware ArchTree ties random / NSGA-II-style search (96 of 96) | **met as an analysis; no search advantage** |
| new | Fair compute: equal GPU-hours instead of equal evaluations | 1128 runs, 14 methods, NB201 + NB101 + NATS-SSS: 0 losses, rank tied-best / 2nd / 4th, within the critical difference everywhere | **met** |
| new | Explain *why* proxy gates work: are proxies just cost detectors? | Yes in cell spaces (ρ with training time 0.78 NB201, 0.63 NB101), no in the width space (0.20 NATS-SSS). The closed-form gate-cost criterion (slope r_PA·r_PD − r_DA) predicts "cost penalty hurts" on all 7 benchmark/dataset pairs; measured 16 / 21 losses, 0 wins; predicted vs measured Spearman +0.78 / +0.86 (sign right; size under-predicted) | **met: most novel result** |
| new | Does the tree structure itself help? (A7) | With dedup, trie = flat UCB (0 of 12). The gain comes from **deduplication**, not the hierarchy | **answered (negative)** |

---

## 3. How it's different from prior work (final framing)

| vs. | What this project adds |
|---|---|
| Proxy-filtered evolution / RE + ZC warm-up (Abdelfattah et al. 2021, the closest prior work) | a **label-free** gate inside a deduplicating search, tested at **equal GPU-hours** on 3 benchmarks, plus the gate-cost criterion explaining when cost correction helps or hurts |
| ZiCo / AZ-NAS / NAS-Bench-Suite-Zero (proxy studies) | proxies used as a *gate inside search*, and shown to be cost detectors in cell spaces but not in width spaces |
| MCTS / AlphaX / LaNAS (tree search) | dedup guarantee and a rank-normalised UCB reward. Shown honestly that the tree adds nothing beyond dedup |
| LLM-NAS, GENIUS, EvoPrompting (LLM-guided NAS) | LLM as a *prior over a structured space*, plus **knowledge controls** (anonymous / described prompts, size prior, model size) that separate general knowledge from memorisation and from a size heuristic |
| Local search (White et al. 2021) / regularized evolution | included as baselines. They are the strongest ones, and with the same gate local search beats ArchTree on NB101 / NATS-SSS (stated as a limitation) |

No "first" claims are made. The contribution is the gate, the evidence protocol and the explanations.

---

## 4. Benchmarks, baselines, metrics (final)

**Benchmarks:**

| Benchmark | Space | Datasets |
|---|---|---|
| NAS-Bench-201 | 15,625 cells | CIFAR-10, CIFAR-100, ImageNet16-120 |
| NAS-Bench-101 | 423,624 cells | CIFAR-10 |
| NATS-Bench SSS | 32,768 width configurations | CIFAR-10, CIFAR-100, ImageNet16-120 |
| HW-NAS-Bench | NB201 latencies on 6 devices | edgegpu, raspi4, edgetpu, pixel3, eyeriss, fpga |

Download / build: `data/README.md`, `third_party/README.md`.

**Baselines (9 external):** random search, regularized evolution, RE + zero-cost warm-up, local search,
ENAS-style, MCTS-style, ZiCo-style, RZ-style, LLM-NAS (gpt-oss-20b). Plus internal arms (no gate, #params gate,
supervised gate, cost-penalised gates, v1, flat UCB) and NSGA-II-style / random for the hardware part.

**Protocol:** search on validation, report the **test** accuracy of the best-validation architecture.
**12 seeds + 12 held-out seeds** (design checked on unseen seeds). Cost counted in real evaluations **and**
GPU-hours (the benchmark's training time of the architectures actually evaluated).

**Statistics:** paired Wilcoxon with Holm per family, Friedman + Nemenyi critical difference (CD diagrams),
TOST equivalence (±0.5 / ±0.25 points), probability of improvement with stratified-bootstrap CIs, Cohen's d.
Comparisons with no seed-to-seed variation are not tested.

**Primary metrics:** test accuracy at equal GPU-hours; real evaluations / GPU-hours to equal accuracy;
evaluations to within 1 point of the final result (convergence); HV / IGD (hardware); Kendall τ (proxies).

---

## 5. Formal statements (final status)

| Theorem | Statement | Status |
|---|---|---|
| T1 | Trie insert runs in O(L) | holds |
| T2 | `is_visited()` guarantees zero re-evaluations | holds, confirmed empirically (0 duplicates) |
| T3 | UCB1 traversal has sub-linear regret | informal argument only, not claimed as a theorem |
| T4 | Pruning never removes the optimal architecture | **false as stated** → value-based pruning, presented as a heuristic |
| T5 | Weight inheritance is consistent | holds (but weight inheritance is not part of the claims, see C3) |
| T6 | LLM prior with alpha decay keeps the UCB1 regret bound | near-trivial structural remark, not a contribution |
| T7 | TCR-H = 1 iff all Pareto-optimal architectures are found | definitional, not a contribution |

The paper is an **empirical** study. The theory is limited to T1 / T2 and the gate-cost criterion.

---

## 6. Milestones

| Phase | Planned | Status (Oct 4, 2026) |
|---|---|---|
| 0 | Environment, baselines, storage | ✓ (9 external baselines) |
| 1 | Trie + UCB1 (C1-C6) | ✓ (v2: rank reward, mutation) |
| 2 | Zero-cost proxy ensemble (C8) | ✓ (label-free gate chosen) |
| 3 | LLM warm-start (C7) | ✓ (re-tested Oct 3-4 with knowledge controls) |
| 4 | HW-NAS-Bench + Pareto UCB + TCR-H (C9) | ✓ (after the index fix) |
| 5 | Formal statements | ✓ (T4 restated, T3 informal) |
| 6 | Full campaign: all tables, plots, ablations | ✓ (Oct 2-4: core v2, NB101, extend, equal GPU-hours, Groq studies, final aggregate; 50 figures in `plots/figures/`) |
| **7** | **Paper writing + submission** | **next** |
| 8 | Reproducibility bundle (`release/`) | at submission: `.\run_full_pipeline.ps1 -Plan report` + NATS / HW checksums + result CSVs |

---

## 7. Limitations the paper must state (goal-relevant)

* Tabular benchmarks only: nothing is trained, and weight inheritance is simulated.
* ArchTree is not the best search strategy everywhere: local search with the same gate is stronger on NB101 /
  NATS-SSS. The portable contribution is the gate.
* The LLM prior gives no final-accuracy gain and no gain on ImageNet16-120.
* Hardware-aware search ties simpler multi-objective baselines.
* 12 seeds detect ~0.3-point differences but rarely prove equivalence within ±0.5 points (TOST), so the
  wording is "no significant difference".

Full list: master `README.md` §15 and `ReadMe/RESULTS_LOG_OCT2026.md` §7.

---

## 8. Where to find things

| What | Where |
|---|---|
| All results, tables, statistics | master `README.md` §9; full log `ReadMe/RESULTS_LOG_OCT2026.md` §5 (§5.9 = what the paper reports) |
| Figures + data points | `plots/figures/` (50 figures, each with a CSV) |
| Final tables | `outputs/final/results/` |
| What was fixed and why | `ReadMe/FIXES_OCT2026.md` (36 items) |
| How to rerun / rebuild | master `README.md` §10-11, `run_full_pipeline.ps1`, `command.txt`, `setup_env.ps1` |
| Datasets | `data/README.md`, `third_party/README.md` |
| Regret argument | `ReadMe/REGRET_ARGUMENT.md` |

---

*ArchTree-LLM | MIT-WPU | AY 2025-26 / 2026-27*
*Imaad Hajwane | Guide: Dr. Jayshree Aher*
