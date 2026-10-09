# tools/gate_cost_model.py
"""
A small model of WHEN a zero-cost gate should be corrected for cost (Oct 3, 2026).

Setting. Each architecture has a search accuracy A (validation), a zero-cost proxy score P, a
training cost C (GPU time of one real evaluation) and a predicted cost D known without training
(#params). The gate really evaluates a proposal only if its score S = z(P) - lambda * z(D) is in the
top q of the scores seen so far; lambda = 0 is the plain gate, lambda > 0 penalises predicted cost.
Search runs until a GPU-hour budget H is spent.

Gaussian-copula model. Replace every variable by its normal score z = Phi^-1(rank / (n + 1)) and
let R be their correlation matrix (r_PA = corr(P, A), r_PD, r_DA, ...).

Proposition (selection quality). corr(S, A) = (r_PA - lambda r_DA) / sqrt(1 + lambda^2 - 2 lambda r_PD),
so  d corr(S, A) / d lambda at lambda = 0  =  r_PA r_PD - r_DA.
A small cost penalty makes the gate pick better architectures per evaluation iff
    r_DA  <  r_PA * r_PD          ("accuracy is less cost-coupled than the proxy's cost bias implies").
If r_DA > r_PA r_PD the penalty lowers the quality of what the gate lets through, and it can only pay
off through the extra (cheaper) evaluations it buys - which the simulation below quantifies.

Prediction under a GPU-hour budget. Monte Carlo of gated RANDOM search in the copula model:
draws z ~ N(0, R), maps A, test accuracy T and cost C back through their empirical quantiles,
applies the exact gate rule of the experiments (WARMUP proposals always evaluated, then top-q of
the scores seen so far), stops before the budget is exceeded, and reports the test accuracy of the
best-validation architecture. delta_pred(lambda, k) = E[best | lambda] - E[best | lambda = 0].
experiments/run_equal_gpu_hours.py compares it with the measured delta of the arms
random_costpen* / archtree_costpen* vs their plain-gate arms (predict-then-confirm).

    from tools.gate_cost_model import fit_copula, criterion, predict
"""
from typing import Dict, Iterable, List

import numpy as np
from scipy.stats import norm, rankdata

VARS = ("A", "T", "P", "C", "D")    # valid acc, test acc, proxy, train cost, predicted cost (#params)


def normal_scores(x) -> np.ndarray:
    x = np.asarray(x, float)
    return norm.ppf(rankdata(x) / (len(x) + 1.0))


def fit_copula(A, T, P, C, D) -> Dict:
    """Normal-score correlation matrix + the empirical marginals needed to map back."""
    Z = np.column_stack([normal_scores(v) for v in (A, T, P, C, D)])
    R = np.corrcoef(Z, rowvar=False)
    R = (R + R.T) / 2
    w, V = np.linalg.eigh(R)                         # guard: nearest positive-definite matrix
    R = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    return {"R": R, "A": np.sort(np.asarray(A, float)), "T": np.asarray(T, float),
            "Aidx": np.argsort(np.asarray(A, float)), "C": np.sort(np.asarray(C, float)),
            "TA": np.asarray(T, float)[np.argsort(np.asarray(A, float))]}


def r(model, a, b) -> float:
    return float(model["R"][VARS.index(a), VARS.index(b)])


def criterion(model) -> Dict[str, float]:
    """Closed-form quantities of the proposition."""
    r_pa, r_pd, r_da = r(model, "P", "A"), r(model, "P", "D"), r(model, "D", "A")
    return {"r_PA": r_pa, "r_PD": r_pd, "r_DA": r_da, "r_PC": r(model, "P", "C"), "r_AC": r(model, "A", "C"),
            "r_DC": r(model, "D", "C"),
            "slope_at_0": r_pa * r_pd - r_da,            # > 0: a small cost penalty improves selection
            "penalty_helps_selection": bool(r_pa * r_pd > r_da)}


def corr_SA(model, lam: float) -> float:
    r_pa, r_pd, r_da = r(model, "P", "A"), r(model, "P", "D"), r(model, "D", "A")
    return (r_pa - lam * r_da) / np.sqrt(1 + lam ** 2 - 2 * lam * r_pd)


def _quantile(sorted_vals, u):
    n = len(sorted_vals)
    return sorted_vals[np.clip((u * n).astype(int), 0, n - 1)]


def _gate_mask(scores_row, q: float, warmup: int) -> np.ndarray:
    """Exact gate rule of the experiments: the first `warmup` proposals pass; afterwards a proposal
    passes iff its score >= np.percentile(scores seen so far, 100 (1 - q)) (linear interpolation)."""
    import bisect
    seen, out = [], np.ones(len(scores_row), bool)
    pct = 1 - q
    for j, s in enumerate(scores_row):
        if j < warmup:
            continue
        bisect.insort(seen, s)
        pos = (len(seen) - 1) * pct
        lo = int(pos)
        hi = min(lo + 1, len(seen) - 1)
        thr = seen[lo] + (seen[hi] - seen[lo]) * (pos - lo)
        out[j] = s >= thr
    return out


def predict(model, lambdas: Iterable[float], ks: Iterable[int], q: float = 0.35, warmup: int = 10,
            n_runs: int = 300, max_props: int = 1500, seed: int = 0) -> List[Dict]:
    """Expected test accuracy of the best-validation architecture found by gated random search
    within H = k x mean cost, for each lambda (common random numbers across lambdas)."""
    rng = np.random.default_rng(seed)
    Lc = np.linalg.cholesky(model["R"])
    z = rng.standard_normal((n_runs, max_props, len(VARS))) @ Lc.T
    u = norm.cdf(z)
    ia = np.clip((u[..., 0] * len(model["A"])).astype(int), 0, len(model["A"]) - 1)
    A = model["A"][ia]
    T = model["TA"][ia]                               # test acc of the architecture with that valid rank
    C = _quantile(model["C"], u[..., 3])
    zP, zD = z[..., 2], z[..., 4]
    mean_c = float(model["C"].mean())
    ks = list(ks)
    out = []
    for lam in lambdas:
        S = zP - lam * zD
        res = {k: ([], []) for k in ks}
        for i in range(n_runs):
            m = _gate_mask(S[i], q, warmup)
            a, t, c = A[i][m], T[i][m], C[i][m]
            cum = np.cumsum(c)
            for k in ks:
                n = int(np.searchsorted(cum, k * mean_c, side="right"))   # stop before exceeding H
                if n == 0:
                    res[k][0].append(np.nan)
                else:
                    res[k][0].append(t[int(np.argmax(a[:n]))])
                res[k][1].append(n)
        for k in ks:
            out.append({"lambda": lam, "k": k, "pred_best_test": float(np.nanmean(res[k][0])),
                        "pred_real_evals": float(np.mean(res[k][1])), "corr_S_A": float(corr_SA(model, lam))})
    base = {o["k"]: o["pred_best_test"] for o in out if o["lambda"] == 0}
    for o in out:
        o["pred_delta_vs_lambda0"] = o["pred_best_test"] - base.get(o["k"], np.nan)
    return out
