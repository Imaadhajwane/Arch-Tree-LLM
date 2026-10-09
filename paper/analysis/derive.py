"""Derived tables from the raw benchmark caches (no search is run). Writes CSVs to ./out."""
import sys, os, random, numpy as np, pandas as pd
from load import ROOT
sys.path.insert(0, ROOT)
from benchmarks.nb101 import tokens_to_spec, spec_hash
from load import *
from scipy.stats import spearmanr, kendalltau
os.makedirs("out", exist_ok=True)
nb = load_nb201(); na = load_nats(); n1 = load_nb101(); raw = load_raw()

# ---------------- proxy scores per benchmark (label-free, Eq. 2) ----------------
rows201 = [raw["|".join(nb201_tokens(i))] for i in range(15625)]
P201, X201 = label_free_score(rows201, [rows201[i] for i in random.Random(0).sample(range(15625), 1500)])
C201 = np.array([r["C"] for r in rows201])

TOK = [f"c{c}" for c in [8, 16, 24, 32, 40, 48, 56, 64]]
def nats_tok(i):
    out = []
    for _ in range(5): out.append(TOK[i % 8]); i //= 8
    return out[::-1]
rowsN = [raw["|".join(nats_tok(i))] for i in range(32768)]
PN, XN = label_free_score(rowsN, [rowsN[i] for i in random.Random(0).sample(range(32768), 1500)])

h2i = {h: i for i, h in enumerate(n1["hashes"])}
keys101 = [k for k in raw if len(k.split("|")) == 26]
idx101, rows101 = [], []
for k in keys101:
    m, ops = tokens_to_spec(k.split("|"))
    h = spec_hash(m, ops)
    if h in h2i:
        idx101.append(h2i[h]); rows101.append(raw[k])
idx101 = np.array(idx101)
_, uniq = np.unique(idx101, return_index=True)
idx101 = idx101[uniq]; rows101 = [rows101[i] for i in uniq]
P101, X101 = label_free_score(rows101, rows101[:1500])
print("NB101 proxy sample:", len(idx101))

bench = {}
for ds in DS:
    bench[("NB201", ds)] = dict(P=P201, X=X201, v=nb[ds]["valid"], t=nb[ds]["test"], s=nb[ds]["sec"])
    bench[("NATS-SSS", ds)] = dict(P=PN, X=XN, v=na[ds]["valid"], t=na[ds]["test"], s=na[ds]["sec"], prm=na["params"])
bench[("NB101", "cifar10")] = dict(P=P101, X=X101, v=100 * n1["valid"].mean(1)[idx101], t=100 * n1["test"].mean(1)[idx101],
                                   s=n1["train_time"].mean(1)[idx101], prm=n1["params"][idx101].astype(float))

# ---------------- dataset statistics (full spaces) ----------------
rec = []
def st(b, ds, v, t, s, n_ep, n):
    rec.append(dict(benchmark=b, dataset=ds, n=n, epochs=n_ep, test_mean=t.mean(), test_sd=t.std(), test_median=np.median(t),
                    test_max=t.max(), test_p99=np.percentile(t, 99), test_p90=np.percentile(t, 90), valid_max=v.max(),
                    sec_mean=s.mean(), sec_median=np.median(s), sec_min=s.min(), sec_max=s.max(), sec_cv=s.std() / s.mean(),
                    rho_valid_test=spearmanr(v, t).statistic, rho_test_sec=spearmanr(t, s).statistic,
                    gap_top1pct=t.max() - np.percentile(t, 99)))
for ds in DS: st("NB201", ds, nb[ds]["valid"], nb[ds]["test"], nb[ds]["sec"], 200, 15625)
st("NB101", "cifar10", 100 * n1["valid"].mean(1), 100 * n1["test"].mean(1), n1["train_time"].mean(1), 108, len(n1["hashes"]))
for ds in DS: st("NATS-SSS", ds, na[ds]["valid"], na[ds]["test"], na[ds]["sec"], 90, 32768)
pd.DataFrame(rec).to_csv("out/dataset_stats.csv", index=False)

# ---------------- gate operating curve ----------------
qs = np.round(np.arange(0.05, 1.0001, 0.05), 2)
rec = []
for (b, ds), d in bench.items():
    P, v, t, s = d["P"], d["v"], d["t"], d["s"]
    order = np.argsort(-P)
    top1 = v >= np.percentile(v, 99); top5 = v >= np.percentile(v, 95)
    for q in qs:
        keep = order[: int(round(q * len(P)))]
        rec.append(dict(benchmark=b, dataset=ds, q=q, recall_top1=top1[keep].mean() / top1.mean() * q,
                        recall_top5=top5[keep].mean() / top5.mean() * q, mean_test_kept=t[keep].mean(),
                        best_test_kept=t[keep][np.argmax(v[keep])], rel_cost_kept=s[keep].mean() / s.mean(),
                        rel_cost_oracle=s[np.argsort(-v)[: len(keep)]].mean() / s.mean()))
pd.DataFrame(rec).to_csv("out/gate_operating.csv", index=False)

# ---------------- proxy correlation matrices ----------------
rec = []
for (b, ds), d in bench.items():
    cols = {"E": d["X"][:, 0], "S": d["X"][:, 1], "T": d["X"][:, 2], "P": d["P"], "acc": d["v"], "time": d["s"]}
    if "prm" in d: cols["params"] = d["prm"]
    for a in cols:
        for c in cols:
            rec.append(dict(benchmark=b, dataset=ds, x=a, y=c, rho=spearmanr(cols[a], cols[c]).statistic))
    rec.append(dict(benchmark=b, dataset=ds, x="P", y="acc_tau", rho=kendalltau(d["P"], d["v"]).statistic))
pd.DataFrame(rec).to_csv("out/proxy_corr.csv", index=False)

# ---------------- arrays for the dataset figure ----------------
np.savez_compressed("out/space_arrays.npz",
    nb201_t=np.stack([nb[ds]["test"] for ds in DS]), nb201_s=np.stack([nb[ds]["sec"] for ds in DS]),
    nats_t=np.stack([na[ds]["test"] for ds in DS]), nats_s=np.stack([na[ds]["sec"] for ds in DS]), nats_p=na["params"],
    nb101_t=100 * n1["test"].mean(1), nb101_s=n1["train_time"].mean(1), nb101_p=n1["params"],
    nb201_P=P201, nats_P=PN, nb101_P=P101, nb101_idx=idx101, nb201_v=np.stack([nb[ds]["valid"] for ds in DS]))
print("done")
