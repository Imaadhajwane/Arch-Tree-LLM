# `third_party/` — external code used as-is

* This folder holds **unmodified** copies of external projects that are not on PyPI.
Nothing in here was changed, and the project's own code never edits these files. They're used only through
the wrappers in `benchmarks/`.

| Folder | What | Used by | Needed for |
|---|---|---|---|
| `HW-NAS-Bench/` | HW-NAS-Bench API + dataset (Li et al., ICLR 2021) | `benchmarks/hwnb.py` | Phase 4 (hardware-aware search, step 9), HW true Pareto fronts (step 3), cross-device analysis (step 18), TCR-H replay |

If this folder is missing, every experiment that doesn't touch hardware still runs: NB201, NB101, NATS-SSS,
the LLM studies and equal GPU-hours. Only the steps above need it, and `tools/preflight.py` reports it
as "HW-NAS-Bench repo + pickle".

---

## HW-NAS-Bench

**What it is:** a lookup table with the measured **latency** (and energy) of all **15,625 NAS-Bench-201
architectures** (and the FBNet space) on 6 hardware devices. Example entry from the authors' README
(ms / mJ): `edgegpu_latency 5.81, edgegpu_energy 24.2, raspi4_latency 10.5, edgetpu_latency 0.96,
pixel3_latency 3.61, eyeriss_latency 3.65, eyeriss_energy 0.69, fpga_latency 2.57, fpga_energy 18.0`.

| Device key | Hardware |
|---|---|
| `edgegpu` | edge GPU (latency + energy) |
| `raspi4` | Raspberry Pi 4 |
| `edgetpu` | Edge TPU (some pairs unmeasured, see below) |
| `pixel3` | Pixel 3 phone |
| `eyeriss` | Eyeriss ASIC (latency + energy) |
| `fpga` | FPGA (latency + energy) |

This project uses the **NAS-Bench-201 space** and **latency** only (`{device}_latency`, in ms), on CIFAR-10.
Accuracy always comes from our own NB201 cache (`data/nb201_cache_v2.json`), not from HW-NAS-Bench.

### Install

It ships with its data (the 10.8 MB pickle is inside the repository), so a clone is all you need:

```powershell
git clone https://github.com/GATECH-EIC/HW-NAS-Bench.git third_party/HW-NAS-Bench
```
(The repository was previously at `RICE-EIC/HW-NAS-Bench`; the old URL in `benchmarks/hwnb.py` points to
the same project.) No `pip install` is needed: `benchmarks/hwnb.py` adds the folder to `sys.path` and imports
`hw_nas_bench_api` directly. Its own prerequisites (python ≥ 3.6, pytorch ≥ 1.2, numpy ≥ 1.18) are already
covered by `requirements.txt`.

### Contents (as cloned)

```
HW-NAS-Bench/
├── HW-NAS-Bench-v1_0.pickle   # the dataset (10,782,467 bytes) - the only file the code reads
├── hw_nas_bench_api/          # HWNASBenchAPI (+ fbnet_models/, nas_201_models/ model definitions)
├── nas_201_api/               # a bundled copy of the NAS-Bench-201 API (not used; we use pip nas-bench-201)
├── dev/proj_edgegpu/          # the authors' Edge-GPU measurement project (~15,625 files; not used)
├── example.ipynb, devices.jpg # authors' examples
├── README.md                  # the authors' README
└── LICENSE                    # MIT, Copyright (c) 2022 GaTech-EIC
```

### How the code uses it (`benchmarks/hwnb.py`)

```python
from hw_nas_bench_api import HWNASBenchAPI
api = HWNASBenchAPI("third_party/HW-NAS-Bench/HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")
api.query_by_index(hw_index, "cifar10")["edgegpu_latency"]
```

* **Paths:** `config.yaml` → `paths.hwnb_repo: "third_party/HW-NAS-Bench"`. The wrapper finds the `.pickle`
  inside that folder by itself. `paths.hwnb` (`data/...pickle`) is informational only.
* **Index mapping (important, fixed Oct 2026):** HW-NAS-Bench numbers architectures in NAS-Bench-201's
  *original* order, while this project uses the *canonical* base-5 index of the 6 op tokens. The two orders
  differ for 15,624 of the 15,625 architectures. On load, the wrapper turns every entry's `arch_str` into
  tokens and builds a verified one-to-one canonical → HW-NAS-Bench map, and every lookup goes through it.
  Check: `check_alignment()`, the Spearman correlation between latency and the number of conv ops of the same
  architecture. It is ≈ 0.6 when aligned and ≈ 0 when not (the pre-fix Sep-2026 results had 0.001 and are invalid).
* **Unmeasured pairs:** HW-NAS-Bench returns a **negative** latency for (architecture, device) pairs it didn't
  measure (e.g. some `edgetpu` entries). The wrapper raises `HWMetricUnavailable`. The real evaluation is still
  counted, but the architecture is kept out of Pareto / HV / IGD.
* **Derived files** written to `data/`: `true_pareto_front__<device>.json` (step 3,
  `tools/build_igd_reference_front.py`) and `hw_table__cifar10.npz` (step 18 lookup cache). See `data/README.md`.

### Verify

```powershell
python tools/preflight.py                         # "HW-NAS-Bench repo + pickle  ok"
Get-FileHash third_party\HW-NAS-Bench\HW-NAS-Bench-v1_0.pickle -Algorithm SHA256
#   57F92C2D7B88814FA337D00F4597C8CA30D507C5A855C18D6FAC57E5AD872700
python -c "import sys; sys.path.insert(0,'third_party/HW-NAS-Bench'); from hw_nas_bench_api import HWNASBenchAPI as A; a=A('third_party/HW-NAS-Bench/HW-NAS-Bench-v1_0.pickle', search_space='nasbench201'); print(a.query_by_index(0,'cifar10')['edgegpu_latency'])"
```

### Licence and citation

MIT licence (`HW-NAS-Bench/LICENSE`, Copyright (c) 2022 GaTech-EIC). The folder is kept unmodified, with its
licence. `.gitignore` excludes the pickle (`third_party/HW-NAS-Bench/*.pickle`), so re-clone rather than
commit it.

> Li, C., Yu, Z., Fu, Y., Zhang, Y., Zhao, Y., You, H., Yu, Q., Wang, Y., Lin, Y.
> *HW-NAS-Bench: Hardware-Aware Neural Architecture Search Benchmark.* ICLR 2021 (spotlight).

---

## Not in this folder (installed from PyPI instead)

| Package | Why it isn't vendored |
|---|---|
| `nas-bench-201` (NB201 API) | `pip install nas-bench-201`; only needed to rebuild `data/nb201_cache_v2.json` |
| `nats_bench` (NATS-Bench API) | `pip install nats_bench`; only needed to rebuild `data/natssss_cache.json` |
| `nasbench` (NB101, Google) | **not used**: it needs TensorFlow 1.x. `tools/build_nb101_cache.py` reads the tfrecord without it |

Dataset downloads, cache builds and checksums: `data/README.md`.
