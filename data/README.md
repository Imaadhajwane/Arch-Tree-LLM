# `data/` — benchmark files, derived caches and how to get them

Matches `config.yaml` (`paths:`) and every experiment of the paper.*

Every script reads its data through the paths in `config.yaml`, so nothing else in the code has to change.
**Don't edit files in this folder by hand.** The caches are rebuilt only with the `tools/` scripts listed below.

There are two kinds of files here:

1. **Raw benchmark files.** You download these once. They're only needed to *build* the caches, and
   the experiments never read them directly.
2. **Derived caches.** Small, fast files built from the raw files. **All experiments read only these.**

If the derived caches are present, the raw files are no longer needed and can be moved out of the
project. The NATS-Bench download already has been.

---

## 1. Overview

| Benchmark | Used for | Raw file (download) | Derived cache (what the code reads) | Built by |
|---|---|---|---|---|
| **NAS-Bench-201** | main benchmark: all phases, ablations, LLM studies, equal GPU-hours | `NAS-Bench-201-v1_1-096897.pth` (5.0 GB) | `nb201_cache_v2.json` (4.9 MB) | step 2, `tools/build_nb201_cache.py` |
| **NAS-Bench-101** | second benchmark: steps 21, 22 | `nasbench_only108.tfrecord` (523 MB) | `nb101_cache.npz` (22 MB) | step 20, `tools/build_nb101_cache.py` |
| **NATS-Bench SSS** (size space) | third benchmark: step 22 (equal GPU-hours, cost analysis) | `NATS-sss-v1_0-50262-simple.tar` (1.01 GB) or `NATS-sss-v1_0-50262.pickle.pbz2` (874 MB) | `natssss_cache.json` (12.8 MB) | step 23, `tools/build_natsbench_sss_cache.py` |
| **HW-NAS-Bench** | hardware-aware search: Phase 4, cross-device analysis | lives in `third_party/HW-NAS-Bench/` (see `third_party/README.md`) | `true_pareto_front__<device>.json` (6 files), `hw_table__cifar10.npz` | step 3 `tools/build_igd_reference_front.py`; step 18 `tools/analyze_cross_device_pareto.py` |
| **CIFAR-10** (images) | 3 fixed mini-batches of 256 images for the zero-cost proxies | `cifar10/` (170 MB) | – | downloaded automatically by torchvision |

The datasets inside the benchmarks (CIFAR-10, CIFAR-100, ImageNet16-120) are **not** needed as images.
The benchmarks contain the already-trained accuracies. Only the CIFAR-10 mini-batch is real image data.


## 2. Download and build, step by step

All commands run from the project root with the environment active (see `setup_env.ps1`).
Check every download against the SHA-256 checksums in section 5.

### 2.1 NAS-Bench-201 (Dong & Yang, ICLR 2020)

* **What it is:** a cell-based search space. There are 6 edges, each with 5 operations (`none`, `skip_connect`,
  `nor_conv_1x1`, `nor_conv_3x3`, `avg_pool_3x3`), so 5^6 = **15,625 architectures**. Every architecture was
  trained on CIFAR-10, CIFAR-100 and ImageNet16-120.
* **Download:** `NAS-Bench-201-v1_1-096897.pth` (5.0 GB), from the Google Drive link in the official repository
  https://github.com/D-X-Y/NAS-Bench-201 (the same link is in `third_party/HW-NAS-Bench/README.md`).
  Put it in `data/`.
* **API:** `pip install nas-bench-201` (in `requirements.txt`).
* **Build the cache** (CPU is fine, but it needs about 10-16 GB RAM to load the .pth file):
  ```powershell
  python tools/build_nb201_cache.py          # -> data/nb201_cache_v2.json   (pipeline step 2)
  ```
* **What the cache stores** (v2): for every architecture and dataset, 200-epoch accuracies averaged over the
  benchmark's training seeds. It uses the split protocol of the NB201 paper: **search on validation, report test**.

  | Dataset | search metric (valid) | report metric (test) |
  |---|---|---|
  | cifar10 | `cifar10-valid` / x-valid | `cifar10` / ori-test |
  | cifar100 | x-valid | x-test |
  | ImageNet16-120 | x-valid | x-test |

  It also stores `train_sec`, the training time of one real evaluation, which is used for GPU-hour accounting.
  Keys are the canonical base-5 index of the op sequence.
* The old `nb201_cifar10_cache.json` (v1) stored 12-epoch cifar10 **training** accuracy. **Never use it.**
  The background is in `ReadMe/FIXES_OCT2026.md`.

### 2.2 NAS-Bench-101 (Ying et al., ICML 2019)

* **What it is:** cell DAGs with up to 7 nodes and 9 edges, giving **423,624 unique architectures**, trained on
  CIFAR-10 (3 repeats at 108 epochs).
* **Download:** `nasbench_only108.tfrecord` (523 MB) from
  https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord. Put it in `data/`.
* **No TensorFlow needed.** The official `nasbench` package needs TensorFlow 1.x, which no longer installs
  on current Windows / Python. `tools/build_nb101_cache.py` reads the tfrecord directly, with a minimal
  protobuf parser.
  ```powershell
  python tools/build_nb101_cache.py          # -> data/nb101_cache.npz   (pipeline step 20, ~3-8 min, ~2 GB RAM)
  python tools/test_nb101.py                 # pure-function tests (step 0)
  ```
* **The cache refuses to build unless all three checks pass:**
  1. every graph has exactly 3 repeats;
  2. every stored module hash is reproduced by the pure-Python port in `benchmarks/nb101.py`;
  3. there are exactly 423,624 graphs.
* **Stores:** module hash, adjacency matrix, ops, #vertices, #params, validation / test accuracy of the 3 repeats
  (search on the mean validation, report the mean test) and the training time, for CIFAR-10 only.

### 2.3 NATS-Bench, size search space "SSS" (Dong et al., TPAMI 2021)

* **What it is:** the cell is **fixed** and only the channel widths of 5 layers vary, each one of
  {8, 16, ..., 64}. That gives 8^5 = **32,768 architectures**, trained on CIFAR-10, CIFAR-100 and ImageNet16-120.
  Training cost varies a lot and largely independently of the cell design, which is why the paper uses it
  to test the "zero-cost proxies are cost detectors" finding.
* **Download:** from the Google Drive linked in https://github.com/D-X-Y/NATS-Bench, get **one** of:
  * `NATS-sss-v1_0-50262-simple.tar` (1.01 GB). **Preferred**, because it needs little RAM. Extract it:
    `tar -xf <path>\NATS-sss-v1_0-50262-simple.tar -C data`, which gives `data\NATS-sss-v1_0-50262-simple\`.
  * `NATS-sss-v1_0-50262.pickle.pbz2` (874 MB), loaded into RAM in one go (~10 GB+). Put it in `data/`.
* **API:** `pip install nats_bench` (in `requirements.txt`).
  ```powershell
  python tools/build_natsbench_sss_cache.py  # -> data/natssss_cache.json   (pipeline step 23, no GPU)
  ```
* **Stores:** `hp = "90"`, the 90-epoch schedule of the NATS-Bench paper, `is_random=False`, with the same split
  protocol as NB201 (valid for search, test for reporting). It also stores `train_sec`, `params` (M) and
  `flops` (M). Keys are the base-8 code of the 5 widths, with the first layer most significant.
* **The build ends with a sanity check:** the widest network (64:64:64:64:64) must have more parameters and a
  higher accuracy than the narrowest (8:8:8:8:8).
* **After the cache is built, the download (tar / folder / pbz2) can be deleted.** On this machine it already has been.

### 2.4 HW-NAS-Bench (Li et al., ICLR 2021)

Measured latency and energy of all 15,625 NB201 architectures on 6 devices (edgegpu, raspi4, edgetpu,
pixel3, eyeriss, fpga). The data ships **inside its git repository**, so it isn't downloaded into `data/`.
See `third_party/README.md`.

Derived files written here:
```powershell
python tools/build_igd_reference_front.py      # true_pareto_front__<device>.json  (step 3; true accuracy/latency
                                               #   front over all 15,625 archs, for IGD / hypervolume in Phase 4)
python tools/analyze_cross_device_pareto.py    # hw_table__cifar10.npz  (step 18; lookup cache, safe to delete)
```

Note: `config.yaml` → `paths.hwnb` (`data/HW-NAS-Bench-v1_0.pickle`) is informational only.
`benchmarks/hwnb.py` finds the pickle inside `paths.hwnb_repo` (`third_party/HW-NAS-Bench`).

### 2.5 CIFAR-10 images (Krizhevsky, 2009)

The zero-cost proxies (SynFlow, NASWOT, SVD entropy) need real input data: 3 fixed mini-batches of 256 CIFAR-10 images
(`NASWOT_BATCH_SIZE`, `N_NASWOT_BATCHES` in `archtree/zc_proxy.py`). `archtree/zc_proxy.py`
calls torchvision, which downloads `cifar-10-python.tar.gz` (170 MB) into `data/cifar10/` the first time it
runs (log line: `Files already downloaded and verified`). There's nothing to do by hand.
To place it manually, download https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz into `data/cifar10/`.

---

## 3. Check that everything is in place

```powershell
python tools/preflight.py          # checks every step's data (NB201 / NB101 / NATS-SSS caches, HW repo, fronts, ...)
.\setup_env.ps1 -CheckOnly         # packages + GPU + data + .env presence
```

---

## 4. Checksums (SHA-256 of the files used for the paper)

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `NAS-Bench-201-v1_1-096897.pth` | 5,003,180,601 | `2317a7dc911654322f4f932730a821cf38ecc5b5f1adde0aff26ff6b3f736666` |
| `nasbench_only108.tfrecord` | 522,767,376 | `4c39c3936e36a85269881d659e44e61a245babcb72cb374eacacf75d0e5f4fd1` |
| `nb201_cache_v2.json` | 4,894,199 | `b6f19d2d0adee7b09d3a9ff098a379695d3c224cdcf5dafeb25f0924afee2694` |
| `nb101_cache.npz` | 22,126,761 | `4e0fb9901ff6952fc1d345c6ee1be8f6c7b4406c0f0843bbd255bc14a302d19e` |
| `natssss_cache.json` | 12,828,328 | `8cac2574ab56b07dcf9b82461e03d0195a34a0187eb4239b362f940cdd28d18a` |
| `third_party/HW-NAS-Bench/HW-NAS-Bench-v1_0.pickle` | 10,782,467 | `57f92c2d7b88814fa337d00f4597c8ca30d507c5a855c18d6fac57e5ad872700` |

Windows: `Get-FileHash <file> -Algorithm SHA256`. Linux / macOS: `sha256sum <file>`.
Step 19 (`tools/build_repro_appendix.py`) writes the checksums to `outputs/final/repro/data_checksums.json`.

---

## 5. Licences and citations

Licences are as stated in each project's repository; check them before redistributing any file. The raw
benchmark files are **not** redistributed with this project. The reproducibility bundle ships only checksums.

| Benchmark | Licence (repository) | Cite |
|---|---|---|
| NAS-Bench-201 | see https://github.com/D-X-Y/NAS-Bench-201 | Dong & Yang, *NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search*, ICLR 2020 |
| NAS-Bench-101 | Apache-2.0 (https://github.com/google-research/nasbench) | Ying et al., *NAS-Bench-101: Towards Reproducible Neural Architecture Search*, ICML 2019 |
| NATS-Bench | see https://github.com/D-X-Y/NATS-Bench | Dong et al., *NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size*, TPAMI 2021 |
| HW-NAS-Bench | MIT (`third_party/HW-NAS-Bench/LICENSE`) | Li et al., *HW-NAS-Bench: Hardware-Aware Neural Architecture Search Benchmark*, ICLR 2021 |
| CIFAR-10 / CIFAR-100 | – | Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009 |
| ImageNet16-120 | (through the benchmarks) | Chrabaszcz et al., *A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets*, 2017 |

---

