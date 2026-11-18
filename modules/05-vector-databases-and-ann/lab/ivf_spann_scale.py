"""Scale test placeholder for SPANN (SPTAG) -- SKIPPED on this machine.

The course's shared generator is kept here verbatim so the file is drop-in once
SPTAG builds. SPTAG is a C++/CMake project needing Boost (+ SWIG for Python);
on this macOS arm64 host `brew list boost` finds nothing and `cmake` is absent,
so no build was attempted and NO numbers are produced. Nothing below is fabricated.

If you have a working `SPTAG` Python module, fill in `build_and_search()`:
  - Base:          ValueType=Float, DistCalcMethod=L2, Dim=256, IndexAlgoType=BKT
  - SelectHead:    isExecute=true, Count=4096 (== nlist; sets Ratio = 4096/N),
                   BKTKmeansK=32, BKTLeafSize=8, SelectThreshold=0 (auto), SplitThreshold=0, SplitFactor=0
  - BuildHead:     isExecute=true
  - BuildSSDIndex: isExecute=true, BuildSsdIndex=true, InternalResultNum=64, ReplicaCount=8,
                   PostingPageLimit=3, RNGFactor=1.0
  - query dial:    SearchInternalResultNum in (1,2,4,8,16,32,64,128,256)  (== nprobe), MaxDistRatio=10000
Run:  python -I ivf_spann_scale.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

N, D, NQ, K = 187_043, 256, 1000, 10


def make_data():
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((50, D), dtype=np.float32)
    topic = rng.integers(0, 50, N + NQ)
    x = centres[topic] + 0.35 * rng.standard_normal((N + NQ, D), dtype=np.float32)
    return x[:N], x[N:]


def brute_force_top10(xb, xq, chunk=200):
    out = np.empty((len(xq), K), dtype=np.int64)
    nb2 = (xb * xb).sum(1)
    for s in range(0, len(xq), chunk):
        q = xq[s:s + chunk]
        d = nb2[None, :] - 2.0 * (q @ xb.T) + (q * q).sum(1)[:, None]
        out[s:s + chunk] = np.argpartition(d, K, axis=1)[:, :K]
    return out


def main():
    try:
        import SPTAG  # noqa: F401
    except ImportError:
        print("SPANN scale test SKIPPED: the SPTAG Python module is not importable on this machine "
              "(no Boost/cmake/SWIG build). No recall/latency numbers are reported. "
              "See supplementary/ivf/spann.md section 6 for the paper's own numbers.")
        sys.exit(0)

    xb, xq = make_data()
    t = time.perf_counter()
    gt = brute_force_top10(xb, xq)
    print(f"ground truth: {time.perf_counter() - t:.1f}s")
    raise NotImplementedError("fill in build_and_search() with the SPTAG SPANN config in the module docstring")


if __name__ == "__main__":
    main()
