"""cuVS IVF-Flat scale test (module 5 common generator): NOT RUNNABLE ON THIS MAC.

cuVS (https://github.com/rapidsai/cuvs) ships GPU-only kernels; there is no CPU
fallback for IVF-Flat / IVF-PQ and PyPI only has manylinux wheels
(`cuvs-cu12`, `cuvs-cu13`). On macOS this script prints that and exits 0 without
fabricating numbers. On a Linux box with an NVIDIA GPU and
`pip install cuvs-cu12 cupy-cuda12x` it runs the course's standard sweep.

Run:  python -I lab/ivf_cuvs_scale.py
"""

from __future__ import annotations

import platform
import sys
import time

import numpy as np

N, D, NQ, K = 187_043, 256, 1000, 10
NLISTS = (4096, 1024)
NPROBES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def make_data():
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((50, D), dtype=np.float32)
    topic = rng.integers(0, 50, N + NQ)
    x = centres[topic] + 0.35 * rng.standard_normal((N + NQ, D), dtype=np.float32)
    return x[:N], x[N:]


def ground_truth(xb, xq, chunk=4096):
    gt = np.empty((len(xq), K), dtype=np.int64)
    xb_norm = (xb * xb).sum(1)
    for s in range(0, len(xq), chunk):
        q = xq[s:s + chunk]
        d = xb_norm[None, :] - 2.0 * q @ xb.T  # + |q|^2 is constant per row
        gt[s:s + chunk] = np.argsort(d, axis=1)[:, :K]
    return gt


def recall_at_k(found, gt):
    return float(np.mean([len(set(f) & set(g)) / K for f, g in zip(found, gt)]))


def run_cuvs(xb, xq, gt):
    import cupy as cp  # noqa: F401  (device arrays)
    from cuvs.neighbors import ivf_flat

    for nlist in NLISTS:
        t0 = time.perf_counter()
        index = ivf_flat.build(ivf_flat.IndexParams(n_lists=nlist, metric="sqeuclidean"),
                               cp.asarray(xb))
        cp.cuda.Device().synchronize()
        build_s = time.perf_counter() - t0
        print(f"\nn_lists={nlist}  build {build_s:.1f}s  "
              f"(cuVS keeps full vectors: ~{N * D * 4 / 2**20:.0f} MiB + list padding)")
        print(f"{'n_probes':>9} {'recall@10':>10} {'ms/query':>9}")
        first = None
        xq_d = cp.asarray(xq)
        for npb in NPROBES:
            params = ivf_flat.SearchParams(n_probes=npb)
            ivf_flat.search(params, index, xq_d, K)  # warm-up (JIT kernels)
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            dist, ids = ivf_flat.search(params, index, xq_d, K)
            cp.cuda.Device().synchronize()
            ms = (time.perf_counter() - t0) * 1000 / NQ
            r = recall_at_k(cp.asnumpy(ids), gt)
            if first is None and r >= 0.95:
                first = npb
            print(f"{npb:>9} {r:>10.4f} {ms:>9.4f}")
        print(f"smallest n_probes with recall>=0.95: {first}")


def main() -> None:
    try:
        import cuvs  # noqa: F401
        have_cuvs = True
    except ImportError:
        have_cuvs = False
    if not have_cuvs:
        print("cuVS scale test: SKIPPED on this machine.")
        print(f"  platform = {platform.platform()}, python = {sys.version.split()[0]}")
        print("  cuVS has no CPU backend for IVF and publishes manylinux-only wheels; "
              "`pip install cuvs-cu12` fails on macOS with 'No matching distribution found'.")
        print("  No recall/QPS numbers are reported for cuVS at scale (nothing was fabricated).")
        print("  To run: Linux + NVIDIA GPU, `pip install cuvs-cu12 cupy-cuda12x`, then rerun this file.")
        return
    xb, xq = make_data()
    gt = ground_truth(xb, xq)
    run_cuvs(xb, xq, gt)


if __name__ == "__main__":
    main()
