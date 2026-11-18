"""RaBitQ 1-bit estimator (float query) with c = data mean vs c = 0, on offset ('cone') data."""
import numpy as np
rng = np.random.default_rng(0)
N, D, NQ, K = 20000, 256, 200, 10
q_, _ = np.linalg.qr(rng.standard_normal((D, D)))
R = q_.astype(np.float32)
for offset in (0.0, 1.0, 3.0):
    m = offset * rng.standard_normal(D).astype(np.float32) / np.sqrt(D) * 4  # mean vector, length ~4*offset
    X = m + rng.standard_normal((N, D)).astype(np.float32) / np.sqrt(D)       # spread ~1
    Q = m + rng.standard_normal((NQ, D)).astype(np.float32) / np.sqrt(D)
    gt = np.argsort(((Q[:, None] - X[None]) ** 2).sum(-1), 1)[:, :K]
    print(f"\n|mean| = {np.linalg.norm(m):.2f}, within-cloud spread |x - mean| ~ 1.00")
    for name, c in (("c = 0", np.zeros(D, np.float32)), ("c = mean", X.mean(0))):
        r = (X - c) @ R.T; qc = (Q - c) @ R.T
        nr = np.linalg.norm(r, axis=1); o = r / nr[:, None]
        ob = np.where(r > 0, 1.0, -1.0) / np.sqrt(D)
        oo = (ob * o).sum(1)
        est = nr[None] ** 2 + (qc ** 2).sum(1)[:, None] - 2 * (nr / oo)[None] * (qc @ ob.T)
        I = np.argsort(est, 1)[:, :K]
        rec = np.mean([len(set(I[i]) & set(gt[i])) / K for i in range(NQ)])
        true = ((Q[:, None] - X[None]) ** 2).sum(-1)
        err = np.abs(est - true).mean()
        print(f"  {name:<9} mean <o_bar,o> {oo.mean():.3f}  mean |r| {nr.mean():.2f}  "
              f"mean |dist2 error| {err:.3f}  recall@10 {rec:.3f}")
