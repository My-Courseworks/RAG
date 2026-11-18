"""FAISS QT_8bit vs Qdrant int8 (replicated from encoded_vectors_u8.rs), same corpus.
Reports held-out reconstruction MSE and recall@10 of each one's own scoring path:
  FAISS : float query vs decoded codes (asymmetric), per-dim range, 256 levels, floor + 0.5
  Qdrant: query quantized too (symmetric); L2 score = step^2 * sum (cq - cv)^2 = ||q_hat - v_hat||^2
"""
import numpy as np, faiss

N, D, NQ, K, NTR = 187_043, 256, 1000, 10, 150_000


def topics():
    r = np.random.default_rng(0); c = r.standard_normal((50, D), dtype=np.float32)
    t = r.integers(0, 50, N + NQ); x = c[t] + 0.35 * r.standard_normal((N + NQ, D), dtype=np.float32)
    return x[:N], x[N:]


def embedlike():
    r = np.random.default_rng(1); s = (np.arange(1, D + 1) ** -0.5).astype(np.float32)
    q, _ = np.linalg.qr(r.standard_normal((D, D)))
    z = r.standard_normal((N + NQ, D)).astype(np.float32) * s
    x = (z @ q.T.astype(np.float32)) + 0.3 * s[0] * r.standard_normal(D).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x[:N].astype(np.float32), x[N:].astype(np.float32)


def qdrant_encode(x, lo, step):
    return np.clip((x - lo) / step, 0, 127).round()


def recall(I, gt):
    return np.mean([len(set(I[i][:K]) & set(gt[i])) / K for i in range(len(gt))])


def topk_l2(qs, base, k):
    out = []
    bn = (base ** 2).sum(1)
    for i in range(0, len(qs), 100):
        q = qs[i:i + 100]
        d = bn[None, :] - 2 * q @ base.T
        out.append(np.argpartition(d, k, axis=1)[:, :k][np.arange(len(q))[:, None], np.argsort(np.take_along_axis(d, np.argpartition(d, k, axis=1)[:, :k], 1), 1)])
    return np.vstack(out)


def rescored(cand, xb, xq):
    return np.vstack([c[np.argsort(((xb[c] - xq[i]) ** 2).sum(1))] for i, c in enumerate(cand)])


for name, (xb, xq) in (("50-topic mixture", topics()), ("embedding-like", embedlike())):
    flat = faiss.IndexFlatL2(D); flat.add(xb); _, gt = flat.search(xq, K)
    tr, ho = xb[:NTR], xb[NTR:]
    # FAISS
    sq = faiss.IndexScalarQuantizer(D, faiss.ScalarQuantizer.QT_8bit); sq.train(tr); sq.add(xb)
    f_dec_ho = sq.sa_decode(sq.sa_encode(ho))
    f_mse = ((f_dec_ho - ho) ** 2).sum(1).mean()
    _, If = sq.search(xq, K); _, If40 = sq.search(xq, 4 * K)
    # Qdrant replication: one global range from the training vectors
    lo, hi = tr.min(), tr.max(); step = (hi - lo) / 127
    q_dec_ho = lo + qdrant_encode(ho, lo, step) * step
    q_mse = ((q_dec_ho - ho) ** 2).sum(1).mean()
    vb = (lo + qdrant_encode(xb, lo, step) * step).astype(np.float32)
    vq = (lo + qdrant_encode(xq, lo, step) * step).astype(np.float32)
    Iq = topk_l2(vq, vb, K); Iq40 = topk_l2(vq, vb, 4 * K)
    print(f"\n{name} (N = {N:,}, d = {D}; ranges fit on the first {NTR:,}, MSE on the other {N - NTR:,})")
    print(f"  {'':<26}{'MSE / vector':>14}{'recall@10':>11}{'4x + rescore':>14}")
    print(f"  {'FAISS QT_8bit':<26}{f_mse:>14.3e}{recall(If, gt):>11.3f}{recall(rescored(If40, xb, xq), gt):>14.3f}")
    print(f"  {'Qdrant int8 (replicated)':<26}{q_mse:>14.3e}{recall(Iq, gt):>11.3f}{recall(rescored(Iq40, xb, xq), gt):>14.3f}")
    print(f"  ratio Qdrant / FAISS MSE: {q_mse / f_mse:.1f}x")
