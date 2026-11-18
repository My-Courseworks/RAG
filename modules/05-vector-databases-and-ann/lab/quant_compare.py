"""SQ8 vs PQ vs plain binary vs RaBitQ, flat (no IVF), same data, recall@10 vs exact.
Two synthetic corpora, 187,043 x 256: (a) the module's 50-topic mixture, (b) an embedding-like
cloud: unit-norm vectors with a power-law spectrum and no tight clusters."""
import time, sys
import numpy as np
import faiss

N, D, NQ, K = 187_043, 256, 1000, 10


def topics():
    rng = np.random.default_rng(0)
    c = rng.standard_normal((50, D), dtype=np.float32)
    t = rng.integers(0, 50, N + NQ)
    x = c[t] + 0.35 * rng.standard_normal((N + NQ, D), dtype=np.float32)
    return x[:N], x[N:]


def embedlike():
    rng = np.random.default_rng(1)
    s = (np.arange(1, D + 1) ** -0.5).astype(np.float32)          # decaying spectrum
    q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    z = rng.standard_normal((N + NQ, D)).astype(np.float32) * s
    x = (z @ q.T.astype(np.float32)) + 0.3 * s[0] * rng.standard_normal(D).astype(np.float32)  # off-centre mean
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x[:N].astype(np.float32), x[N:].astype(np.float32)


def recall(I, gt):
    return np.mean([len(set(I[i][:K]) & set(gt[i])) / K for i in range(len(gt))])


def rescored(cand, xb, xq, gt):
    r = 0.0
    for i in range(len(xq)):
        c = cand[i][cand[i] >= 0]
        d = ((xb[c] - xq[i]) ** 2).sum(1)
        r += len(set(c[np.argsort(d)[:K]]) & set(gt[i])) / K
    return r / len(xq)


def run(name, xb, xq):
    flat = faiss.IndexFlatL2(D); flat.add(xb); _, gt = flat.search(xq, K)
    rows = []

    def eval_index(label, nbytes, search):           # search(k) -> I
        I1 = search(K); I10 = search(K * 10); I100 = search(K * 100)
        rows.append((label, nbytes, recall(I1, gt), rescored(I10, xb, xq, gt), rescored(I100, xb, xq, gt)))
        print(f"  {label:<38}{nbytes:>6} B{rows[-1][2]:>9.3f}{rows[-1][3]:>12.3f}{rows[-1][4]:>13.3f}", flush=True)

    print(f"\n{name}: recall@10 (raw code order | rescored from 100 | rescored from 1000 candidates)")
    print(f"  {'method':<38}{'bytes':>8}{'raw':>9}{'100 cand.':>12}{'1000 cand.':>13}")
    # SQ8
    ix = faiss.IndexScalarQuantizer(D, faiss.ScalarQuantizer.QT_8bit); ix.train(xb[:100_000]); ix.add(xb)
    eval_index("SQ8 (QT_8bit, per-dim range)", D, lambda k: ix.search(xq, k)[1])
    # PQ at 64 and 32 bytes
    for m in (64, 32):
        p = faiss.IndexPQ(D, m, 8); p.train(xb[:100_000]); p.add(xb)
        eval_index(f"PQ{m}x8", m, lambda k, p=p: p.search(xq, k)[1])
    # plain binary: sign of raw values, Hamming
    for lab, mu in (("binary, x > 0 (as stored)", np.zeros(D, np.float32)),
                    ("binary, centred first (x - mean > 0)", xb.mean(0))):
        bb = np.packbits(xb - mu > 0, axis=1); bq = np.packbits(xq - mu > 0, axis=1)
        bi = faiss.IndexBinaryFlat(D); bi.add(bb)
        eval_index(lab, D // 8, lambda k, bi=bi, bq=bq: bi.search(bq, k)[1])
    # RaBitQ flat, 1-bit (query 4-bit default, and float query), and 4-bit
    rq = faiss.index_factory(D, f"RR{D},RaBitQ"); rq.train(xb[:100_000]); rq.add(xb)
    inner = faiss.downcast_index(rq.index)
    eval_index("RaBitQ 1-bit (query 4-bit, default)", D // 8 + 8, lambda k: rq.search(xq, k)[1])
    inner.qb = 0
    eval_index("RaBitQ 1-bit (float query)", D // 8 + 8, lambda k: rq.search(xq, k)[1])
    try:
        rq4 = faiss.index_factory(D, f"RR{D},RaBitQ4"); rq4.train(xb[:100_000]); rq4.add(xb)
        cs = faiss.downcast_index(rq4.index).code_size
        eval_index("RaBitQ 4-bit (1 sign + 3 extra bits)", cs, lambda k: rq4.search(xq, k)[1])
    except Exception as e:
        print("  RaBitQ4 unavailable:", e)


which = sys.argv[1:] or ["topics", "embedlike"]
if "topics" in which:
    run("(a) 50-topic mixture", *topics())
if "embedlike" in which:
    run("(b) embedding-like (unit norm, power-law spectrum, off-centre)", *embedlike())
