"""Scalar, binary and product quantization on one 8-vector, 4-d toy, checked against FAISS.

Every number the README's quantization section quotes for its toy comes from here.
Run: python -I quantization_toy.py   (needs numpy, faiss-cpu)

The hand-written encoders are checked against FAISS where FAISS has the same thing:
  scalar  -> faiss.ScalarQuantizer(QT_8bit) (per-dimension min/max)
  binary  -> numpy sign bits + faiss.IndexBinaryFlat (Hamming)
  PQ      -> faiss.IndexPQ (codebooks and codes read back; ADC re-done by hand)
"""
import numpy as np
import faiss

X = np.array([
    [1.0, 2.0, -1.0, 0.5],   # v1
    [1.2, 1.8, -0.8, 0.7],   # v2  (near v1)
    [4.0, 0.5, 2.0, -1.5],   # v3
    [4.2, 0.3, 1.8, -1.7],   # v4  (near v3)
    [-2.0, -1.0, 3.0, 2.5],  # v5
    [-2.2, -0.8, 3.2, 2.3],  # v6  (near v5)
    [0.0, 3.5, -3.0, -2.5],  # v7
    [0.2, 3.3, -2.8, -2.7],  # v8  (near v7)
], dtype=np.float32)
NAMES = [f"v{i}" for i in range(1, 9)]  # not the HNSW toy's 1-d A..H
Q = np.array([[1.1, 1.9, -0.9, 0.6]], dtype=np.float32)  # sits next to v1 and v2
d = X.shape[1]
np.set_printoptions(precision=3, suppress=True, linewidth=120)


def hr(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# ---------------------------------------------------------------- scalar
hr("SCALAR (int8)")
lo, hi = X.min(0), X.max(0)
step = (hi - lo) / 255
print("per-dimension min :", lo)
print("per-dimension max :", hi)
print("per-dimension step:", step, " (= (max-min)/255)")


def sq_encode(v, rule):
    t = (v - lo) / (hi - lo) * 255
    return (np.floor(t) if rule == "floor" else np.round(t)).clip(0, 255).astype(np.uint8)


def sq_decode(c, rule):
    return lo + (c + (0.5 if rule == "floor" else 0.0)) * step


sqf = faiss.ScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
sqf.train(X)
fa_codes = sqf.compute_codes(X)
fa_dec = sqf.decode(fa_codes)
rule = next(r for r in ("floor", "round")
            if np.array_equal(sq_encode(X, r), fa_codes) and np.allclose(sq_decode(sq_encode(X, r), r), fa_dec, atol=1e-5))
print(f"FAISS rule reproduced exactly by: code = {rule}((x - min)/(max - min) * 255)"
      + ("; decode = min + (code + 0.5) * step" if rule == "floor" else "; decode = min + code * step"))
print("v1 :", X[0], "->", fa_codes[0], "->", fa_dec[0])
print("worst per-component error over the 8 vectors:", np.abs(fa_dec - X).max(), " (half a step is", (step / 2).max(), ")")
print("bytes: 4 x 4 B =", 4 * d, "B  ->", d, "B (4x)")

# quantile clipping, as Qdrant's `quantile` parameter describes it
x_out = np.vstack([X, [[40.0, 0.0, 0.0, 0.0]]]).astype(np.float32)   # one outlier in dim 0
outlier_step = (x_out[:, 0].max() - x_out[:, 0].min()) / 255
print(f"dim 0 step without the outlier: {step[0]:.4f}; with one value at 40: {outlier_step:.4f} "
      f"({outlier_step / step[0]:.1f}x coarser grid for every other vector)")

# ---------------------------------------------------------------- binary
hr("BINARY (1 bit)")
bits = (X > 0).astype(np.uint8)
print("rule: bit = 1 if component > 0 else 0")
for n, r, b in zip(NAMES, X, bits):
    print(f"{n}: {r} -> {b}")
packed = np.packbits(bits, axis=1)  # 4 bits -> one byte, padded
qbits = (Q > 0).astype(np.uint8)
ham = (bits != qbits).sum(1)
print("query:", Q[0], "->", qbits[0])
print("Hamming distance query->each:", dict(zip(NAMES, ham.tolist())))
bi = faiss.IndexBinaryFlat(8)  # 8 bits = 1 byte per vector in FAISS (4 used)
bi.add(packed)
Dh, Ih = bi.search(np.packbits(qbits, axis=1), 8)
assert [int(v) for v in Dh[0]] == sorted(ham.tolist()), "FAISS Hamming disagrees with hand count"
print("FAISS IndexBinaryFlat ranking:", [NAMES[i] for i in Ih[0]], "distances", Dh[0].tolist())
exact = np.argsort(((X - Q) ** 2).sum(1))
print("exact L2 ranking             :", [NAMES[i] for i in exact])
print("ties in Hamming:", {int(k): int(v) for k, v in zip(*np.unique(ham, return_counts=True)) if v > 1},
      "(distance -> how many vectors share it)")
print("v7=[0.0, 3.5, ...] and v8=[0.2, 3.3, ...] differ by 0.2 in dim 0 but get different first bits:", bits[6], bits[7])
print("bytes: 4 x 4 B =", 4 * d, "B  ->", d / 8, "B in a real 1,536-d vector (32x); here padded to 1 B")

# ---------------------------------------------------------------- PQ
hr("PRODUCT QUANTIZATION (m = 2 chunks of 2 dims, nbits = 2 -> 4 centroids per chunk)")
M, NBITS = 2, 2


def build_pq(nredo):
    ix = faiss.IndexPQ(d, M, NBITS)
    ix.pq.cp.seed = 1234
    ix.pq.cp.niter = 25
    ix.pq.cp.nredo = nredo
    ix.train(X)
    ix.add(X)
    return ix


def pq_worst_error(ix):
    return float(np.abs(ix.reconstruct_n(0, len(X)) - X).max())


single = build_pq(1)
print(f"one k-means run (nredo=1): worst reconstruction error {pq_worst_error(single):.3f}; "
      f"codes shared by distinct points: "
      f"{len(X) - len({tuple(r) for r in faiss.vector_to_array(single.codes).reshape(len(X), -1).tolist()})} collisions")
pq = build_pq(25)
print(f"best of 25 restarts (nredo=25): worst reconstruction error {pq_worst_error(pq):.3f}  <- used below")
cent = faiss.vector_to_array(pq.pq.centroids).reshape(M, 2 ** NBITS, d // M)
codes = pq.pq.compute_codes(X) if hasattr(pq.pq, "compute_codes") else None
codes = faiss.vector_to_array(pq.codes).reshape(len(X), -1)
# unpack 2-bit codes: M=2 chunks x 2 bits = 4 bits -> packed into one byte, chunk 0 in the low bits
idx = np.stack([(codes[:, 0] >> (NBITS * m)) & (2 ** NBITS - 1) for m in range(M)], 1)
for m in range(M):
    print(f"chunk {m} (dims {m * 2}-{m * 2 + 1}) codebook:")
    for j, c in enumerate(cent[m]):
        print(f"   centroid {j}: {c}")
rec = np.concatenate([cent[m][idx[:, m]] for m in range(M)], 1)
for n, r, i, rr in zip(NAMES, X, idx, rec):
    print(f"{n}: {r} -> codes {i.tolist()} -> decoded {rr}  (error {np.abs(rr - r).max():.3f})")
fa_dec_pq = pq.reconstruct_n(0, len(X))
assert np.allclose(fa_dec_pq, rec, atol=1e-5), "hand decode disagrees with FAISS"
# asymmetric distance computation, by hand: table per chunk, then M lookups
table = np.stack([((cent[m] - Q[0, m * 2:m * 2 + 2]) ** 2).sum(1) for m in range(M)])
print("query distance table (rows = chunk, cols = centroid id):\n", table)
adc = np.array([sum(table[m, idx[i, m]] for m in range(M)) for i in range(len(X))])
Dp, Ip = pq.search(Q, 8)
fa_adc = dict(zip(Ip[0].tolist(), Dp[0].tolist()))
assert all(abs(adc[i] - fa_adc[i]) < 1e-4 for i in range(len(X))), "hand ADC disagrees with FAISS search"
print("ADC distance = sum of M lookups; matches FAISS search() exactly:")
print({n: round(float(a), 3) for n, a in zip(NAMES, adc)})
print("true squared L2:", {n: round(float(v), 3) for n, v in zip(NAMES, ((X - Q) ** 2).sum(1))})
print("PQ ranking   :", [NAMES[i] for i in np.argsort(adc)])
print("exact ranking:", [NAMES[i] for i in exact])
print(f"bytes: {4 * d} B -> {M * NBITS / 8} B (here); real: 6,144 B -> 96 B (m = 96, 8 bits each)")
print("implied codebook size: (2**nbits)**m =", (2 ** NBITS) ** M, "vectors from", M * 2 ** NBITS, "stored centroids")
