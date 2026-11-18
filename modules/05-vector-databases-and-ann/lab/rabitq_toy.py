"""RaBitQ, derived by hand and checked against FAISS 1.16 (IndexRaBitQ, 1-bit, unquantised query).

Sections (each prints what the README's RaBitQ block quotes):
  1. toy      : the 8 x 4-d vectors of quantization_toy.py, one rotation, hand-derived codes, factors
                and distance estimates, asserted equal to FAISS's stored bytes and search() output.
  2. dimension: how the estimator's error shrinks with D, against the same estimator without its
                correction factor (plain sign bits). Hand estimator only; it is first asserted equal
                to FAISS at D = 256.
  3. --scale  : the shared 187,043 x 256 generator: flat 1-bit scan with oversample + rescore, then the
                same code paired with IVF (a centroid per list).

Run: python -I rabitq_toy.py [--scale]      (numpy, faiss-cpu)
"""
import sys
import numpy as np
import faiss

np.set_printoptions(precision=3, suppress=True, linewidth=130)


def hr(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}")


def build(x, d):
    """IndexPreTransform(RandomRotation(d), IndexRaBitQ): FAISS expects the rotation to be external."""
    ix = faiss.index_factory(d, f"RR{d},RaBitQ")
    ix.train(x)
    ix.add(x)
    vt = faiss.downcast_VectorTransform(ix.chain.at(0))
    A = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
    b = faiss.vector_to_array(vt.b) if vt.have_bias else np.zeros(d, np.float32)
    rq = faiss.downcast_index(ix.index)
    center = faiss.vector_to_array(rq.center)
    cs = (d + 7) // 8 + 8
    raw = faiss.vector_to_array(rq.codes).reshape(len(x), cs)
    return ix, A, b, center, raw


def hand_encode(xr, c):
    """sign bits and the two stored floats, from FAISS's compute_factors_from_intermediates."""
    d = xr.shape[1]
    r = xr - c                                   # residual against the centroid
    bits = (r > 0)                               # 1 bit per dimension
    norm = np.linalg.norm(r, axis=1)             # ||x - c||
    ip_ob = np.abs(r).sum(1) / (norm * np.sqrt(d))   # <o_bar, o>: how well the signs capture the direction
    return bits, norm ** 2, norm / ip_ob, ip_ob  # code, or_minus_c_l2sqr, dp_multiplier, <o_bar,o>


def hand_distance(qr, c, bits, norm2, dpm):
    """||x-c||^2 + ||q-c||^2 - 2 * dp_multiplier * <o_bar, q-c>,  o_bar = (2*bit - 1) / sqrt(d)"""
    d = len(qr)
    qc = qr - c
    dot = ((2.0 * bits - 1.0) @ qc) / np.sqrt(d)
    return norm2 + qc @ qc - 2.0 * dpm * dot


# ------------------------------------------------------------------ 1. toy
hr("1. TOY: 8 vectors, d = 4, 1 bit per dimension")
X = np.array([[1.0, 2.0, -1.0, 0.5], [1.2, 1.8, -0.8, 0.7], [4.0, 0.5, 2.0, -1.5], [4.2, 0.3, 1.8, -1.7],
              [-2.0, -1.0, 3.0, 2.5], [-2.2, -0.8, 3.2, 2.3], [0.0, 3.5, -3.0, -2.5], [0.2, 3.3, -2.8, -2.7]], np.float32)
Q = np.array([[1.1, 1.9, -0.9, 0.6]], np.float32)
NAMES = [f"v{i}" for i in range(1, 9)]  # not the HNSW toy's 1-d A..H
d = 4
ix, A, bias, center, raw = build(X, d)
print("rotation matrix A (random orthogonal; A @ A.T = I):\n", A, "\nmax |A A^T - I| =", np.abs(A @ A.T - np.eye(d)).max())
XR = (X @ A.T + bias).astype(np.float32)
assert np.allclose(center, XR.mean(0), atol=1e-5), "FAISS centroid is not the mean of the rotated training vectors"
print("centroid c = mean of the rotated vectors:", center)
bits, norm2, dpm, ip_ob = hand_encode(XR, center)
fa_bits = np.unpackbits(raw[:, :1], axis=1, bitorder="little")[:, :d].astype(bool)
fa_fac = raw[:, 1:].copy().view(np.float32).reshape(len(X), 2)
assert np.array_equal(bits, fa_bits), "hand sign bits disagree with FAISS codes"
assert np.allclose(fa_fac[:, 0], norm2, rtol=1e-4) and np.allclose(fa_fac[:, 1], dpm, rtol=1e-4), "factors disagree"
print("hand sign bits and both stored floats equal FAISS's stored bytes (asserted).")
print(f"code size: {raw.shape[1]} B per vector = {(d + 7) // 8} B of bits + 8 B of two float32 factors")
print(f"{'':2}{'rotated residual x-c':<32}{'bits':<10}{'||x-c||^2':>10}{'<o_bar,o>':>11}{'dp_mult':>9}")
for n, r, b, n2, io, dp in zip(NAMES, XR - center, bits, norm2, ip_ob, dpm):
    print(f"{n:2}{str(np.round(r, 2)):<32}{''.join(str(int(v)) for v in b):<10}{n2:>10.3f}{io:>11.3f}{dp:>9.3f}")

qr = (Q @ A.T + bias).astype(np.float32)[0]
est = np.array([hand_distance(qr, center, bits[i], norm2[i], dpm[i]) for i in range(len(X))])
rq_inner = faiss.downcast_index(ix.index)
print("FAISS default query bits qb =", rq_inner.qb, "(the query is itself quantised to 4 bits; qb = 0 keeps it float)")
rq_inner.qb = 0
Dfa, Ifa = ix.search(Q, len(X))
fa = dict(zip(Ifa[0].tolist(), Dfa[0].tolist()))
rq_inner.qb = 4
D4, I4 = ix.search(Q, len(X))
fa4 = dict(zip(I4[0].tolist(), D4[0].tolist()))
# FAISS clamps negative squared-distance estimates to 0, so compare after the same clamp
assert all(abs(max(est[i], 0.0) - fa[i]) < 1e-3 for i in range(len(X))), "hand estimate disagrees with FAISS search()"
true = ((X - Q) ** 2).sum(1)
print("\nsquared distance from Q: hand estimate == FAISS search() at qb=0 (asserted, negatives clamped to 0)")
print(f"{'':2}{'estimate':>10}{'FAISS qb=0':>11}{'FAISS qb=4':>11}{'true L2^2':>11}{'error':>9}")
for i, (n, e, t) in enumerate(zip(NAMES, est, true)):
    print(f"{n:2}{e:>10.3f}{fa[i]:>11.3f}{fa4[i]:>11.3f}{t:>11.3f}{e - t:>9.3f}")
print("A's raw estimate is negative (a squared distance cannot be); FAISS clamps it to 0.")
print("at d = 4 the errors are large (the bound shrinks like 1/sqrt(D)); section 2 measures that.")
print("estimate ranking:", [NAMES[i] for i in np.argsort(est)])
print("exact ranking   :", [NAMES[i] for i in np.argsort(true)])
print("(FAISS seeds its rotation, so these numbers repeat run to run)")

# ------------------------------------------------------------------ 2. dimension
hr("2. ERROR VERSUS DIMENSION (hand estimator, random unit-scale data)")
rng = np.random.default_rng(0)


def rotation(D):
    q, r = np.linalg.qr(rng.standard_normal((D, D)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


# first: the hand estimator equals FAISS at D = 256 on random data
D0 = 256
Xr = rng.standard_normal((2000, D0)).astype(np.float32)
Qr = rng.standard_normal((5, D0)).astype(np.float32)
ix2, A2, b2, c2, raw2 = build(Xr, D0)
faiss.downcast_index(ix2.index).qb = 0   # float query, to compare with the hand estimator
XR2 = (Xr @ A2.T + b2).astype(np.float32)
bt, n2_, dp_, _ = hand_encode(XR2, c2)
D_fa, I_fa = ix2.search(Qr, 2000)
worst = 0.0
for qi in range(len(Qr)):
    qrr = (Qr[qi:qi + 1] @ A2.T + b2).astype(np.float32)[0]
    h = np.array([hand_distance(qrr, c2, bt[i], n2_[i], dp_[i]) for i in range(2000)])
    f = np.empty(2000)
    f[I_fa[qi]] = D_fa[qi]
    worst = max(worst, np.abs(h - f).max() / np.abs(f).mean())
print(f"hand estimator vs FAISS at D=256: worst relative difference {worst:.2e}")
assert worst < 1e-3

def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


print("\n(a) error scale: random vector pairs, true cosine near 0")
print(f"{'D':>6}{'mean |error|':>14}{'x sqrt(D)':>11}")
for D in (16, 64, 256, 1024):
    n = 3000
    base = rng.standard_normal((n, D)).astype(np.float32)
    qs = rng.standard_normal((200, D)).astype(np.float32)
    R = rotation(D)
    c = base.mean(0)
    o = unit((base - c) @ R.T)
    qh = unit((qs - c) @ R.T)
    obar = np.where(o > 0, 1.0, -1.0) / np.sqrt(D)
    ip_ob = (obar * o).sum(1)
    e = np.abs((qh @ obar.T) / ip_ob[None, :] - qh @ o.T).mean()
    print(f"{D:>6}{e:>14.4f}{e * np.sqrt(D):>11.3f}")
print("error x sqrt(D) is flat: the error of the cosine estimate shrinks like 1/sqrt(D).")

print("\n(b) what the correction buys: a NEAR neighbour, true cosine exactly 0.8 to its data vector")
print(f"{'D':>6}{'plain sign bits':>17}{'RaBitQ (divided by <o_bar,o>)':>31}{'  true':>7}")
for D in (64, 256, 1024):
    n = 2000
    base = rng.standard_normal((n, D)).astype(np.float32)
    R = rotation(D)
    o = unit((base - base.mean(0)) @ R.T)
    obar = np.where(o > 0, 1.0, -1.0) / np.sqrt(D)
    ip_ob = (obar * o).sum(1)
    u = rng.standard_normal((n, D)).astype(np.float32)
    u = unit(u - (u * o).sum(1, keepdims=True) * o)      # exactly orthogonal to o
    qh = 0.8 * o + 0.6 * u                                # unit vector, cosine 0.8 with o
    raw = (qh * obar).sum(1)
    print(f"{D:>6}{raw.mean():>17.3f}{(raw / ip_ob).mean():>31.3f}{0.8:>7.1f}")
print("plain sign bits read 0.8 as ~0.64 (a 0.8x shrink, = mean <o_bar,o>); dividing by <o_bar,o> removes it.")

print("\n(c) why the rotation exists: 3,000 vectors, each one large spike on a random axis, D = 1024")
D = 1024
spikes = np.zeros((3000, D), np.float32)
spikes[np.arange(3000), rng.integers(0, D, 3000)] = 5.0
spikes += 0.05 * rng.standard_normal((3000, D)).astype(np.float32)
c = spikes.mean(0)
for label, R in (("no rotation", np.eye(D, dtype=np.float32)), ("random rotation", rotation(D))):
    r = (spikes - c) @ R.T
    _, _, dpm_, ip_ = hand_encode(r, np.zeros(D, np.float32))
    print(f"{label:>16}: mean <o_bar,o> = {ip_.mean():.3f}")
print("<o_bar,o> is how much of a vector's direction its sign bits capture (1.0 = all of it). The estimator divides by it,")
print("so a small value inflates the noise; a random rotation pushes it to sqrt(2/pi) = 0.798 whatever the data looks like.")

# ------------------------------------------------------------------ 3. scale
if "--scale" in sys.argv:
    hr("3. SCALE: 187,043 x 256, 50-topic mixture (the module's shared generator), recall@10")
    N, DD, NQ, K = 187_043, 256, 1000, 10
    g = np.random.default_rng(0)
    centres = g.standard_normal((50, DD), dtype=np.float32)
    topic = g.integers(0, 50, N + NQ)
    x = centres[topic] + 0.35 * g.standard_normal((N + NQ, DD), dtype=np.float32)
    xb, xq = x[:N], x[N:]
    flat = faiss.IndexFlatL2(DD)
    flat.add(xb)
    _, gt = flat.search(xq, K)
    ixs = faiss.index_factory(DD, f"RR{DD},RaBitQ")
    ixs.train(xb)
    ixs.add(xb)
    print(f"code size {(DD + 7) // 8 + 8} B vs {4 * DD} B raw ({4 * DD / ((DD + 7) // 8 + 8):.1f}x); "
          f"1-bit part alone is {(DD + 7) // 8} B ({4 * DD / ((DD + 7) // 8):.0f}x)")
    print(f"{'oversample':>11}{'candidates':>11}{'recall@10':>11}")
    for over in (1, 2, 4, 10, 20, 50, 100):
        _, cand = ixs.search(xq, K * over)
        if over == 1:
            r = np.mean([len(set(cand[i]) & set(gt[i])) / K for i in range(NQ)])
        else:  # rescore the candidates with the full vectors
            r = 0.0
            for i in range(NQ):
                cd = cand[i]
                dd = ((xb[cd] - xq[i]) ** 2).sum(1)
                top = cd[np.argsort(dd)[:K]]
                r += len(set(top) & set(gt[i])) / K
            r /= NQ
        print(f"{over:>10}x{K * over:>11}{r:>11.3f}")

    # the same code paired with IVF: each list has its own centroid, so the residual x - c keeps the
    # within-topic signal that one global centroid swamps with topic identity
    NL = 1024
    ixi = faiss.index_factory(DD, f"RR{DD},IVF{NL},RaBitQ")
    ixi.train(xb)
    ixi.add(xb)
    ivf = faiss.extract_index_ivf(ixi)
    print(f"\nIVF{NL} + RaBitQ (centroid per list), nprobe sweep; recall@10 raw 1-bit order, then 10x oversample + rescore")
    print(f"{'nprobe':>7}{'raw':>8}{'10x+rescore':>13}")
    for npb in (8, 16, 32, 64):
        ivf.nprobe = npb
        _, c1 = ixi.search(xq, K)
        raw_r = np.mean([len(set(c1[i]) & set(gt[i])) / K for i in range(NQ)])
        _, c10 = ixi.search(xq, K * 10)
        rr = 0.0
        for i in range(NQ):
            cd = c10[i][c10[i] >= 0]
            dd = ((xb[cd] - xq[i]) ** 2).sum(1)
            rr += len(set(cd[np.argsort(dd)[:K]]) & set(gt[i])) / K
        print(f"{npb:>7}{raw_r:>8.3f}{rr / NQ:>13.3f}")
