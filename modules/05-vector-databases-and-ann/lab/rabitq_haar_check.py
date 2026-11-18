"""Is the rotation uniform? For a uniform rotation, every entry is symmetric around 0 (mean 0)."""
import numpy as np, faiss
D, T = 8, 4000
rng = np.random.default_rng(0)
raw, fixed, fa = [], [], []
for s in range(T):
    q, r = np.linalg.qr(rng.standard_normal((D, D)))
    raw.append(np.diag(q)); fixed.append(np.diag(q * np.sign(np.diag(r))))
    rr = faiss.RandomRotationMatrix(D, D); rr.init(s + 1)
    fa.append(np.diag(faiss.vector_to_array(rr.A).reshape(D, D)))
for name, v in (("numpy QR, no sign fix", raw), ("numpy QR, sign fix", fixed), ("FAISS RandomRotationMatrix", fa)):
    v = np.array(v)
    print(f"{name:<28} mean of diagonal entries {v.mean():+.3f}   (uniform: 0, spread +/- {1/np.sqrt(D*T*D):.3f})")
