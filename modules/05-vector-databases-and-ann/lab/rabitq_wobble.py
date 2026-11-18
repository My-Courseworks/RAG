"""Fix a data direction o and a perpendicular query part e; draw many random rotations R.
Pull the bit vector back to the original space and check its wobble is a random direction perpendicular to o."""
import numpy as np
rng = np.random.default_rng(0); D, T = 64, 4000
o = rng.standard_normal(D); o /= np.linalg.norm(o)
e = rng.standard_normal(D); e -= (e @ o) * o; e /= np.linalg.norm(e)          # unit, perpendicular to o
dots, perp_len = [], []
for _ in range(T):
    R, _ = np.linalg.qr(rng.standard_normal((D, D)))
    ob = R.T @ (np.sign(R @ o) / np.sqrt(D))       # bit vector, back in the original space
    wob = ob - (ob @ o) * o                        # part perpendicular to o
    dots.append(wob @ e); perp_len.append(np.linalg.norm(wob))
dots, L = np.array(dots), np.mean(perp_len)
print(f"mean <wobble, e>                       {dots.mean():+.4f}   (0 if no preferred direction)")
print(f"std  <wobble, e>                       {dots.std():.4f}")
print(f"predicted |wobble| / sqrt(D - 1)       {L / np.sqrt(D - 1):.4f}")
ob0 = np.sign(o) / np.sqrt(D); w0 = ob0 - (ob0 @ o) * o
print(f"no rotation: <wobble, e> is one fixed  {w0 @ e:+.4f}, same every time")
