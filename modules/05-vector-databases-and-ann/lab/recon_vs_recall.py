import numpy as np, faiss, importlib.util, sys
spec = importlib.util.spec_from_file_location("qc", "quant_compare.py"); qc = importlib.util.module_from_spec(spec)
src = open("quant_compare.py").read().split("which = sys.argv")[0]; exec(src, qc.__dict__)
xb, xq = qc.embedlike(); D = qc.D; K = 10
flat = faiss.IndexFlatL2(D); flat.add(xb); _, gt = flat.search(xq, K)
tr, ho = xb[:150_000], xb[150_000:]
def mse(ix): return float(((ix.sa_decode(ix.sa_encode(ho)) - ho) ** 2).sum(1).mean())
pq = faiss.IndexPQ(D, 32, 8); pq.train(tr); pq.add(xb)
rq = faiss.index_factory(D, f"RR{D},RaBitQ"); rq.train(tr); rq.add(xb); faiss.downcast_index(rq.index).qb = 0
for name, ix, b in (("PQ32x8", pq, 32), ("RaBitQ 1-bit (float query)", rq, 40)):
    _, I = ix.search(xq, K)
    print(f"{name:<28}{b:>4} B   reconstruction MSE {mse(ix):.4f}   recall@10 {qc.recall(I, gt):.3f}")
