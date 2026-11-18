"""Build the M5 toy (8 one-dimensional vectors, M=2) with hnswlib and dump the
graph hnswlib actually builds, layer by layer.

hnswlib draws each element's top layer from its own seeded RNG, so we scan
random_seed until the draws reproduce the README's layers
(A0 B0 C1 D0 E2 F0 G1 H0), then read the neighbour lists out of the pickled state.
"""
import struct, sys
import numpy as np
import hnswlib

NAMES = list("ABCDEFGH")
VALUES = np.array([[1], [2], [4], [7], [8], [11], [14], [15]], dtype=np.float32)
WANT_LEVELS = [0, 0, 1, 0, 2, 0, 1, 0]
M = 2
EF_C = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def build(seed):
    idx = hnswlib.Index(space="l2", dim=1)
    idx.init_index(max_elements=8, M=M, ef_construction=EF_C, random_seed=seed)
    idx.set_num_threads(1)
    idx.add_items(VALUES, np.arange(8))
    return idx


def state_of(idx):
    st = idx.__getstate__()
    if isinstance(st, tuple):  # (space, dim, params...) in some versions
        for part in st:
            if isinstance(part, dict) and "data_level0" in part:
                return part
    return st


def levels_of(idx):
    st = state_of(idx)
    return list(st["element_levels"])


def dump_graph(idx):
    st = state_of(idx)
    n = st["cur_element_count"]
    sde = st["size_data_per_element"]
    slpe = st["size_links_per_element"]
    maxM0, maxM = st["max_M0"], st["max_M"]
    ext = st["label_lookup_external"]
    internal = st["label_lookup_internal"]
    lab = {int(i): int(e) for e, i in zip(ext, internal)}  # internal -> label
    d0 = st["data_level0"]
    ll = st["link_lists"]
    lv = list(st["element_levels"])

    edges = {}
    # layer 0: per element block, starts with unsigned short count then maxM0 uints
    for i in range(n):
        off = i * sde
        cnt = struct.unpack_from("H", d0, off)[0]
        nb = struct.unpack_from(f"{cnt}I", d0, off + 4)
        edges.setdefault(0, {})[NAMES[lab[i]]] = [NAMES[lab[j]] for j in nb]
    # upper layers: link_lists is the concatenation, in internal-id order, of each
    # element's buffer of size slpe * level; layer l lives at (l-1)*slpe
    pos = 0
    for i in range(n):
        if lv[i] == 0:
            continue
        for l in range(1, lv[i] + 1):
            off = pos + (l - 1) * slpe
            cnt = struct.unpack_from("H", ll, off)[0]
            nb = struct.unpack_from(f"{cnt}I", ll, off + 4)
            edges.setdefault(l, {})[NAMES[lab[i]]] = [NAMES[lab[j]] for j in nb]
        pos += slpe * lv[i]
    return edges, st


def main():
    for seed in range(1, 200_000):
        idx = build(seed)
        if levels_of(idx) == WANT_LEVELS:
            break
    else:
        sys.exit("no seed reproduced the README's levels")
    edges, st = dump_graph(idx)
    print(f"seed={seed}  M={st['M']}  maxM={st['max_M']}  maxM0={st['max_M0']}  "
          f"ef_construction={st['ef_construction']}  mult={st['mult']:.4f}  "
          f"max_level={st['max_level']}  entry={NAMES[st['enterpoint_node']]}")
    for l in sorted(edges, reverse=True):
        print(f"\nLayer {l}")
        for name in NAMES:
            if name in edges[l]:
                print(f"  {name} ({int(VALUES[NAMES.index(name)][0])}): -> {edges[l][name]}")
    # undirected edge sets for comparison with the README
    for l in sorted(edges, reverse=True):
        und = sorted({tuple(sorted((a, b))) for a, nbs in edges[l].items() for b in nbs})
        print(f"\nLayer {l} undirected edges ({len(und)}): {['-'.join(e) for e in und]}")
    # sanity: every edge symmetric?
    for l, adj in edges.items():
        for a, nbs in adj.items():
            for b in nbs:
                if a not in adj.get(b, []):
                    print(f"  asymmetric on layer {l}: {a}->{b} but not back")


if __name__ == "__main__":
    main()
