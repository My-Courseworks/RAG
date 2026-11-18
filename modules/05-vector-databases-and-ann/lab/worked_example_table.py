"""Print the M5 worked-example tables from supplementary/worked-example/results.json (no measurement here).

Usage: python -I worked_example_table.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worked_example_common import RESULTS  # noqa: E402

R = json.loads(RESULTS.read_text())
MB = 1e6


def mb(b):
    return f"{b / MB:,.0f} MB" if b >= 10 * MB else f"{b / MB:,.1f} MB"


def qmem(phase):
    m = R[phase]["memory_after_queries"]
    p = m["proc_status_bytes"]
    sm = m["smaps_rss_bytes_by_mapping"]
    vec = sm.get("vector_storage/matrix.dat", 0)
    quant = sum(v for k, v in sm.items() if "quantized" in k or "quant" in k)
    return p["VmRSS"], p["RssAnon"], vec, quant, m["docker_stats_mem_usage"].split(" / ")[0]


def qrow(name, phase, key, notes=""):
    s = R[phase]["searches"][key]
    rss, anon, vec, quant, dstats = qmem(phase)
    ar = R[phase]["arithmetic"]
    links = R[phase]["memory_after_queries"]["storage_bytes_by_file"].get("vector_index/links_compressed.bin", 0)
    b = R[phase]["build"]
    print(f"| {name} | Qdrant {R[phase]['image'].split(':')[1]} | {s['recall_at_10']:.4f} | {s['p50_ms']:.2f} | {s['p99_ms']:.2f} "
          f"| RSS {mb(rss)} (anon {mb(anon)}, vectors mmap {mb(vec)}{', codes ' + mb(quant) if quant else ''}); docker stats {dstats}; links file {mb(links)} "
          f"| {mb(ar['total_bytes'])} (vectors {mb(ar['vectors_bytes'])} + links {mb(ar['links_bytes'])}{' + codes ' + mb(ar['codes_bytes']) if ar['codes_bytes'] else ''}) "
          f"| {b['index_build_s']} s (+{b['upload_s']} s upload) | {notes} |")


def main():
    c = R["corpus"]
    print(f"corpus: {c['dataset']} @ {c['revision'][:12]}, rows {c['corpus_rows']}, queries {c['query_rows']}")
    print("\n| Config | Engine | Recall@10 | p50 ms | p99 ms | Resident memory (measured) | Memory by arithmetic | Build | Notes |")
    print("|---|---|---|---|---|---|---|---|---|")
    if "faiss_ivfpq" in R:
        f = R["faiss_ivfpq"]["faiss_flat_exact"]
        print(f"| 1 Exact | FAISS IndexFlatIP (in process) | {f['recall_at_10']:.4f} | {f['p50_ms']:.2f} | {f['p99_ms']:.2f} | {mb(f['bytes'])} array | {mb(f['bytes'])} | 0 | ground truth is numpy; FAISS ties |")
    if "qdrant_m16" in R:
        qrow("1 Exact", "qdrant_m16", "exact", "exact=True")
        qrow("2 HNSW m16 ef64", "qdrant_m16", "ef64")
        qrow("3 HNSW m16 ef256", "qdrant_m16", "ef256")
    if "qdrant_m48" in R:
        qrow("4 HNSW m48 ef256", "qdrant_m48", "ef256")
        qrow("4b HNSW m48 ef64", "qdrant_m48", "ef64")
    if "qdrant_int8" in R:
        qrow("5 HNSW + int8, os 2, rescore on", "qdrant_int8", "ef64_os2_rescoreOn", "ef=64; originals on disk")
    if "qdrant_binary" in R:
        qrow("6 HNSW + binary, os 2, rescore on", "qdrant_binary", "ef64_os2_rescoreOn", "ef=64; originals on disk")
    if "faiss_ivfpq" in R:
        f = R["faiss_ivfpq"]
        s = f["searches"]["nprobe32"]
        a = f["arithmetic"]
        print(f"| 7 IVF-PQ nlist4096 nprobe32 PQ96x8 | FAISS {f['faiss']} (in process) | {s['recall_at_10']:.4f} | {s['p50_ms']:.2f} | {s['p99_ms']:.2f} "
              f"| serialized {mb(f['serialized_index_bytes'])}; RSS delta of build {mb(f['rss_delta_build_bytes'])} | {mb(a['total_bytes'])} "
              f"(codes {mb(a['codes_bytes'])} + ids {mb(a['ids_bytes'])} + centroids {mb(a['coarse_centroids_bytes'])}) | {f['build_s']} s | no RPC |")
    if "pgvector" in R:
        p = R["pgvector"]
        for key in ("sim_postfilter_ef40", "band_postfilter_ef40"):
            s = p["searches"][key]
            print(f"| 8 post-filter {key.split('_')[0]} | pgvector {p['pgvector']} | {s['recall_at_10']:.4f} | {s['p50_ms']:.2f} | {s['p99_ms']:.2f} "
                  f"| HNSW index {mb(p['sizes']['hnsw_index_bytes'])} on disk | {mb(p['arithmetic']['vectors_bytes'] + p['arithmetic']['hnsw_layer0_links_bytes'])} "
                  f"| {p['hnsw_build_s']} s | rows returned {s['mean_rows_returned']} |")

    for ph in ("qdrant_int8", "qdrant_binary"):
        if ph not in R:
            continue
        print(f"\n{ph} sweep (recall@10 / p50 ms / p99 ms):")
        print("| ef | oversampling | rescore off | rescore on |\n|---|---|---|---|")
        S = R[ph]["searches"]
        for ef in (64, 256):
            for os_ in ("1", "2", "4"):
                on, off = S[f"ef{ef}_os{os_}_rescoreOn"], S[f"ef{ef}_os{os_}_rescoreOff"]
                print(f"| {ef} | {os_} | {off['recall_at_10']:.4f} / {off['p50_ms']:.2f} / {off['p99_ms']:.2f} "
                      f"| {on['recall_at_10']:.4f} / {on['p50_ms']:.2f} / {on['p99_ms']:.2f} |")

    if "qdrant_m16" in R and "filters" in R["qdrant_m16"]:
        print("\nQdrant filters:")
        for k, v in R["qdrant_m16"]["filters"].items():
            if isinstance(v, dict) and "recall_at_10" in v:
                print(f"| {k} | {v['recall_at_10']:.4f} | {v['p50_ms']:.2f} | {v['p99_ms']:.2f} |")
    if "pgvector" in R:
        print("\npgvector:")
        for k, v in R["pgvector"]["searches"].items():
            print(f"| {k} | {v['recall_at_10']:.4f} | {v['mean_rows_returned']} | {v['queries_with_zero_rows']} | {v['p50_ms']:.2f} | {v['p99_ms']:.2f} |")
    if "faiss_ivfpq" in R:
        print("\nFAISS IVF-PQ:")
        for k, v in R["faiss_ivfpq"]["searches"].items():
            print(f"| {k} | {v['recall_at_10']:.4f} | {v['p50_ms']:.2f} | {v['p99_ms']:.2f} | {v.get('batch_qps_1000', '')} |")
    if "milvus_lite_ivfpq_check" in R:
        m = R["milvus_lite_ivfpq_check"]
        print("\nMilvus Lite check:", m["verdict"], {k: v["recall_at_10"] for k, v in m["searches"].items()})


if __name__ == "__main__":
    main()
