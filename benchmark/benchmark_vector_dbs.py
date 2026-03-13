import time
import os
import json
import csv
import statistics
import requests
import psutil

# -----------------------------
# CONFIG
# -----------------------------
DB_ENDPOINTS = {
    "FAISS": "http://localhost:8008/search",
    "ChromaDB": "http://localhost:8502/search",
    "Milvus": "http://localhost:8600/search",
}

TOP_K = 5
REPEATS = 10          # her sorgu için tekrar sayısı
TIMEOUT_S = 120.0

# Query listesi (istersen dosyadan da okuyabilirsin)
QUERIES = [
    "python backend developer",
    "data scientist machine learning",
    "devops kubernetes docker",
    "frontend react typescript",
    "nlp bert embeddings",
    "postgresql database performance",
]

# Index/DB size paths
FAISS_INDEX_PATH = "faiss/faiss_index.bin"
CHROMA_DB_DIR = "chroma/chroma_db"
MILVUS_DATA_DIR = ""  # opsiyonel

OUT_JSON = "benchmark_results.json"
OUT_CSV = "benchmark_results.csv"

# -----------------------------
# UTIL
# -----------------------------
def system_ram_used_bytes() -> int:
    return int(psutil.virtual_memory().used)

def get_file_size_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return -1

def get_dir_size_bytes(path: str) -> int:
    if not path or not os.path.isdir(path):
        return -1
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total

def percentile(values, p):
    if not values:
        return None
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1

def post_search(url: str, query: str, top_k: int):
    payload = {"query": query, "top_k": int(top_k)}
    return requests.post(url, json=payload, timeout=TIMEOUT_S)

# -----------------------------
# BENCH
# -----------------------------
def main():
    sizes = {
        "faiss_index_bytes": get_file_size_bytes(FAISS_INDEX_PATH),
        "chroma_db_bytes": get_dir_size_bytes(CHROMA_DB_DIR),
        "milvus_data_bytes": get_dir_size_bytes(MILVUS_DATA_DIR) if MILVUS_DATA_DIR else -1
    }

    all_rows = []
    summary = {
        "config": {
            "top_k": TOP_K,
            "repeats": REPEATS,
            "timeout_s": TIMEOUT_S,
            "queries": QUERIES,
            "endpoints": DB_ENDPOINTS,
        },
        "sizes": sizes,
        "per_db": {}
    }

    for db_name, url in DB_ENDPOINTS.items():
        db_latencies = []
        db_errors = 0

        per_query_stats = []

        for q in QUERIES:
            latencies = []
            for r in range(REPEATS):
                ram_before = system_ram_used_bytes()
                t0 = time.perf_counter()
                ok = True
                err = ""

                try:
                    resp = post_search(url, q, TOP_K)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    if resp.status_code != 200:
                        ok = False
                        err = resp.text[:300]
                except Exception as e:
                    ok = False
                    latency_ms = None
                    err = str(e)

                ram_after = system_ram_used_bytes()
                ram_delta = ram_after - ram_before

                if ok and latency_ms is not None:
                    latencies.append(latency_ms)
                    db_latencies.append(latency_ms)
                else:
                    db_errors += 1

                all_rows.append({
                    "db": db_name,
                    "query": q,
                    "repeat": r,
                    "ok": ok,
                    "latency_ms": None if latency_ms is None else round(latency_ms, 3),
                    "system_ram_used_bytes": ram_after,
                    "system_ram_delta_bytes": ram_delta,
                    "error": err
                })

            # per query aggregate
            if latencies:
                per_query_stats.append({
                    "query": q,
                    "count": len(latencies),
                    "mean_ms": round(statistics.mean(latencies), 3),
                    "p95_ms": round(percentile(latencies, 95), 3),
                    "min_ms": round(min(latencies), 3),
                    "max_ms": round(max(latencies), 3),
                })
            else:
                per_query_stats.append({
                    "query": q,
                    "count": 0,
                    "mean_ms": None,
                    "p95_ms": None,
                    "min_ms": None,
                    "max_ms": None,
                })

        # db aggregate
        if db_latencies:
            summary["per_db"][db_name] = {
                "total_ok": len(db_latencies),
                "total_errors": db_errors,
                "mean_ms": round(statistics.mean(db_latencies), 3),
                "p95_ms": round(percentile(db_latencies, 95), 3),
                "min_ms": round(min(db_latencies), 3),
                "max_ms": round(max(db_latencies), 3),
                "per_query": per_query_stats
            }
        else:
            summary["per_db"][db_name] = {
                "total_ok": 0,
                "total_errors": db_errors,
                "mean_ms": None,
                "p95_ms": None,
                "min_ms": None,
                "max_ms": None,
                "per_query": per_query_stats
            }

    # Save JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": all_rows}, f, ensure_ascii=False, indent=2)

    # Save CSV
    fieldnames = list(all_rows[0].keys()) if all_rows else ["db","query","repeat","ok","latency_ms","system_ram_used_bytes","system_ram_delta_bytes","error"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    print("✅ Benchmark bitti.")
    print("JSON:", OUT_JSON)
    print("CSV :", OUT_CSV)
    print("Özet:", json.dumps(summary["per_db"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
