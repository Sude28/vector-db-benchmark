# benchmark_runner.py (FINAL)

import time
import requests
import csv
import statistics
import psutil
from itertools import combinations
from collections import defaultdict

# ===============================
# CONFIG
# ===============================

FAISS_URL   = "http://localhost:8601/search"
#MILVUS_URL  = "http://localhost:8600/search"
#QDRANT_URL  = "http://localhost:8602/search"

TOP_K = 5
QUERY_FILE = "benchmark_queries.txt"

DETAIL_CSV  = "benchmark_results_detail.csv"
SUMMARY_CSV = "benchmark_results_summary.csv"

DBS = {
    "faiss": FAISS_URL,
    #"milvus": MILVUS_URL,
    #"qdrant": QDRANT_URL
}

process = psutil.Process()

# ===============================
# HELPERS
# ===============================

def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return [q.strip() for q in f if q.strip()]

def run_query(db_name, url, query):
    payload = {"query": query, "top_k": TOP_K}

    mem_before = process.memory_info().rss / 1024 / 1024
    t0 = time.time()

    r = requests.post(url, json=payload, timeout=300)
    t1 = time.time()

    mem_after = process.memory_info().rss / 1024 / 1024

    data = r.json()
    results = data.get("results", [])

    sentence_ids = [
        res.get("sentence_id")
        for res in results
        if res.get("sentence_id") is not None
    ]

    return {
        "latency_ms": (t1 - t0) * 1000,
        "ram_mb": mem_after - mem_before,
        "ids": set(sentence_ids),
        "result_count": len(sentence_ids)
    }

def overlap(a, b):
    return len(a & b)

# ===============================
# MAIN
# ===============================

def main():
    queries = load_queries(QUERY_FILE)
    print(f"🔎 {len(queries)} query yüklendi")

    detail_rows = []
    summary_rows = []

    for q in queries:
        print(f"\n▶ Query: {q}")
        per_db = {}

        # -------- RUN QUERIES --------
        for db, url in DBS.items():
            res = run_query(db, url, q)
            per_db[db] = res
            
            time.sleep(0.2)

            detail_rows.append({
                "query": q,
                "db": db,
                "latency_ms": round(res["latency_ms"], 2),
                "ram_mb": round(res["ram_mb"], 2),
                "result_count": res["result_count"],
                "ids": "|".join(map(str, res["ids"]))
            })

        # -------- OVERLAP --------
        #overlaps = {}
        #for (db1, r1), (db2, r2) in combinations(per_db.items(), 2):
            #overlaps[f"{db1}_{db2}"] = overlap(r1["ids"], r2["ids"])

        # -------- CONSENSUS --------
        #consensus_ids = set.intersection(
            #per_db["faiss"]["ids"],
            #per_db["milvus"]["ids"],
            #per_db["qdrant"]["ids"]
        #)

        summary_rows.append({
            "query": q,
            "faiss_latency": per_db["faiss"]["latency_ms"],
            #"milvus_latency": per_db["milvus"]["latency_ms"],
            #"qdrant_latency": per_db["qdrant"]["latency_ms"],
            "faiss_ram": per_db["faiss"]["ram_mb"],
            #"milvus_ram": per_db["milvus"]["ram_mb"],
            #"qdrant_ram": per_db["qdrant"]["ram_mb"],
            #"faiss_milvus_overlap": overlaps["faiss_milvus"],
            #"faiss_qdrant_overlap": overlaps["faiss_qdrant"],
            #"milvus_qdrant_overlap": overlaps["milvus_qdrant"],
            #"consensus_at_k": len(consensus_ids)
        })

    # ===============================
    # WRITE DETAIL CSV
    # ===============================
    with open(DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "db", "latency_ms", "ram_mb", "result_count" , "ids"]
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    # ===============================
    # WRITE SUMMARY CSV
    # ===============================
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # ===============================
    # CONSOLE SUMMARY
    # ===============================
    print("\n📊 OVERALL SUMMARY\n")

    for db in DBS:
        latencies = [r["latency_ms"] for r in detail_rows if r["db"] == db]
        rams = [r["ram_mb"] for r in detail_rows if r["db"] == db]

        print(f"▶ {db.upper()}")
        print(f"  Avg latency: {statistics.mean(latencies):.2f} ms")
        print(f"  Median latency: {statistics.median(latencies):.2f} ms")
        print(f"  P95 latency: {statistics.quantiles(latencies, n=20)[18]:.2f} ms")
        print(f"  Peak RAM: {max(rams):.2f} MB\n")

    print("🎯 Benchmark tamamlandı.")
    print(f"📄 Detail CSV : {DETAIL_CSV}")
    print(f"📄 Summary CSV: {SUMMARY_CSV}")

# ===============================
if __name__ == "__main__":
    main()
