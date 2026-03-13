import csv
import statistics
from collections import defaultdict

# ===============================
# CONFIG
# ===============================

INPUT_CSV  = "benchmark_results.csv"
OUTPUT_CSV = "benchmark_summary.csv"

DBS = ["faiss", "milvus", "qdrant"]

# ===============================
# LOAD DATA
# ===============================

rows = []

with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["latency_ms"] = float(r["latency_ms"])
        r["ram_mb"] = float(r["ram_mb"])
        r["result_count"] = int(r["result_count"])
        rows.append(r)

print(f"📥 {len(rows)} satır yüklendi")

# ===============================
# PER-DB STATS
# ===============================

stats = {}

for db in DBS:
    db_rows = [r for r in rows if r["db"] == db]

    latencies = [r["latency_ms"] for r in db_rows]
    rams      = [r["ram_mb"] for r in db_rows]
    counts    = [r["result_count"] for r in db_rows]

    stats[db] = {
        "avg_latency_ms": round(statistics.mean(latencies), 2),
        "median_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2),
        "peak_ram_mb": round(max(rams), 2),
        "avg_result_count": round(statistics.mean(counts), 2),
        "num_queries": len(db_rows)
    }

# ===============================
# OVERLAP & CONSENSUS
# ===============================

overlaps = defaultdict(list)
consensus_vals = []

with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    grouped = defaultdict(dict)

    for r in reader:
        grouped[r["query"]][r["db"]] = int(r["result_count"])

    for q, per_db in grouped.items():
        if len(per_db) == 3:
            overlaps["faiss_milvus"].append(min(per_db["faiss"], per_db["milvus"]))
            overlaps["faiss_qdrant"].append(min(per_db["faiss"], per_db["qdrant"]))
            overlaps["milvus_qdrant"].append(min(per_db["milvus"], per_db["qdrant"]))
            consensus_vals.append(min(per_db.values()))

overlap_stats = {
    k: round(statistics.mean(v), 2)
    for k, v in overlaps.items()
}

consensus_avg = round(statistics.mean(consensus_vals), 2)

# ===============================
# WRITE SUMMARY CSV
# ===============================

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "db",
        "avg_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "peak_ram_mb",
        "avg_result_count",
        "num_queries",
        "avg_overlap_faiss_milvus",
        "avg_overlap_faiss_qdrant",
        "avg_overlap_milvus_qdrant",
        "avg_consensus_at_k"
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for db in DBS:
        writer.writerow({
            "db": db,
            **stats[db],
            "avg_overlap_faiss_milvus": overlap_stats.get("faiss_milvus", ""),
            "avg_overlap_faiss_qdrant": overlap_stats.get("faiss_qdrant", ""),
            "avg_overlap_milvus_qdrant": overlap_stats.get("milvus_qdrant", ""),
            "avg_consensus_at_k": consensus_avg
        })

print(f"✅ Analiz tamamlandı → {OUTPUT_CSV}")
