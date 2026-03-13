import os
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================

INPUT_CSV = "benchmark_summary.csv"
PLOT_DIR = "plots2"

os.makedirs(PLOT_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(INPUT_CSV)

# ===============================
# 1️⃣ Latency vs RAM (Scatter)
# ===============================

plt.figure()
plt.scatter(df["avg_latency_ms"], df["peak_ram_mb"])

for _, row in df.iterrows():
    plt.text(
        row["avg_latency_ms"],
        row["peak_ram_mb"],
        row["db"].upper(),
        fontsize=9
    )

plt.xlabel("Average Latency (ms)")
plt.ylabel("Peak RAM Usage (MB)")
plt.title("Latency vs RAM Usage")
plt.grid(True)

plt.savefig(f"{PLOT_DIR}/latency_vs_ram.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================
# 2️⃣ Avg Latency Bar
# ===============================

plt.figure()
plt.bar(df["db"], df["avg_latency_ms"])
plt.ylabel("Average Latency (ms)")
plt.title("Average Latency Comparison")
plt.yscale("log")
plt.grid(axis="y")

plt.savefig(f"{PLOT_DIR}/avg_latency.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================
# 3️⃣ P95 Latency Bar
# ===============================

plt.figure()
plt.bar(df["db"], df["p95_latency_ms"])
plt.ylabel("P95 Latency (ms)")
plt.title("P95 Latency Comparison")
plt.yscale("log")
plt.grid(axis="y")

plt.savefig(f"{PLOT_DIR}/p95_latency.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================
# 4️⃣ Peak RAM Bar
# ===============================

plt.figure()
plt.bar(df["db"], df["peak_ram_mb"])
plt.ylabel("Peak RAM Usage (MB)")
plt.title("Peak RAM Usage Comparison")
plt.grid(axis="y")

plt.savefig(f"{PLOT_DIR}/peak_ram.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================
# 5️⃣ Overlap Bar
# ===============================

overlap_cols = [
    "avg_overlap_faiss_milvus",
    "avg_overlap_faiss_qdrant",
    "avg_overlap_milvus_qdrant"
]

overlap_vals = df.iloc[0][overlap_cols].values
labels = ["FAISS–Milvus", "FAISS–Qdrant", "Milvus–Qdrant"]

plt.figure()
plt.bar(labels, overlap_vals)
plt.ylabel("Average Overlap Count")
plt.title("Pairwise Result Overlap")
plt.grid(axis="y")

plt.savefig(f"{PLOT_DIR}/overlap.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================
# 6️⃣ Consensus@k
# ===============================

plt.figure()
plt.bar(["Consensus@k"], [df["avg_consensus_at_k"].iloc[0]])
plt.ylabel("Average Consensus Count")
plt.title("Consensus@k Across Databases")
plt.grid(axis="y")

plt.savefig(f"{PLOT_DIR}/consensus.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================
# 7️⃣ Speedup vs FAISS
# ===============================

faiss_latency = df[df["db"] == "faiss"]["avg_latency_ms"].values[0]

speedups = []

for _, row in df.iterrows():
    speedups.append(row["avg_latency_ms"] / faiss_latency)

plt.figure()
plt.bar(df["db"], speedups)

plt.ylabel("Latency Relative to FAISS (x)")
plt.title("Relative Latency Speedup")
plt.grid(axis="y")

plt.savefig(f"{PLOT_DIR}/speedup_vs_faiss.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Tüm grafikler üretildi → plots2/")
