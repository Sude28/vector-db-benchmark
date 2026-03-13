import pandas as pd

# ===============================
# CONFIG
# ===============================

INPUT_CSV = "benchmark_summary.csv"
OUTPUT_TEX = "benchmark_table.tex"

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(INPUT_CSV)

# ===============================
# BUILD LATEX TABLE
# ===============================

latex = []
latex.append("\\begin{table}[ht]")
latex.append("\\centering")
latex.append("\\small")
latex.append("\\begin{tabular}{l l l c c c c}")
latex.append("\\hline")
latex.append(
    "Database & Index Type & Distance Metric & "
    "Avg Latency (ms) & P95 Latency (ms) & "
    "Peak RAM (MB) & Consensus@k \\\\"
)
latex.append("\\hline")

for _, row in df.iterrows():
    latex.append(
        f"{row['db'].upper()} & "
        f"Vector Index & "
        f"Cosine & "
        f"{row['avg_latency_ms']:.2f} & "
        f"{row['p95_latency_ms']:.2f} & "
        f"{row['peak_ram_mb']:.2f} & "
        f"{row['avg_consensus_at_k']:.2f} \\\\"
    )

latex.append("\\hline")
latex.append("\\end{tabular}")
latex.append("\\caption{Vector Database Benchmark Results}")
latex.append("\\label{tab:vector_db_benchmark}")
latex.append("\\end{table}")

# ===============================
# WRITE FILE
# ===============================

with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
    f.write("\n".join(latex))

print("✅ LaTeX tablo üretildi → benchmark_table.tex")
