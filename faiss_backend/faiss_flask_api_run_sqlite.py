from flask import Flask, request, jsonify
import faiss
import numpy as np
import sqlite3
import nltk
import unicodedata, re
from embedding.cpu_tinybert_embedder import TinyBertCPUEmbedder

nltk.download("punkt")

FAISS_INDEX_PATH = "faiss_backend/index.faiss"
SQLITE_PATH = "faiss_backend/meta.sqlite"

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_for_display(text: str) -> str:
    # çıktı daha temiz görünsün diye (istersen genişletiriz)
    text = clean_text(text)
    return text.strip('"\'')

print("📦 FAISS index loading...")
index = faiss.read_index(FAISS_INDEX_PATH)
print("✅ FAISS index yüklendi")

# embedder
embedder = TinyBertCPUEmbedder(
    model_path="embedding/tinybert_model",
    vocab_path="embedding/vocab.txt",
    max_len=64,
    l2_normalize=True,
    device="cpu"
)

app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "query gerekli"}), 400

    query = clean_text(query)

    # lexical filtre ifadesi (Milvus'taki expr benzeri)
    q_tokens = nltk.word_tokenize(query.lower())
    like_tokens = [t for t in q_tokens if len(t) > 2]
    if like_tokens:
        lexical_filter = " or ".join([f'text like "%{t}%"' for t in like_tokens])
    else:
        lexical_filter = None

    # 1) FAISS arama (önce geniş al)
    q_vec = embedder.encode_text(query).astype("float32").reshape(1, -1)
    faiss.normalize_L2(q_vec)

    # lexical sonrası eleme için biraz büyük çekiyoruz
    overfetch = max(top_k * 20, 200)
    scores, ids = index.search(q_vec, overfetch)

    # 2) SQLite ile lexical eleme + response
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.cursor()

    results = []
    for score, faiss_id in zip(scores[0], ids[0]):
        if faiss_id < 0:
            continue

        cur.execute("SELECT sentence_id, ngram_size, text FROM meta WHERE faiss_id=?", (int(faiss_id),))
        row = cur.fetchone()
        if not row:
            continue

        sentence_id, ngram_size, raw_text = row

        # lexical filter kontrolü (SQL like'ı burada uyguluyoruz)
        if like_tokens:
            raw_lower = raw_text.lower()
            if not any(t in raw_lower for t in like_tokens):
                continue

        results.append({
            "ngram_size": int(ngram_size),
            "score": round(float(score),4),
            "sentence_id": int(sentence_id),
            "raw_text": raw_text,
            "text": clean_for_display(raw_text)
        })

        if len(results) >= top_k:
            break

    con.close()

    return jsonify({
        "query": query,
        "lexical_filter": lexical_filter if lexical_filter else "",
        "results": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8601, debug=False)
