# milvus_flask_api_run.py
# -*- coding: utf-8 -*-

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from flask import Flask, request, jsonify
import numpy as np
import nltk
from pymilvus import connections, Collection
from embedding.cpu_tinybert_embedder import TinyBertCPUEmbedder

nltk.download("punkt")

import re
import unicodedata

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")   # NBSP
    text = text.replace("\u202f", " ")   # narrow no-break space
    text = text.replace("\u200b", "")    # zero-width space

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def clean_for_display(text: str) -> str:
    if not text:
        return ""

    # problemli karakterler
    text = text.replace("†", "")
    text = text.replace("\"", "")
    text = text.replace("'", "")

    # baştaki / sondaki tire vb.
    text = re.sub(r"^\W+", "", text)
    text = re.sub(r"\W+$", "", text)

    # fazla boşluk
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# -------------------------
# 1️⃣ MILVUS BAĞLANTI
# -------------------------
connections.connect(
    alias="default",
    host="127.0.0.1",
    port="19530"
)

COLLECTION_NAME = "kased_collection_v4"
collection = Collection(COLLECTION_NAME)
collection.load()

print("✅ Milvus collection yüklendi")

# -------------------------
# 2️⃣ QUERY EMBEDDER (CPU – Jetson uyumlu)
# -------------------------
embedder = TinyBertCPUEmbedder(
    model_path="embedding/tinybert_model",
    vocab_path="embedding/vocab.txt",
    max_len=64,
    l2_normalize=True,
    device="cpu"
)

# -------------------------
# 3️⃣ FLASK
# -------------------------
app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    query = clean_text(query)
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "query gerekli"}), 400

    # -------------------------
    # 4️⃣ LEXICAL DARALTMA (EN KRİTİK ADIM)
    # -------------------------
    tokens = nltk.word_tokenize(query.lower())

    # örn: note_ngram like "%ay%" OR "%ışığı%"
    like_filters = [f'text like "%{t}%"' for t in tokens if len(t) > 2]

    if like_filters:
        lexical_filter = " or ".join(like_filters)
    else:
        lexical_filter = None

    # -------------------------
    # 5️⃣ QUERY EMBEDDING
    # -------------------------
    q_vec = embedder.encode_text(query).astype("float32")
    q_vec /= np.linalg.norm(q_vec) + 1e-12

    # -------------------------
    # 6️⃣ SEMANTIC SEARCH (DARALTILMIŞ ALANDA)
    # -------------------------
    results = collection.search(
        data=[q_vec.tolist()],
        anns_field="embedding",
        param={"metric_type": "COSINE"},
        limit=top_k,
        expr=lexical_filter,
        output_fields=["text", "ngram_size", "sentence_id"]
    )

    # -------------------------
    # 7️⃣ FORMAT
    # -------------------------
    hits = []
    for hit in results[0]:
        raw_text = hit.entity.get("text")

        hits.append({
            "text": clean_for_display(raw_text),
            "raw_text": raw_text, 
            "sentence_id": hit.entity.get("sentence_id"),
            "ngram_size": hit.entity.get("ngram_size"),
            "score": round(hit.score, 4)
        })

    return jsonify({
        "query": query,
        "lexical_filter": lexical_filter,
        "results": hits
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8600, debug=False)












