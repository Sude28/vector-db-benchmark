# qdrant_backend/qdrant_flask_api_run.py

from flask import Flask, request, jsonify
import numpy as np
import nltk
import re, unicodedata
from embedding.cpu_tinybert_embedder import TinyBertCPUEmbedder

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText

nltk.download("punkt")

# -------------------------
# CLEANING
# -------------------------
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_for_display(text: str) -> str:
    if not text:
        return ""
    text = text.replace('"', "").replace("'", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------
# QDRANT
# -------------------------
client = QdrantClient(
    url="http://localhost:6333"
)
print("✅ Qdrant Docker client hazır")


COLLECTION_NAME = "kased_collection_v4"

print("✅ Qdrant client hazır")

# -------------------------
# EMBEDDER
# -------------------------
embedder = TinyBertCPUEmbedder(
    model_path="embedding/tinybert_model",
    vocab_path="embedding/vocab.txt",
    max_len=64,
    l2_normalize=True,
    device="cpu"
)

# -------------------------
# FLASK
# -------------------------
app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "query gerekli"}), 400

    query = clean_text(query)

    # -------------------------
    # LEXICAL FILTER
    # -------------------------
    tokens = nltk.word_tokenize(query.lower())
    must_conditions = [
        FieldCondition(
            key="text",
            match=MatchText(text=t)
        )
        for t in tokens if len(t) > 2
    ]

    q_filter = Filter(must=must_conditions) if must_conditions else None

    # -------------------------
    # EMBEDDING
    # -------------------------
    q_vec = embedder.encode_text(query).astype("float32")
    q_vec /= np.linalg.norm(q_vec) + 1e-12

    # -------------------------
    # SEARCH
    # -------------------------
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vec.tolist(),
        limit=top_k,
        query_filter=q_filter
    )


    results_map = {} 
    for hit in hits:
        payload = hit.payload
        raw_text = payload.get("text", "")

        sentence_id = payload.get("sentence_id")

        # ❌ sentence olmayanları alma
        if sentence_id is None:
            continue

        score = round(hit.score, 4)

        # Aynı cümle geldiyse en yüksek score'u tut
        if sentence_id not in results_map or score > results_map[sentence_id]["score"]:
            results_map[sentence_id] = {
                "text": clean_for_display(raw_text),
                #"raw_text": raw_text,
                "sentence_id": sentence_id,
                "score": score
            }
    results = list(results_map.values())
    
    #NGRAM SİZE DA KOYABİLİRİZ İSTERSEN, RAW TEXT DE EKLEYEBİLİRİZ İSTERSEN

    #results = []
    #for hit in hits:
        #payload = hit.payload
        #raw_text = payload.get("text", "")

        #results.append({
            #"text": clean_for_display(raw_text),
            #"raw_text": raw_text,
            #"sentence_id": payload.get("sentence_id"),
            #"ngram_size": payload.get("ngram_size"),
            #"score": round(hit.score, 4)
        #})

    return jsonify({
        "query": query,
        "lexical_filter": [t for t in tokens if len(t) > 2],
        "results": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8602, debug=False)
