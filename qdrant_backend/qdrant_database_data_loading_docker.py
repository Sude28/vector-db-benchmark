# qdrant_backend/qdrant_database_data_loading.py

import json, re, unicodedata
import numpy as np
from tqdm import tqdm
import nltk

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

nltk.download("punkt")

EMBEDDINGS_PATH = "data/embeddings.npy"
TEXTS_PATH = "data/texts.jsonl"

COLLECTION_NAME = "kased_collection_v4"

DIM = 128
BATCH_SIZE = 1000
MAX_TEXT_LEN = 512


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_utf8_bytes(s: str, max_bytes: int) -> str:
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    return b[:max_bytes].decode("utf-8", errors="ignore")


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    return (vec / (np.linalg.norm(vec) + 1e-12)).astype(np.float32)


def main():
    print("🚀 Qdrant Docker data loading başlıyor")

    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    emb = np.memmap(
        EMBEDDINGS_PATH,
        dtype=np.float32,
        mode="r",
        shape=(total, DIM)
    )

    print("✅ embeddings yüklendi:", emb.shape)

    client = QdrantClient(
    url="http://localhost:6333",
    check_compatibility=False
)

    # ⚠️ collection_exists KULLANMA
    from qdrant_client.http.exceptions import UnexpectedResponse

    try:
        client.delete_collection(COLLECTION_NAME)
        print("🧹 Eski collection silindi")
    except UnexpectedResponse:
        print("ℹ️ Collection yok, devam")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=DIM,
            distance=Distance.COSINE
        )
    )

    points = []
    inserted = 0

    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f), total=total, desc="📥 Qdrant insert"):
            try:
                text = json.loads(line)
            except:
                continue

            if not isinstance(text, str):
                continue

            text = clean_text(text)
            if len(text) < 3:
                continue

            text = truncate_utf8_bytes(text, MAX_TEXT_LEN)

            tokens = nltk.word_tokenize(text.lower())
            if not tokens:
                continue

            vec = l2_normalize(np.array(emb[idx], dtype=np.float32))

            points.append(
                PointStruct(
                    id=idx,
                    vector=vec.tolist(),
                    payload={
                        "text": text,
                        "tokens": " ".join(tokens),
                        "ngram_size": len(tokens),
                        "sentence_id": idx
                    }
                )
            )

            if len(points) >= BATCH_SIZE:
                client.upsert(COLLECTION_NAME, points)
                inserted += len(points)
                points = []

    if points:
        client.upsert(COLLECTION_NAME, points)
        inserted += len(points)

    print("🎉 Qdrant insert tamamlandı:", inserted)


if __name__ == "__main__":
    main()
