# milvus_database_data_loading.py
# -*- coding: utf-8 -*-

from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)
import json
import numpy as np
import nltk
from tqdm import tqdm
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


nltk.download("punkt")

# -------------------------
# 1️⃣ MILVUS BAĞLANTI
# -------------------------
connections.connect(
    alias="default",
    host="127.0.0.1",
    port="19530"
)

COLLECTION_NAME = "kased_collection_v4"

# -------------------------
# 2️⃣ SCHEMA
# -------------------------
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),  # Jetson çıktısı
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="tokens", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="ngram_size", dtype=DataType.INT64),
    FieldSchema(name="sentence_id", dtype=DataType.INT64)
]

schema = CollectionSchema(fields, description="Jetson TinyBERT Hybrid Search")

# -------------------------
# 3️⃣ COLLECTION OLUŞTUR
# -------------------------
if utility.has_collection(COLLECTION_NAME):
    print(f"⚠️ Collection var, siliniyor: {COLLECTION_NAME}")
    utility.drop_collection(COLLECTION_NAME)

collection = Collection(
    name=COLLECTION_NAME,
    schema=schema
)


print("✅ Collection hazır")

# -------------------------
# 4️⃣ VERİ DOSYALARI
# -------------------------
TEXTS_PATH = "data/texts.jsonl"
EMB_PATH = "data/embeddings.npy"

# -------------------------
# EMBEDDING MEMMAP LOAD (JETSON RAW FORMAT)
# -------------------------

# texts.jsonl satır sayısını say
with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

DIM = 128  # TinyBERT dimension

embeddings = np.memmap(
    EMB_PATH,
    dtype=np.float32,
    mode="r",
    shape=(total_lines, DIM)
)

print("✅ embeddings memmap yüklendi:", embeddings.shape)

# -------------------------sude@Sude:~$ export TMPDIR="/media/sude/DEBIA
# 5️⃣ BATCH PARAM
# -------------------------
BATCH_SIZE = 5000
buffer = []

def flush():
    global buffer
    if buffer:
        collection.insert(buffer)
        buffer = []

# -------------------------
# 6️⃣ STREAM + INSERT (SAFE)
# -------------------------
MAX_TEXT_LEN = 512
BATCH_SIZE = 500

def truncate_utf8_bytes(s: str, max_bytes: int) -> str:
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    b = b[:max_bytes]
    return b.decode("utf-8", errors="ignore")

buffer = []

def flush():
    global buffer
    if buffer:
        collection.insert(buffer)
        buffer = []

with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    for idx, line in tqdm(enumerate(f), desc="📥 Insert"):

        try:
            text = json.loads(line)  # jsonl satırı -> string
        except Exception:
            continue

        if not isinstance(text, str):
            continue

        text = text.strip()
        if len(text) < 3:
            continue

        text = clean_text(text)

        # ✅ BYTE bazlı kırp
        text = truncate_utf8_bytes(text, MAX_TEXT_LEN)

        tokens_list = nltk.word_tokenize(text.lower())
        if not tokens_list:
            continue

        tokens_str = " ".join(tokens_list)
        tokens_str = truncate_utf8_bytes(tokens_str, MAX_TEXT_LEN)

        buffer.append({
            "embedding": embeddings[idx].tolist(),
            "text": text,
            "tokens": tokens_str,
            "ngram_size": len(tokens_list),
            "sentence_id": idx
        })

        if len(buffer) >= BATCH_SIZE:
            flush()

flush()
collection.flush()
print("🎉 Milvus insert tamamlandı")


# -------------------------
# 7️⃣ INDEX (EN SON!)
# -------------------------
print("⏳ Index oluşturuluyor...")

collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "FLAT",
        "metric_type": "COSINE"
    }
)

print("✅ Index hazır")






















