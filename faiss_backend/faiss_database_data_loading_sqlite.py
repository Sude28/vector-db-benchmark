# faiss_backend/faiss_database_data_loading_sqlite.py
import os, json, re, unicodedata, sqlite3
import numpy as np
import faiss
import nltk
from tqdm import tqdm

nltk.download("punkt")

EMBEDDINGS_PATH = "data/embeddings.npy"
TEXTS_PATH = "data/texts.jsonl"

OUT_DIR = "faiss_backend"
FAISS_INDEX_PATH = os.path.join(OUT_DIR, "index.faiss")
SQLITE_PATH = os.path.join(OUT_DIR, "meta.sqlite")

DIM = 128
BATCH_SIZE = 5000
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

os.makedirs(OUT_DIR, exist_ok=True)

print("📦 embeddings mmap loading...")

# satır sayısını texts.jsonl'den say
with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

DIM = 128

emb = np.memmap(
    EMBEDDINGS_PATH,
    dtype=np.float32,
    mode="r",
    shape=(total_lines, DIM)
)
total = emb.shape[0]

print("✅ embeddings memmap yüklendi:", emb.shape)

# SQLite hazırlık
if os.path.exists(SQLITE_PATH):
    os.remove(SQLITE_PATH)

con = sqlite3.connect(SQLITE_PATH)
cur = con.cursor()
cur.execute("""
CREATE TABLE meta (
  faiss_id INTEGER PRIMARY KEY,
  sentence_id INTEGER,
  ngram_size INTEGER,
  text TEXT,
  tokens TEXT
)
""")
cur.execute("CREATE INDEX idx_text ON meta(text)")
con.commit()

# FAISS index
index = faiss.IndexFlatIP(DIM)

batch_vecs = []
batch_rows = []
faiss_id = 0

with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    for sentence_id, line in tqdm(enumerate(f), total=total, desc="📥 FAISS+SQLite insert"):
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

        tokens_list = nltk.word_tokenize(text.lower())
        if not tokens_list:
            continue

        tokens_str = " ".join(tokens_list)
        tokens_str = truncate_utf8_bytes(tokens_str, MAX_TEXT_LEN)

        vec = np.array(emb[sentence_id], dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)

        batch_vecs.append(vec)
        batch_rows.append((faiss_id, sentence_id, len(tokens_list), text, tokens_str))
        faiss_id += 1

        if len(batch_vecs) >= BATCH_SIZE:
            vec_np = np.vstack(batch_vecs)
            index.add(vec_np)

            cur.executemany("INSERT INTO meta VALUES (?,?,?,?,?)", batch_rows)
            con.commit()

            batch_vecs, batch_rows = [], []

# kalanlar
if batch_vecs:
    vec_np = np.vstack(batch_vecs)
    index.add(vec_np)
    cur.executemany("INSERT INTO meta VALUES (?,?,?,?,?)", batch_rows)
    con.commit()

con.close()

faiss.write_index(index, FAISS_INDEX_PATH)

print("🎉 Tamamlandı!")
print("✅ index:", FAISS_INDEX_PATH)
print("✅ sqlite:", SQLITE_PATH)
print("✅ toplam kayıt:", faiss_id)
