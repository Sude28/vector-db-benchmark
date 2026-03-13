# -- coding: utf-8 --
"""
TXT → TinyBERT TensorRT → embeddings.npy
Jetson Nano üzerinde çalışır.

Özellikler:
- Dynamic batch destekli TRT engine ile uyumlu
- Son batch opt_batch'e pad edilir
- RAM şişmeden disk’e memmap ile yazar
"""

import json
import numpy as np
from tinybert_trt_embed_module_final import TinyBertTRTEmbedder


# ===============================
# 1) Ayarlar
# ===============================

TXT_FILE = "tr_corpus.txt"
ENGINE_FILE = "tinybert_dynamic_fp16.engine"
VOCAB_FILE = "vocab.txt"

MAX_LEN = 64
BATCH_SIZE = 8


# ===============================
# 2) Metni yükle
# ===============================

def load_texts(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


# ===============================
# 3) Batch embedding (MEMMAP SAFE)
# ===============================

def batch_embed(embedder, texts):
    n = len(texts)
    opt_batch = embedder.opt_batch

    print(f"[INFO] Toplam {n} satır bulundu.")

    # İlk batch ile dimension öğren
    first_batch = texts[:BATCH_SIZE]
    first_emb = embedder.encode_batch(first_batch)

    if first_emb is None:
        raise RuntimeError("İlk batch embedding üretilemedi.")

    dim = first_emb.shape[1]

    # Disk üzerinde array oluştur
    mmap_array = np.memmap(
        "embeddings.npy",
        dtype=np.float32,
        mode="w+",
        shape=(n, dim)
    )

    # İlk batch yaz
    mmap_array[0:len(first_emb)] = first_emb
    write_index = len(first_emb)

    # Kalan batchler
    for i in range(BATCH_SIZE, n, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        real_bs = len(batch)

        print(f"[INFO] {i}/{n} işleniyor...")

        if real_bs < opt_batch:
            pad_count = opt_batch - real_bs
            padded_batch = batch + [""] * pad_count
            emb = embedder.encode_batch(padded_batch)
            emb = emb[:real_bs]
        else:
            emb = embedder.encode_batch(batch)

        if emb is None:
            raise RuntimeError(f"{i}. batch embedding üretilemedi.")

        mmap_array[write_index:write_index+real_bs] = emb
        write_index += real_bs

    mmap_array.flush()
    print("[OK] embeddings.npy disk’e yazıldı (RAM şişmeden).")


# ===============================
# 4) Ana fonksiyon
# ===============================

def main():
    print("\n=== TXT → TinyBERT TRT → embeddings.npy Export (FINAL) ===\n")

    embedder = TinyBertTRTEmbedder(
        engine_path=ENGINE_FILE,
        vocab_path=VOCAB_FILE,
        max_len=MAX_LEN,
        l2_normalize=False,
        verbose_bindings=True,
    )
    print("[OK] TinyBERT TRT embedder hazır.")

    texts = load_texts(TXT_FILE)
    print(f"[OK] {len(texts)} satır metin okundu.")

    batch_embed(embedder, texts)

    # texts.json kaydet
    with open("texts.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    print("[OK] texts.json kaydedildi.")
    print("\n=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
