# -*- coding: utf-8 -*-
"""
TinyBERT TensorRT text → embedding modülü (FINAL v3)
---------------------------------------------------
- PyCUDA YOK
- Transformers YOK
- Sadece TensorRT + CUDA runtime (ctypes)
- Jetson Nano için optimize
- Engine output: raw last_hidden_state (B, seq_len, hidden)
- Bu modül CLS pooling yapar → (B, hidden)
- Dinamik batch: min/opt/max batch profilden okunur + runtime check
- Binding index GARANTİ DEĞİL → binding isimlerinden çözülür (V3 fix)
"""

import os
import re
import unicodedata
import ctypes
import numpy as np
import tensorrt as trt
from typing import Optional, Dict
from typing import List

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ============================================================
# CUDA RUNTIME (ctypes)
# ============================================================
libcudart = ctypes.CDLL("libcudart.so")


#SON EKLEDİM
# ---- ctypes CUDA signatures (ÇOK ÖNEMLİ - 64bit pointer issues fix) ----
libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
libcudart.cudaMalloc.restype  = ctypes.c_int

libcudart.cudaFree.argtypes   = [ctypes.c_void_p]
libcudart.cudaFree.restype    = ctypes.c_int

libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libcudart.cudaMemcpy.restype  = ctypes.c_int

libcudart.cudaDeviceSynchronize.argtypes = []
libcudart.cudaDeviceSynchronize.restype  = ctypes.c_int
#SON EKLEDİM


cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2


def cuda_check(status, msg):
    if status != 0:
        raise RuntimeError(f"{msg} (status={status})")


def cuda_malloc(nbytes: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    status = libcudart.cudaMalloc(ctypes.byref(ptr), nbytes)
    cuda_check(status, "cudaMalloc failed")
    return ptr


def cuda_free(ptr: ctypes.c_void_p):
    if ptr and ptr.value:
        status = libcudart.cudaFree(ptr)
        cuda_check(status, "cudaFree failed")


def cuda_memcpy_htod_bytes(dst: ctypes.c_void_p, src_np: np.ndarray, nbytes: int):
    status = libcudart.cudaMemcpy(
        dst,
        src_np.ctypes.data_as(ctypes.c_void_p),
        nbytes,
        cudaMemcpyHostToDevice,
    )
    cuda_check(status, "cudaMemcpy HtoD failed")

def cuda_memcpy_dtoh_bytes(dst_np: np.ndarray, src: ctypes.c_void_p, nbytes: int):
    status = libcudart.cudaMemcpy(
        dst_np.ctypes.data_as(ctypes.c_void_p),
        src,
        nbytes,
        cudaMemcpyDeviceToHost,
    )
    cuda_check(status, "cudaMemcpy DtoH failed")


# ============================================================
# WORDPIECE TOKENIZER (Light, HF vocab.txt uyumlu)
# ============================================================
class WordPieceTokenizerLite:
    _punct_re = re.compile(r"([!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~])")

    def __init__(self, vocab_path: str):
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab.txt bulunamadı: {vocab_path}")

        self.vocab = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for idx, tok in enumerate(f):
                tok = tok.strip()
                if tok:
                    self.vocab[tok] = idx

        self.unk_id = self.vocab.get("[UNK]", 0)
        self.cls_id = self.vocab.get("[CLS]", 101)
        self.sep_id = self.vocab.get("[SEP]", 102)
        self.pad_id = self.vocab.get("[PAD]", 0)

    def _basic_tokenize(self, text: str):
        text = unicodedata.normalize("NFKC", text)
        text = text.lower().strip()
        text = self._punct_re.sub(r" \1 ", text)
        return [t for t in text.split() if t]

    def _wordpiece(self, token: str):
        if token in self.vocab:
            return [token]

        sub_tokens = []
        start = 0
        while start < len(token):
            end = len(token)
            cur = None
            while start < end:
                piece = token[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    cur = piece
                    break
                end -= 1
            if cur is None:
                return ["[UNK]"]
            sub_tokens.append(cur)
            start = end
        return sub_tokens

    def encode(self, text: str, max_len: int = 64):
        basic = self._basic_tokenize(text)
        wp = []
        for t in basic:
            wp.extend(self._wordpiece(t))

        wp = wp[: max_len - 2]  # [CLS] ... [SEP]
        token_ids = [self.cls_id] + [self.vocab.get(t, self.unk_id) for t in wp] + [self.sep_id]

        pad_len = max_len - len(token_ids)
        if pad_len > 0:
            token_ids += [self.pad_id] * pad_len

        attn = [0 if tid == self.pad_id else 1 for tid in token_ids]

        return (
            np.array(token_ids, dtype=np.int32),
            np.array(attn, dtype=np.int32),
        )


# ============================================================
# TENSORRT EMBEDDER (CLS pooling + dynamic batch + name bindings)
# ============================================================
class TinyBertTRTEmbedder:
    def __init__(
        self,
        engine_path: str,
        vocab_path: str,
        max_len: int = 64,
        max_batch_override: Optional[int] = None,
        l2_normalize: bool = False,
        profile_index: int = 0,
        binding_names: Optional[Dict] = None,
        verbose_bindings: bool = True,
    ):
        """
        binding_names: TRT engine binding isimleri farklıysa override edebilirsin.
          default:
            {"input_ids": "input_ids", "attention_mask": "attention_mask", "output": "last_hidden_state"}
        """
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine bulunamadı: {engine_path}")

        self.max_len = int(max_len)
        self.seq_len = self.max_len
        self.l2_normalize = bool(l2_normalize)
        self.profile_index = int(profile_index)

        # ---- load engine
        print("[INFO] TRT engine yükleniyor:", engine_path)
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Engine deserialize edilemedi!")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Execution context oluşturulamadı!")

        # Opt profile
        try:
            self.context.active_optimization_profile = self.profile_index
        except Exception:
            pass

        # ---- tokenizer
        self.tokenizer = WordPieceTokenizerLite(vocab_path)

        # ---- binding name map (V3)
        names = binding_names or {
            "input_ids": "input_ids",
            "attention_mask": "attention_mask",
            "output": "last_hidden_state",
        }
        self._resolve_bindings_by_name(
            input_ids_name=names["input_ids"],
            attention_mask_name=names["attention_mask"],
            output_name=names["output"],
            verbose=verbose_bindings,
        )

        # ---- output shape → hidden size
        out_shape = tuple(self.engine.get_binding_shape(self.binding_out))
        # last_hidden_state: (B, L, H) bekliyoruz (dinamik batch olabilir)
        if len(out_shape) < 3:
            raise RuntimeError(f"Output beklenenden farklı: shape={out_shape} (last_hidden_state değil gibi)")

        self.hidden_size = int(out_shape[-1]) if out_shape[-1] != -1 else None
        if self.hidden_size is None:
            raise RuntimeError("Engine output hidden_size tespit edilemedi (shape dinamik).")

        # ---- dynamic batch aralığı (profile shape → input_ids binding üzerinden)
        try:
            min_shape, opt_shape, max_shape = self.engine.get_profile_shape(
                self.profile_index, self.binding_in_ids
            )
            self.min_batch = int(min_shape[0])
            self.opt_batch = int(opt_shape[0])
            self.max_batch = int(max_shape[0])
        except Exception:
            self.min_batch = self.opt_batch = self.max_batch = 1

        if max_batch_override is not None:
            self.max_batch = min(self.max_batch, int(max_batch_override))

        print(f"[OK] Engine yüklendi. hidden={self.hidden_size}, seq_len={self.seq_len}")
        print(f"[INFO] TRT dynamic batch: min={self.min_batch}, opt={self.opt_batch}, max={self.max_batch}")

        # ---- persistent buffers
        self._allocate_persistent_buffers()

    # ------------------ V3: binding resolve by name ------------------
    def _resolve_bindings_by_name(self, input_ids_name: str, attention_mask_name: str, output_name: str, verbose: bool):
        binding_map = {}
        for i in range(self.engine.num_bindings):
            binding_map[self.engine.get_binding_name(i)] = i

        if verbose:
            print("[INFO] TRT bindings (index → name, I/O, dtype, shape):")
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                io = "INPUT" if self.engine.binding_is_input(i) else "OUTPUT"
                dt = self.engine.get_binding_dtype(i)
                sh = self.engine.get_binding_shape(i)
                print(f"  - {i}: {name} | {io} | {dt} | shape={tuple(sh)}")

        for req in [input_ids_name, attention_mask_name, output_name]:
            if req not in binding_map:
                raise RuntimeError(
                    f"TRT engine binding bulunamadı: '{req}'. "
                    f"Mevcut binding isimleri: {list(binding_map.keys())}"
                )

        self.binding_in_ids = binding_map[input_ids_name]
        self.binding_in_mask = binding_map[attention_mask_name]
        self.binding_out = binding_map[output_name]

        # küçük güvenlik: input/output yönü kontrolü
        if not self.engine.binding_is_input(self.binding_in_ids):
            raise RuntimeError(f"'{input_ids_name}' binding input değil görünüyor.")
        if not self.engine.binding_is_input(self.binding_in_mask):
            raise RuntimeError(f"'{attention_mask_name}' binding input değil görünüyor.")
        if self.engine.binding_is_input(self.binding_out):
            raise RuntimeError(f"'{output_name}' binding output değil görünüyor.")

        print(f"[OK] Binding resolve: input_ids={self.binding_in_ids}, attention_mask={self.binding_in_mask}, output={self.binding_out}")

    # ------------------ buffers ------------------
    def _allocate_persistent_buffers(self):
        B, L, H = self.max_batch, self.max_len, self.hidden_size

        self.host_in_ids = np.zeros((B, L), dtype=np.int32)
        self.host_in_mask = np.zeros((B, L), dtype=np.int32)
        self.host_out = np.zeros((B, L, H), dtype=np.float32)

        self.dev_in_ids = cuda_malloc(self.host_in_ids.nbytes)
        self.dev_in_mask = cuda_malloc(self.host_in_mask.nbytes)
        self.dev_out = cuda_malloc(self.host_out.nbytes)

        # V3: bindings list engine.num_bindings uzunluğunda olmalı
        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.binding_in_ids] = int(self.dev_in_ids.value)
        self.bindings[self.binding_in_mask] = int(self.dev_in_mask.value)
        self.bindings[self.binding_out] = int(self.dev_out.value)

    def close(self):
        cuda_free(self.dev_in_ids)
        cuda_free(self.dev_in_mask)
        cuda_free(self.dev_out)
        self.dev_in_ids = self.dev_in_mask = self.dev_out = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------ helpers ------------------
    def _check_batch(self, batch_size: int):
        if batch_size < self.min_batch or batch_size > self.max_batch:
            raise ValueError(
                f"Batch size {batch_size} TRT engine aralığı dışında "
                f"({self.min_batch}–{self.max_batch}). opt={self.opt_batch}"
            )

    def _postprocess_cls(self, last_hidden: np.ndarray) -> np.ndarray:
        cls = last_hidden[:, 0, :]
        if self.l2_normalize:
            denom = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-12
            cls = cls / denom
        return cls.astype(np.float32, copy=False)

    # ------------------ public API ------------------
    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, text_list: List[str]) -> np.ndarray:
        batch_size = len(text_list)
        if batch_size == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        self._check_batch(batch_size)

        # (ÖNEMLİ) Önce host buffer'ı sıfırla (eski verinin sızmasını engeller)
        self.host_in_ids[:batch_size].fill(0)
        self.host_in_mask[:batch_size].fill(0)

        # tokenize → host buffer’a yaz
        for i, text in enumerate(text_list):
            ids, mask = self.tokenizer.encode(text, max_len=self.max_len)
            self.host_in_ids[i, :] = ids
            self.host_in_mask[i, :] = mask

    # -------------------------
    # HtoD: sadece batch_size kadar byte kopyala
    # -------------------------
        in_ids_bytes  = batch_size * self.max_len * np.dtype(np.int32).itemsize
        in_mask_bytes = batch_size * self.max_len * np.dtype(np.int32).itemsize

        cuda_memcpy_htod_bytes(self.dev_in_ids,  self.host_in_ids,  in_ids_bytes)
        cuda_memcpy_htod_bytes(self.dev_in_mask, self.host_in_mask, in_mask_bytes)

    # -------------------------
    # Shape set
    # -------------------------
        self.context.set_binding_shape(self.binding_in_ids,  (batch_size, self.max_len))
        self.context.set_binding_shape(self.binding_in_mask, (batch_size, self.max_len))

    # (debug) output shape inference'dan sonra netleşir
    # Ama bazı TRT versiyonlarında execute öncesi de görülebilir.
        try:
            out_shape_pre = tuple(self.context.get_binding_shape(self.binding_out))
            print("[DEBUG] out_shape_pre:", out_shape_pre)
        except Exception:
            pass

    # -------------------------
    # Execute
    # -------------------------
        ok = self.context.execute_v2(self.bindings)
        if not ok:
            raise RuntimeError("TRT execute_v2 başarısız döndü.")

    # -------------------------
    # Output shape al
    # -------------------------
        out_shape = tuple(self.context.get_binding_shape(self.binding_out))
    # Beklenen: (B, L, H)
        if len(out_shape) != 3:
            raise RuntimeError(f"Beklenmeyen output shape: {out_shape}")

        B, L, H = out_shape
        if B != batch_size:
            print(f"[WARN] output batch {B} != input batch {batch_size}. Yine de B kullanılacak.")
            batch_size = B

    # -------------------------
    # DtoH: sadece batch_size kadar byte kopyala
    # -------------------------
        out_bytes = batch_size * self.max_len * self.hidden_size * np.dtype(np.float32).itemsize
        cuda_memcpy_dtoh_bytes(self.host_out, self.dev_out, out_bytes)

    # -------------------------
    # CLS pooling
    # -------------------------
        return self._postprocess_cls(
            self.host_out[:batch_size, :, :].copy()
        )

# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("\n=== TinyBERT TRT Module Test (FINAL v3) ===\n")

    engine = "tinybertv2_fp16.engine"
    vocab = "vocab.txt"

    tb = TinyBertTRTEmbedder(
        engine_path=engine,
        vocab_path=vocab,
        max_len=64,
        l2_normalize=False,
        profile_index=0,
        verbose_bindings=True,   # ilk çalıştırmada True kalsın
    )

    emb = tb.encode_text("Jetson Nano üzerinde TRT ile embedding üretiyorum.")
    print("Embedding shape:", emb.shape)
    print("İlk 10 değer:", emb[:10])

    batch = tb.encode_batch([
        "merhaba dünya",
        "semantic search için embedding çıkarıyorum",
        "tinybert tensorrt jetson nano",
        "faiss index build",
        "python ctypes cudart",
        "wordpiece tokenizer",
        "last hidden state cls pooling",
        "tubitak proje"
    ])
    print("Batch shape:", batch.shape)
