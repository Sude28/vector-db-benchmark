# Dataset Information

This directory is intended to store dataset files and generated embeddings used in the benchmark experiments.

The experiments in this project were conducted using a Turkish text corpus stored in a file named tr_corpus.txt. Due to its relatively large size, the corpus file is not included in this repository.

During the embedding generation stage, the corpus was processed using a TinyBERT-based embedding pipeline to produce vector representations. The resulting embedding file embeddings.npy is also large and therefore not included in the repository.

Both the corpus file and the generated embeddings can be recreated by running the embedding generation scripts provided in the embedding/ directory.

These files were excluded from the repository to keep the project lightweight and maintain manageable repository size limits.
