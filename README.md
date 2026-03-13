# Vector Database Benchmark for Semantic Search

This repository presents a benchmarking framework for evaluating vector database systems used in semantic search applications.

The project focuses on generating text embeddings on an edge device and comparing the performance of different vector database systems under the same experimental conditions.

The benchmark evaluates the following vector search systems:
	•	FAISS
	•	Milvus
	•	Qdrant

All systems are tested using the same embedding dataset and the same query set to ensure fair comparison.

⸻

## Project Motivation

Modern information retrieval systems increasingly rely on semantic embeddings instead of traditional keyword-based search.

Embedding models convert textual data into high-dimensional vectors that capture semantic relationships between texts. These vectors can then be used to perform similarity search.

However, performing similarity search over large embedding collections requires specialized indexing and retrieval systems known as vector databases.

Different vector database systems use different indexing strategies and system architectures, which can lead to different performance characteristics.

This project aims to experimentally evaluate these systems using the same dataset, embedding model and benchmark queries.

⸻

## System Overview

The experimental pipeline used in this project consists of the following stages:
	1.	Text corpus preparation
	2.	Embedding generation using a TinyBERT-based model
	3.	Optimization and inference using TensorRT
	4.	Storage of embedding vectors in vector databases
	5.	Similarity search benchmarking

The embedding generation process was executed on an NVIDIA Jetson Nano device using a TensorRT optimized model.

The generated embeddings were then used as input for multiple vector database systems in order to measure and compare their performance.

## Dataset and Embeddings

The experiments were conducted using a Turkish text corpus stored in a file named tr_corpus.txt.

Due to its large size, the corpus file is not included in this repository.

During the embedding generation stage, the corpus was processed to produce embedding vectors which were stored in a file named embeddings.npy.

Both the dataset and the generated embeddings are excluded from the repository to keep the project lightweight.

Embeddings can be regenerated using the scripts provided in the embedding/ directory.

⸻

## Benchmark Methodology

All vector database systems were evaluated under the same experimental setup.

The following steps were applied:
	1.	Load the embedding vectors
	2.	Insert embeddings into the vector database
	3.	Execute similarity search queries
	4.	Measure performance metrics

The benchmark focuses on the following metrics:
	•	Query latency
	•	Memory usage (RAM)
	•	Result overlap between systems
	•	Consensus similarity results

These metrics provide insight into both the efficiency and behavior of different vector database implementations.

⸻

## Results

Benchmark results and visualizations are provided in the plots/ directory.

The results include:
	•	Average query latency
	•	P95 latency
	•	Peak RAM usage
	•	Result overlap between systems
	•	Consensus similarity comparisons

These results allow a direct comparison of vector search performance across different systems.

⸻

## Technologies Used
	•	Python
	•	TinyBERT
	•	TensorRT
	•	CUDA
	•	FAISS
	•	Milvus
	•	Qdrant
	•	NumPy

⸻

## Notes

Large files such as the dataset and generated embeddings are intentionally excluded from the repository.

All experiments can be reproduced using the provided scripts.

## Documentation

The docs/ directory contains the full project report prepared within the scope of the TÜBİTAK 2209-B research project.

The report includes detailed explanations of the system architecture, experimental methodology, benchmark setup, and evaluation results. All figures and benchmark visualizations are presented in the report.

Readers interested in the full experimental details are encouraged to consult the documentation provided in the docs/ folder.
