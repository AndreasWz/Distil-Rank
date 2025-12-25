# Distil-Rank: High-Performance Compression for Bio-Embeddings

**Abstract**
Deploying high-dimensional protein language models (like ESM-2 or ProtT5) in aerospace environments requires extreme efficiency. This project implements **Distil-Rank**, a knowledge distillation pipeline that compresses projection layers by **10x** while achieving **8x faster inference** and recovering **+13.6% functional fidelity** compared to standard mathematical decomposition (SVD).

![Results Plot](results/distil_rank_final_report.png)

## Key Results (Benchmark)
Tested on simulated High-Dimensional Embeddings (`dim=1280`, `rank=64`) with non-uniform feature importance (simulating biological signals).

| Metric | SVD Baseline (Static) | Distil-Rank (Trained) | Improvement |
|--------|-----------------------|-----------------------|-------------|
| **Fidelity (Cosine)** | 0.6954 | **0.8316** | **+13.6 points** (Recovery) |
| **Rel. MSE Error** | 0.7163 | **0.5510** | **-23% Error** |
| **Latency (CPU)** | 0.47 ms | **0.06 ms** | **7.91x Speedup** |
| **Parameters** | 1.64M | 0.16M | **9.9x Compression** |

## Why Distillation beats SVD
Standard **Truncated SVD** minimizes the Frobenius norm of the weight matrix. However, biological data is often sparse or structured (some features matter more). 
* **SVD** treats all dimensions equally (Fidelity: 69%).
* **Distil-Rank** learns from the data distribution, focusing capacity on high-importance features, recovering significant performance (Fidelity: 83%).

## Methodology
1.  **SVD Warmstart:** Initialize the student ($W = A \times B$) using Truncated SVD for stable convergence.
2.  **Combined Loss:** Optimize relative MSE + Cosine Similarity to maximize directionality alignment.
3.  **Robust Benchmarking:** Median latency measurement over 300 runs to exclude Python overhead.

## Quick Start
```bash
# Install dependencies
pip install torch numpy matplotlib

# Run reproduction script
python main.py