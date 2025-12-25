# Distil-Rank: High-Fidelity Compression of Bio-Embedding Projections

**Abstract**
Downstream tasks on protein language models (like **ESM-2**) utilize high-dimensional projection layers that are often redundant. This project demonstrates **Distil-Rank**, a pipeline that compresses these **downstream projection layers** by **10x**. It recovers **near-perfect functional fidelity (99.8%)** through knowledge distillation, exploiting the **low intrinsic dimensionality** of biological embeddings.

![Results Plot](results/distil_rank_final_report.png)

## Key Results (Real ESM-2 Data)
Tested on projection layers operating on embeddings from the **ESM-2 (650M)** model.

| Metric | SVD Baseline (Static) | Distil-Rank (Trained) | Improvement |
|--------|-----------------------|-----------------------|-------------|
| **Fidelity (Cosine)** | 0.6399 | **0.9983** | **+35.8 points** (Recovery) |
| **Rel. MSE Error** | 0.7682 | **0.0578** | **-92% Error Reduction** |
| **Latency (CPU)** | 0.49 ms | **0.06 ms** | **7.91x Speedup** |
| **Parameters** | 1.64M | 0.16M | **9.9x Compression** |

## Methodology
1. **SVD Warmstart:** Initialize the student ($W = A \times B$) using Truncated SVD.
2. **Distillation:** Train the student to minimize combined Loss (Relative MSE + Cosine) against the Teacher using real protein embeddings.
3. **Robust Benchmarking:** Median latency measurement over 300 runs.

## Quick Start
To reproduce the results with synthetic data:
\`\`\`bash
pip install -r requirements.txt
python main.py
\`\`\`
*(For real-data reproduction, generate ESM-2 embeddings using the provided Colab script and place them in the project root.)*
