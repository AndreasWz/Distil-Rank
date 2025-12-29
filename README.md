# Distil-Rank: High-Fidelity Compression of Bio-Embedding Projections

**Abstract**
Downstream tasks on protein language models (like **ESM-2**) utilize high-dimensional projection layers that are often redundant. This project demonstrates **Distil-Rank**, a pipeline that compresses these **downstream projection layers** by **10x**. It recovers **near-perfect functional fidelity (99.8%+)** through knowledge distillation, exploiting the **low intrinsic dimensionality** of biological embeddings.

![Results Plot](results/distil_rank_final_report.png)

## Key Results (Real ESM-2 Data)
Tested on projection layers operating on embeddings from the **ESM-2 (650M)** model across three teacher initialization strategies.

### Comparison: Teacher Initialization Strategies

| Teacher Init | SVD Cosine | Student Cosine | Improvement | MSE Reduction |
|--------------|------------|----------------|-------------|---------------|
| **Random** | 0.467 | **0.998** | +53.1% | 99.5% |
| **Task-Trained** | 0.999 | **1.000** | +0.1% | 98.8% |
| **Spectral Decay** | 0.847 | **0.999** | +15.2% | 99.2% |

### Performance Metrics (All Strategies)

| Metric | Value |
|--------|-------|
| **Speedup** | ~7.5x |
| **Compression** | 9.9x (1.64M → 0.16M params) |
| **Target Rank** | 64 |

### Insights

- **Task-Trained Teacher**: Most realistic scenario. Teacher learns a denoising proxy task on real embeddings, creating weight structure aligned with the data manifold. SVD already achieves 0.999 fidelity, showing learned layers have natural low-rank structure.
- **Random Teacher**: Demonstrates distillation's power—even arbitrary projections achieve near-perfect recovery.
- **Spectral Decay**: Simulates realistic singular value decay patterns found in trained neural networks.

The **singular value spectrum plot** (right panel) reveals why: task-trained weights have sharp spectral decay (energy concentrated in top components), while random weights have flat spectra requiring distillation to recover fidelity.

## Mathematical Foundations
This approach combines linear algebra with neural knowledge distillation.

### 1. The Eckart-Young-Mirsky Theorem
A standard projection layer is a dense matrix $W \in \mathbb{R}^{m \times n}$. Standard compression uses **Truncated SVD** to find the optimal rank-$r$ approximation by keeping the largest singular values $\Sigma_r$:
$$W \approx U_r \Sigma_r V_r^T$$
This minimizes the Frobenius norm $||W - W_{approx}||_F$, assuming all input directions are equally important.

### 2. Distillation on the Data Manifold
In biological data, inputs $x$ are not uniformly distributed; they lie on a low-dimensional manifold. **Distil-Rank** factorizes the layer into two smaller matrices $A \in \mathbb{R}^{r \times n}$ and $B \in \mathbb{R}^{m \times r}$:
$$y_{student} = B(A(x))$$
We initialize $A$ and $B$ using the SVD components ("Warmstart") to ensure stability:
$$A_{init} = \sqrt{\Sigma_r} V_r^T, \quad B_{init} = U_r \sqrt{\Sigma_r}$$
We then train $A, B$ to minimize a **combined loss** on *real data*:
$$L = \mathcal{L}_{MSE}(y_s, y_t) + \alpha \cdot (1 - \text{cosine}(y_s, y_t))$$
This allows the student to rotate the SVD basis to align with the actual bio-embedding manifold, recovering fidelity that static SVD loses.

### 3. Complexity Reduction
By forcing the rank constraint $r \ll \min(m, n)$, we reduce computational complexity quadratically:
* **Teacher:** $O(m \cdot n)$ operations.
* **Student:** $O(r(m + n))$ operations.
* For $m=n=1280$ and $r=64$, this yields a theoretical **~10x reduction** in FLOPs.

### 4. Why Task-Trained Teachers Have Low-Rank Structure
When a neural network layer is trained on structured data (like protein embeddings), the weight matrix naturally develops spectral concentration. The learned transformation $W$ aligns with the data covariance structure:
$$W \approx \sum_{i=1}^{k} \sigma_i u_i v_i^T, \quad k \ll \min(m,n)$$
where $k$ is the effective rank determined by the data's intrinsic dimensionality. This explains why SVD baseline already achieves high fidelity (0.999) on task-trained teachers—the weights are already nearly low-rank.

## Quick Start & Reproduction

The benchmark automatically detects if real data is available.

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Benchmark:**
   ```bash
   python main.py
   ```

### Data Modes

- **Real Data Mode**: If `embeddings_train.pt` and `embeddings_val.pt` are found, it runs the full benchmark on ESM-2 embeddings with three teacher initialization strategies.
- **Synthetic Mode**: If files are missing, it falls back to structured synthetic data for demonstration.

### Output

Results are saved to `results/`:
- `distil_rank_summary.json` - All metrics in JSON format
- `distil_rank_final_report.png` - Visualization comparing strategies

## Project Structure

```
├── main.py                 # Main benchmark script
├── embeddings_train.pt     # ESM-2 training embeddings (optional)
├── embeddings_val.pt       # ESM-2 validation embeddings (optional)
├── requirements.txt        # Dependencies
├── README.md
└── results/
    ├── distil_rank_summary.json
    └── distil_rank_final_report.png
```

## License

MIT
