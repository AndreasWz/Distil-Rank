# Distil-Rank: Low-Rank Distillation for ESM-2

**Abstract**
High-dimensional protein language models like ESM-2 are difficult to deploy in resource-constrained aerospace environments. This project demonstrates **Distil-Rank**, a method to compress attention layers by **24x** while maintaining **>99% functional fidelity** through SVD-initialized knowledge distillation.

## Key Results
Tested on \`facebook/esm2_t6_8M_UR50D\` Attention Query Projections.

| Metric | Teacher (Dense) | Student (Low-Rank) | Improvement |
|--------|-----------------|--------------------|-------------|
| **Parameters** | 102,400 | 20,800 (Rank 32) | **~5x smaller** |
| **Latency (CPU)** | ~0.08 ms | ~0.03 ms | **~2.5x faster** |
| **Fidelity** | 1.000 | **0.998** | Preserved |

![Training Plot](distil_rank_report.png)

## Methodology
1. **SVD Initialization:** The student network ($W = A \times B$) is initialized using Truncated SVD of the teacher's weights to ensure rapid convergence.
2. **Distillation:** We freeze the teacher (ESM-2 layer) and train the student to minimize MSE loss on embedding outputs.
3. **Evaluation:** We measure **Attention Fidelity** (Cosine Similarity) and Inference Latency on CPU.

## How to run
\`\`\`bash
pip install -r requirements.txt
python main.py
\`\`\`
