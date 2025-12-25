#!/usr/bin/env python3
"""
Distil-Rank: final main.py
- Computes SVD baseline (pre-training) on fixed validation set
- Initializes low-rank student via SVD warmstart
- Distills student to teacher with combined relative-MSE + cosine loss
- Measures robust latency (median/mean) using a realistic batch
- Saves plots and JSON summary for reproducibility

Replace synthetic embeddings with real ESM embeddings by setting USE_REAL_EMBEDDINGS=True
and providing a torch-saved tensor "embeddings_train.pt" and "embeddings_val.pt".
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Toggle to load real embeddings if available (must provide files)
USE_REAL_EMBEDDINGS = False
EMB_TRAIN_PATH = "embeddings_train.pt"  # tensor shape [N, INPUT_DIM]
EMB_VAL_PATH = "embeddings_val.pt"      # tensor shape [N_val, INPUT_DIM]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model / training config
INPUT_DIM = 1280
OUTPUT_DIM = 1280
TARGET_RANK = 64
BATCH_SIZE = 64
EVAL_BATCH = 32
STEPS = 1000
LR = 3e-3

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

print("=== DISTIL-RANK FINAL SCRIPT ===")
print(f"Device: {DEVICE}; INPUT_DIM={INPUT_DIM}; TARGET_RANK={TARGET_RANK}")

# -------------------------
# Data / Validation Set
# -------------------------
if USE_REAL_EMBEDDINGS and os.path.exists(EMB_VAL_PATH) and os.path.exists(EMB_TRAIN_PATH):
    print("[Data] Loading real embeddings for train/val")
    embeddings_train = torch.load(EMB_TRAIN_PATH).to(DEVICE)
    embeddings_val = torch.load(EMB_VAL_PATH).to(DEVICE)
    # Use first dims if input dims mismatch
    assert embeddings_train.shape[1] >= INPUT_DIM, "Train embeddings width < INPUT_DIM"
    assert embeddings_val.shape[1] >= INPUT_DIM, "Val embeddings width < INPUT_DIM"
    x_val = embeddings_val[:512, :INPUT_DIM].contiguous()
    def sample_train_batch(bs):
        idx = torch.randint(0, embeddings_train.shape[0], (bs,))
        return embeddings_train[idx, :INPUT_DIM]
else:
    # Structured synthetic data (recommended for demonstration)
    print("[Data] Using structured synthetic data (importance mask).")
    importance_mask = torch.ones(INPUT_DIM, device=DEVICE)
    importance_mask[:100] = 5.0
    importance_mask[500:550] = 0.1
    def sample_train_batch(bs):
        x = torch.randn(bs, INPUT_DIM, device=DEVICE)
        return x * importance_mask
    x_val = (torch.randn(512, INPUT_DIM, device=DEVICE) * importance_mask)

# -------------------------
# Teacher (synthetic high-rank)
# -------------------------
class TeacherLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, OUTPUT_DIM, bias=True)
        with torch.no_grad():
            # Build a higher-rank target (e.g., rank ~256)
            U = torch.randn(OUTPUT_DIM, 256, device=DEVICE)
            V = torch.randn(256, INPUT_DIM, device=DEVICE)
            self.linear.weight.copy_((U @ V) * 0.02)  # scale for numerical stability
            self.linear.bias.fill_(0.01)
    def forward(self, x):
        return self.linear(x)

teacher = TeacherLayer().to(DEVICE)
teacher.eval()

# -------------------------
# SVD baseline (pre-training)
# -------------------------
print("\n[Step] Computing SVD baseline on validation set (pre-training)")
with torch.no_grad():
    W = teacher.linear.weight.data.clone().to(DEVICE)  # [out, in]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # U:[out,r], S:[r], Vh:[r,in]
    r = TARGET_RANK
    W_svd = (U[:, :r] @ torch.diag(S[:r])) @ Vh[:r, :]   # [out, in]
    y_t_val = teacher(x_val)
    y_svd_val = torch.nn.functional.linear(x_val, W_svd, bias=teacher.linear.bias.to(DEVICE))
    svd_fidelity = torch.nn.functional.cosine_similarity(y_t_val, y_svd_val, dim=-1).mean().item()
    svd_rel_mse = (torch.norm(y_t_val - y_svd_val) / (torch.norm(y_t_val) + 1e-12)).item()
print(f" SVD baseline fidelity (cosine): {svd_fidelity:.6f}; rel_MSE: {svd_rel_mse:.6e}")

# -------------------------
# Student model + SVD init
# -------------------------
class LowRankStudent(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A_enc = nn.Linear(in_dim, rank, bias=False)   # [rank, in]
        self.B_dec = nn.Linear(rank, out_dim, bias=True)   # [out, rank]
    def forward(self, x):
        return self.B_dec(self.A_enc(x))

student = LowRankStudent(INPUT_DIM, OUTPUT_DIM, TARGET_RANK).to(DEVICE)

# SVD warmstart: split sqrt(S) between A and B
with torch.no_grad():
    U_r = U[:, :r]        # [out, r]
    S_r = S[:r]           # [r]
    Vh_r = Vh[:r, :]      # [r, in]
    sqrt_S = torch.diag(torch.sqrt(S_r))
    # A_enc.weight expected shape [rank, in] -> sqrt_S @ Vh_r  => [r, in]
    student.A_enc.weight.copy_((sqrt_S @ Vh_r).to(student.A_enc.weight.dtype))
    # B_dec.weight expected shape [out, rank] -> U_r @ sqrt_S => [out, r]
    student.B_dec.weight.copy_((U_r @ sqrt_S).to(student.B_dec.weight.dtype))
    student.B_dec.bias.copy_(teacher.linear.bias.to(student.B_dec.bias.dtype))

# fidelity before training
with torch.no_grad():
    y_student_init = student(x_val)
    init_fidelity = torch.nn.functional.cosine_similarity(y_t_val, y_student_init, dim=-1).mean().item()
    init_rel_mse = (torch.norm(y_t_val - y_student_init) / (torch.norm(y_t_val) + 1e-12)).item()

t_params = sum(p.numel() for p in teacher.parameters())
s_params = sum(p.numel() for p in student.parameters())

print(f"\nStudent pre-train fidelity: {init_fidelity:.6f}; rel_MSE: {init_rel_mse:.6e}")
print(f"Params: teacher={t_params:,}, student={s_params:,}, compression={t_params/s_params:.2f}x")

# -------------------------
# Losses and utilities
# -------------------------
def relative_mse(y_pred, y_true):
    return torch.norm(y_pred - y_true) / (torch.norm(y_true) + 1e-12)

def cosine_loss(y_pred, y_true):
    return 1.0 - torch.nn.functional.cosine_similarity(y_pred, y_true, dim=-1).mean()

def combined_loss(y_pred, y_true, alpha=0.1):
    return relative_mse(y_pred, y_true) + alpha * cosine_loss(y_pred, y_true)

def measure_latency_ms(model, input_data, runs=300, warmup=20):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(input_data)
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), float(np.mean(times))

# -------------------------
# Training loop (distillation)
# -------------------------
print("\n[Step] Distillation training (combined loss)")
optimizer = optim.AdamW(student.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)

train_combined = []
train_rel_mse = []
val_rel_mse = []
val_cos_fid = []
val_steps = []

start_time = time.time()
for step in range(STEPS):
    student.train()
    x = sample_train_batch(BATCH_SIZE).to(DEVICE)
    with torch.no_grad():
        y_t = teacher(x)
    y_s = student(x)

    loss = combined_loss(y_s, y_t, alpha=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    train_combined.append(float(loss.item()))
    train_rel_mse.append(float(relative_mse(y_s, y_t).item()))

    if step % 50 == 0:
        student.eval()
        with torch.no_grad():
            y_s_val = student(x_val)
            rel_m = (torch.norm(y_t_val - y_s_val) / (torch.norm(y_t_val) + 1e-12)).item()
            cos_f = torch.nn.functional.cosine_similarity(y_t_val, y_s_val, dim=-1).mean().item()
            val_rel_mse.append(rel_m)
            val_cos_fid.append(cos_f)
            val_steps.append(step)
        if step % 200 == 0:
            print(f" step {step:04d}: combined_loss={loss.item():.6f}, val_cos_fid={cos_f:.6f}, val_rel_mse={rel_m:.6e}")

train_time = time.time() - start_time
print(f" Training finished, time: {train_time:.2f}s")

# final evaluation
student.eval()
with torch.no_grad():
    y_s_final = student(x_val)
    final_cos_fid = torch.nn.functional.cosine_similarity(y_t_val, y_s_final, dim=-1).mean().item()
    final_rel_mse = (torch.norm(y_t_val - y_s_final) / (torch.norm(y_t_val) + 1e-12)).item()

print("\n[Results]")
print(f" SVD baseline fidelity: {svd_fidelity:.6f}")
print(f" Student pre-train fidelity: {init_fidelity:.6f}")
print(f" Student post-train fidelity: {final_cos_fid:.6f}")
print(f" Improvement over SVD (pp): {(final_cos_fid - svd_fidelity) * 100:.3f} %-points")
print(f" Student rel MSE (post): {final_rel_mse:.6e}")

# -------------------------
# Latency measurement (robust)
# -------------------------
print("\n[Step] Latency measurement")
dummy = torch.randn(EVAL_BATCH, INPUT_DIM, device=DEVICE)
med_t, mean_t = measure_latency_ms(teacher, dummy, runs=300, warmup=20)
med_s, mean_s = measure_latency_ms(student, dummy, runs=300, warmup=20)
print(f" Teacher latency median {med_t:.4f} ms (mean {mean_t:.4f})")
print(f" Student latency median {med_s:.4f} ms (mean {mean_s:.4f})")
print(f" Speedup (median): {med_t/med_s:.2f}x")

# -------------------------
# Plots & Save results
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(train_combined, alpha=0.6)
axes[0].set_title("Combined Training Loss (relativeMSE + 0.1*cos)")
axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss"); axes[0].grid(alpha=0.3)

axes[1].plot(val_steps, val_rel_mse, marker='o', color='tab:orange')
axes[1].set_title("Validation Relative MSE"); axes[1].set_xlabel("Step"); axes[1].grid(alpha=0.3)

axes[2].plot(val_steps, val_cos_fid, marker='o', color='tab:green', label='Student (trained)')
axes[2].axhline(svd_fidelity, color='tab:red', linestyle='--', label='SVD baseline')
axes[2].set_title("Functional Recovery (Cosine Fidelity)")
axes[2].set_xlabel("Step"); axes[2].set_ylabel("Cosine Fidelity"); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "distil_rank_final_report.png")
plt.savefig(plot_path)
print(f" Saved plot: {plot_path}")

summary = {
    "svd_fidelity": float(svd_fidelity),
    "svd_rel_mse": float(svd_rel_mse),
    "student_pre_fidelity": float(init_fidelity),
    "student_post_fidelity": float(final_cos_fid),
    "student_pre_rel_mse": float(init_rel_mse),
    "student_post_rel_mse": float(final_rel_mse),
    "teacher_params": int(t_params),
    "student_params": int(s_params),
    "compression": float(t_params/s_params),
    "latency_teacher_median_ms": float(med_t),
    "latency_student_median_ms": float(med_s),
    "train_time_s": float(train_time),
    "seed": int(SEED),
    "use_real_embeddings": bool(USE_REAL_EMBEDDINGS)
}

json_path = os.path.join(OUT_DIR, "distil_rank_summary.json")
with open(json_path, "w") as fh:
    json.dump(summary, fh, indent=2)
print(f" Saved summary JSON: {json_path}")

print("\n=== Script finished ===")
