import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from transformers import EsmModel
import os

# --- 1. CONFIGURATION & SETUP ---
torch.manual_seed(42)
np.random.seed(42)

# Konfiguration
MODEL_NAME = "facebook/esm2_t6_8M_UR50D" # Kleines ESM-2 für Demo
TARGET_RANK = 32                         # Ziel-Rank (Kompressionsfaktor ~24x)
BATCH_SIZE = 64
STEPS = 1000                             # Kurz & knackig für Demo
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== DISTIL-RANK PROJECT ===")
print(f"Device: {device}")
print(f"Target Model: {MODEL_NAME}")

# --- 2. LOAD REAL TEACHER (ESM-2) ---
print("\n[1] Loading Pre-trained Teacher (ESM-2)...")
try:
    # Versuche ESM-2 zu laden
    esm = EsmModel.from_pretrained(MODEL_NAME)
    # Wir isolieren die Query-Projection Matrix des ersten Layers
    real_weight = esm.encoder.layer[0].attention.self.query.weight.detach().to(device)
    real_bias = esm.encoder.layer[0].attention.self.query.bias.detach().to(device)
    
    INPUT_DIM = real_weight.shape[1]
    OUTPUT_DIM = real_weight.shape[0]
    print(f"   Successfully extracted Attention Query Layer: {INPUT_DIM}x{OUTPUT_DIM}")
    
except Exception as e:
    print(f"   Error loading HF Model: {e}")
    print("   Fallback to Synthetic Teacher")
    INPUT_DIM = 768
    OUTPUT_DIM = 768
    real_weight = torch.randn(OUTPUT_DIM, INPUT_DIM).to(device)
    real_bias = torch.zeros(OUTPUT_DIM).to(device)

# Teacher Wrapper
class TeacherLayer(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, OUTPUT_DIM, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
    
    def forward(self, x):
        return self.linear(x)

teacher = TeacherLayer(real_weight, real_bias).to(device)
teacher.eval() # Freeze

# --- 3. DEFINE STUDENT & SVD INITIALIZATION ---
print("\n[2] Initializing Student with SVD-Warmstart...")

class LowRankStudent(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        # W ~ B @ A 
        self.A_enc = nn.Linear(in_dim, rank, bias=False)  # Encoder
        self.B_dec = nn.Linear(rank, out_dim, bias=True)  # Decoder (+ Bias lernen)

    def forward(self, x):
        return self.B_dec(self.A_enc(x))

student = LowRankStudent(INPUT_DIM, OUTPUT_DIM, TARGET_RANK).to(device)

# SVD Initialization
with torch.no_grad():
    U, S, Vh = torch.linalg.svd(real_weight, full_matrices=False)
    
    # Truncate to Rank
    U_r = U[:, :TARGET_RANK]
    S_r = S[:TARGET_RANK]
    Vh_r = Vh[:TARGET_RANK, :]
    
    # Split Sigma for balanced initialization
    sqrt_S = torch.diag(torch.sqrt(S_r))
    
    # Init Weights (Achtung: PyTorch Linear ist transponiert)
    student.A_enc.weight.copy_(sqrt_S @ Vh_r)
    student.B_dec.weight.copy_(U_r @ sqrt_S)
    student.B_dec.bias.copy_(real_bias)

print("   Student initialized via SVD. Ready for Fine-Tuning.")

# Params Comparison
t_params = sum(p.numel() for p in teacher.parameters())
s_params = sum(p.numel() for p in student.parameters())
ratio = t_params / s_params
print(f"   Compression: {ratio:.2f}x ({t_params} -> {s_params} params)")

# --- 4. DISTILLATION LOOP ---
print(f"\n[3] Starting Distillation ({STEPS} steps)...")

optimizer = optim.AdamW(student.parameters(), lr=LR)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)

losses = []
fidelities = []

start_time = time.time()

for step in range(STEPS):
    # Input simulation (Normalverteilte Embeddings)
    x = torch.randn(BATCH_SIZE, INPUT_DIM).to(device)
    
    # Teacher Target
    with torch.no_grad():
        y_teacher = teacher(x)
    
    # Student Prediction
    y_student = student(x)
    
    # Loss calculation
    loss = criterion(y_student, y_teacher)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    losses.append(loss.item())
    
    # Metric: Attention Fidelity (Cosine Similarity)
    with torch.no_grad():
        cos_sim = torch.nn.functional.cosine_similarity(y_teacher, y_student, dim=-1).mean()
        fidelities.append(cos_sim.item())
    
    if step % 200 == 0:
        print(f"   Step {step:04d}: Loss={loss.item():.6f} | Fidelity={cos_sim.item():.4f}")

train_time = time.time() - start_time
print(f"   Training finished in {train_time:.2f}s")

# --- 5. EVALUATION & LATENCY ---
print(f"\n[4] Final Evaluation")

# A) Inferenz-Latenz Messung (CPU)
student.cpu()
teacher.cpu()
dummy_input = torch.randn(1, INPUT_DIM)

def measure_latency(model, input_data, runs=200):
    # Warmup
    for _ in range(10): _ = model(input_data)
    start = time.time()
    for _ in range(runs):
        _ = model(input_data)
    return (time.time() - start) / runs * 1000 # in ms

lat_t = measure_latency(teacher, dummy_input)
lat_s = measure_latency(student, dummy_input)
speedup = lat_t / lat_s

print(f"   Latency Teacher: {lat_t:.3f} ms")
print(f"   Latency Student: {lat_s:.3f} ms")
print(f"   Speedup: {speedup:.2f}x")

# B) Baseline Comparison
baseline_fidelity = fidelities[0]
final_fidelity = fidelities[-1]

print(f"   SVD Baseline Fidelity: {baseline_fidelity:.4f}")
print(f"   Distilled Fidelity:    {final_fidelity:.4f}")
print(f"   Improvement:           +{(final_fidelity - baseline_fidelity)*100:.2f}%")

# --- 6. PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss
ax1.plot(losses, label='MSE Loss', alpha=0.7)
ax1.set_title('Distillation Loss')
ax1.set_xlabel('Steps')
ax1.set_ylabel('MSE')
ax1.grid(True, alpha=0.3)

# Fidelity
ax2.plot(fidelities, color='green', label='Fidelity')
ax2.axhline(baseline_fidelity, color='gray', linestyle='--', label='SVD Baseline')
ax2.set_title(f'Recovery of Function (Rank {TARGET_RANK})')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Attention Fidelity (Cos Sim)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distil_rank_report.png')
print("\n[5] Results saved to 'distil_rank_report.png'")
