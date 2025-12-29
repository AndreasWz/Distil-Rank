import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cpu")
SEED = 42

# Hyperparameters
TARGET_RANK = 64
BATCH_SIZE = 64
EVAL_BATCH = 32
STEPS = 1000
LR = 3e-3

# File paths
EMB_TRAIN_PATH = "embeddings_train.pt"
EMB_VAL_PATH = "embeddings_val.pt"
OUT_DIR = "results"


class LowRankStudent(nn.Module):
    """Factorized low-rank approximation: W ≈ B @ A"""
    def __init__(self, input_dim, rank, output_dim):
        super().__init__()
        self.A = nn.Linear(input_dim, rank, bias=False)
        self.B = nn.Linear(rank, output_dim, bias=True)
    
    def forward(self, x):
        return self.B(self.A(x))


def init_teacher_random(teacher):
    """Default PyTorch Kaiming initialization"""
    nn.init.kaiming_uniform_(teacher.weight, a=np.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(teacher.weight)
    bound = 1 / np.sqrt(fan_in)
    nn.init.uniform_(teacher.bias, -bound, bound)


def init_teacher_task_trained(teacher, embeddings, steps=500):
    """
    Simulate a task-trained head by training on a proxy objective.
    This creates realistic weight structure without being circular like PCA.
    The teacher learns to predict a noisy reconstruction - mimicking a 
    contrastive or denoising objective common in protein ML.
    """
    teacher.train()
    optimizer = optim.Adam(teacher.parameters(), lr=1e-3)
    
    for step in range(steps):
        idx = torch.randint(0, len(embeddings), (64,))
        x = embeddings[idx]
        # Proxy task: denoise / reconstruct with structure
        noise = torch.randn_like(x) * 0.3
        x_noisy = x + noise
        y_target = x  # Learn to denoise
        
        y_pred = teacher(x_noisy)
        loss = nn.functional.mse_loss(y_pred, y_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    teacher.eval()


def init_teacher_spectral_decay(teacher, input_dim, output_dim):
    """
    Spectral decay: Simulates structure of trained layers.
    Real neural network weights often have rapidly decaying singular values.
    """
    with torch.no_grad():
        U, _ = torch.linalg.qr(torch.randn(output_dim, output_dim))
        V, _ = torch.linalg.qr(torch.randn(input_dim, input_dim))
        rank = min(input_dim, output_dim)
        # Exponential decay - mimics learned compression
        S = torch.exp(-torch.arange(rank).float() / 100) * 2.0
        W = U @ (S.unsqueeze(1) * V[:rank, :])
        teacher.weight.copy_(W)
        teacher.bias.zero_()


def compute_svd_baseline(teacher, x_val, target_rank):
    """Compute truncated SVD approximation and its fidelity."""
    W = teacher.weight.data
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    U_r = U[:, :target_rank]
    S_r = S[:target_rank]
    Vh_r = Vh[:target_rank, :]
    W_svd = U_r @ torch.diag(S_r) @ Vh_r
    
    with torch.no_grad():
        y_true = teacher(x_val)
        y_svd = torch.nn.functional.linear(x_val, W_svd, teacher.bias)
        
        cos_fidelity = torch.nn.functional.cosine_similarity(y_true, y_svd).mean().item()
        mse = torch.nn.functional.mse_loss(y_svd, y_true).item()
        rel_mse = mse / (y_true.pow(2).mean().item() + 1e-8)
    
    return {
        "U_r": U_r, "S_r": S_r, "Vh_r": Vh_r,
        "cosine": cos_fidelity,
        "mse": mse,
        "rel_mse": rel_mse,
    }


def train_student(student, teacher, get_batch, steps, lr):
    """Distill teacher into low-rank student."""
    optimizer = optim.AdamW(student.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()
    history = []
    
    start_time = time.time()
    for step in range(steps):
        x = get_batch(BATCH_SIZE)
        
        with torch.no_grad():
            y_teacher = teacher(x)
        
        y_student = student(x)
        
        # Combined loss: magnitude (MSE) + direction (cosine)
        loss_mse = mse_loss_fn(y_student, y_teacher)
        cos_sim = torch.nn.functional.cosine_similarity(y_student, y_teacher)
        loss_cos = (1.0 - cos_sim).mean()
        total_loss = loss_mse + 0.1 * loss_cos
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        history.append(total_loss.item())
        
        if step % 200 == 0:
            print(f"   Step {step}: Loss = {total_loss.item():.5f}")
    
    train_time = time.time() - start_time
    print(f"   Training finished in {train_time:.2f}s")
    
    return history


def evaluate_model(model, teacher, x_val):
    """Compute fidelity metrics for a model vs teacher."""
    with torch.no_grad():
        y_pred = model(x_val)
        y_true = teacher(x_val)
        
        cos_fidelity = torch.nn.functional.cosine_similarity(y_pred, y_true).mean().item()
        mse = torch.nn.functional.mse_loss(y_pred, y_true).item()
        rel_mse = mse / (y_true.pow(2).mean().item() + 1e-8)
    
    return {"cosine": cos_fidelity, "mse": mse, "rel_mse": rel_mse}


def measure_latency(model, input_dim, batch_size=32, iterations=300):
    """Measure inference latency in milliseconds."""
    dummy = torch.randn(batch_size, input_dim).to(DEVICE)
    # Warmup
    for _ in range(10):
        model(dummy)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        model(dummy)
    t1 = time.perf_counter()
    
    return (t1 - t0) / iterations * 1000


def run_experiment(name, teacher, x_val, get_batch, input_dim, output_dim):
    """Run full distillation experiment for one teacher initialization."""
    print(f"\n{'='*50}")
    print(f"EXPERIMENT: {name}")
    print('='*50)
    
    # Freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    # --- SVD Baseline ---
    print("\n[Baseline] Calculating SVD Reference...")
    svd_result = compute_svd_baseline(teacher, x_val, TARGET_RANK)
    print(f"   SVD Cosine Fidelity: {svd_result['cosine']:.4f}")
    print(f"   SVD Relative MSE:    {svd_result['rel_mse']:.4f}")
    
    # --- Student with SVD Warmstart ---
    print("\n[Student] Initializing with SVD Warmstart...")
    student = LowRankStudent(input_dim, TARGET_RANK, output_dim).to(DEVICE)
    
    sqrt_S = torch.diag(torch.sqrt(svd_result['S_r']))
    with torch.no_grad():
        student.A.weight.copy_(sqrt_S @ svd_result['Vh_r'])
        student.B.weight.copy_(svd_result['U_r'] @ sqrt_S)
        student.B.bias.copy_(teacher.bias)
    
    # --- Training ---
    print(f"\n[Training] Distilling for {STEPS} steps...")
    history = train_student(student, teacher, get_batch, STEPS, LR)
    
    # --- Evaluation ---
    print("\n[Evaluation] Measuring Performance...")
    student_metrics = evaluate_model(student, teacher, x_val)
    
    lat_t = measure_latency(teacher, input_dim, EVAL_BATCH)
    lat_s = measure_latency(student, input_dim, EVAL_BATCH)
    
    params_t = sum(p.numel() for p in teacher.parameters())
    params_s = sum(p.numel() for p in student.parameters())
    
    print("-" * 45)
    print(f"{'Metric':<25} {'SVD':>8} {'Student':>8}")
    print("-" * 45)
    print(f"{'Cosine Fidelity':<25} {svd_result['cosine']:>8.4f} {student_metrics['cosine']:>8.4f}")
    print(f"{'Relative MSE':<25} {svd_result['rel_mse']:>8.4f} {student_metrics['rel_mse']:>8.4f}")
    print("-" * 45)
    print(f"Speedup: {lat_t/lat_s:.1f}x | Compression: {params_t/params_s:.1f}x")
    print("-" * 45)
    
    # Get full singular value spectrum for analysis
    W = teacher.weight.data
    _, S_full, _ = torch.linalg.svd(W, full_matrices=False)
    
    return {
        "svd_cosine": svd_result['cosine'],
        "svd_rel_mse": svd_result['rel_mse'],
        "student_cosine": student_metrics['cosine'],
        "student_rel_mse": student_metrics['rel_mse'],
        "cosine_improvement": student_metrics['cosine'] - svd_result['cosine'],
        "mse_reduction": (svd_result['rel_mse'] - student_metrics['rel_mse']) / svd_result['rel_mse'] if svd_result['rel_mse'] > 0 else 0,
        "latency_teacher_ms": lat_t,
        "latency_student_ms": lat_s,
        "speedup": lat_t / lat_s,
        "params_teacher": params_t,
        "params_student": params_s,
        "compression": params_t / params_s,
        "history": history,
        "singular_values": S_full.numpy(),
    }


def main():
    print("=== DISTIL-RANK: ROBUST CPU VERSION ===")
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # --- Load Data ---
    use_real_data = os.path.exists(EMB_TRAIN_PATH) and os.path.exists(EMB_VAL_PATH)
    
    if use_real_data:
        print("[Data] Loading REAL ESM-2 embeddings...")
        embeddings_train = torch.load(EMB_TRAIN_PATH, weights_only=False).to(DEVICE)
        embeddings_val = torch.load(EMB_VAL_PATH, weights_only=False).to(DEVICE)
        input_dim = embeddings_train.shape[1]
        
        def get_batch(bs):
            idx = torch.randint(0, len(embeddings_train), (bs,))
            return embeddings_train[idx]
        
        x_val = embeddings_val[:512]
    else:
        print("[Data] Fallback to SYNTHETIC Structured Data...")
        input_dim = 1280
        embeddings_train = None
        importance_mask = torch.ones(input_dim).to(DEVICE)
        importance_mask[:100] *= 5.0
        
        def get_batch(bs):
            return torch.randn(bs, input_dim).to(DEVICE) * importance_mask
        
        x_val = get_batch(512)
    
    output_dim = input_dim
    print(f"   Dimensions: {input_dim} -> {output_dim}")
    print(f"   Target Rank: {TARGET_RANK}")
    
    # --- Define Experiments ---
    # We test three realistic scenarios:
    # 1. Random: Baseline - what if teacher is arbitrary?
    # 2. Task-Trained: Realistic - teacher learned a proxy task (most relevant!)
    # 3. Spectral Decay: Structured - simulates learned layer statistics
    
    all_results = {}
    
    # Experiment 1: Random Teacher
    torch.manual_seed(SEED)
    teacher = nn.Linear(input_dim, output_dim, bias=True).to(DEVICE)
    init_teacher_random(teacher)
    all_results["Random"] = run_experiment("Random Teacher", teacher, x_val, get_batch, input_dim, output_dim)
    
    # Experiment 2: Task-Trained Teacher (most realistic!)
    torch.manual_seed(SEED)
    teacher = nn.Linear(input_dim, output_dim, bias=True).to(DEVICE)
    if embeddings_train is not None:
        print("\n[Pre-training] Training teacher on proxy denoising task...")
        init_teacher_task_trained(teacher, embeddings_train, steps=500)
    else:
        print("\n[Pre-training] No real data, using random init for task-trained")
        init_teacher_random(teacher)
    all_results["Task-Trained"] = run_experiment("Task-Trained Teacher", teacher, x_val, get_batch, input_dim, output_dim)
    
    # Experiment 3: Spectral Decay Teacher
    torch.manual_seed(SEED)
    teacher = nn.Linear(input_dim, output_dim, bias=True).to(DEVICE)
    init_teacher_spectral_decay(teacher, input_dim, output_dim)
    all_results["Spectral"] = run_experiment("Spectral Decay Teacher", teacher, x_val, get_batch, input_dim, output_dim)
    
    # --- Final Summary ---
    print("\n" + "="*70)
    print("FINAL COMPARISON: Teacher Initialization Strategies")
    print("="*70)
    print(f"{'Strategy':<15} {'SVD Cos':>10} {'Student Cos':>12} {'Improvement':>12} {'MSE Reduction':>14}")
    print("-"*70)
    for name, res in all_results.items():
        print(f"{name:<15} {res['svd_cosine']:>10.4f} {res['student_cosine']:>12.4f} "
              f"{res['cosine_improvement']*100:>+11.1f}% {res['mse_reduction']*100:>13.1f}%")
    print("="*70)
    
    # --- Save Results ---
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Exclude non-serializable fields (history, singular_values)
    results_json = {
        "real_data": use_real_data,
        "target_rank": TARGET_RANK,
        "input_dim": input_dim,
        "experiments": {k: {kk: vv for kk, vv in v.items() 
                           if kk not in ("history", "singular_values")} 
                       for k, v in all_results.items()},
    }
    with open(os.path.join(OUT_DIR, "distil_rank_summary.json"), "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}/distil_rank_summary.json")
    
    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'Random': '#e74c3c', 'Task-Trained': '#9b59b6', 'Spectral': '#2ecc71'}
    
    # Panel 1: Training Loss Curves
    for name, res in all_results.items():
        axes[0].plot(res['history'], label=name, color=colors[name], linewidth=1.5, alpha=0.8)
    axes[0].set_title("Training Loss Curves")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Cosine Fidelity Comparison (side-by-side bars)
    strategies = list(all_results.keys())
    x_pos = np.arange(len(strategies))
    width = 0.35
    
    svd_cos = [all_results[s]['svd_cosine'] for s in strategies]
    stu_cos = [all_results[s]['student_cosine'] for s in strategies]
    
    axes[1].bar(x_pos - width/2, svd_cos, width, label='SVD Baseline', color='gray', alpha=0.7)
    axes[1].bar(x_pos + width/2, stu_cos, width, label='Distil-Rank', color=[colors[s] for s in strategies])
    axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(strategies)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("Cosine Fidelity (↑ better)")
    axes[1].set_title("Fidelity: SVD vs Distil-Rank")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Singular Value Spectrum (explains WHY results differ)
    for name in strategies:
        S = all_results[name]['singular_values']
        # Normalize to show relative decay
        S_norm = S / S[0]
        axes[2].plot(S_norm[:200], label=name, color=colors[name], linewidth=1.5, alpha=0.8)
    
    # Mark the truncation point
    axes[2].axvline(x=TARGET_RANK, color='black', linestyle='--', alpha=0.7, 
                    label=f'Rank-{TARGET_RANK} cutoff')
    axes[2].set_xlabel("Singular Value Index")
    axes[2].set_ylabel("Normalized σᵢ / σ₁")
    axes[2].set_title("Weight Matrix Spectrum")
    axes[2].set_yscale('log')
    axes[2].set_ylim(1e-4, 1.5)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "distil_rank_final_report.png"), dpi=150)
    print(f"Plot saved to {OUT_DIR}/distil_rank_final_report.png")


if __name__ == "__main__":
    main()
