"""
This illustrates the claim: 
Longer prompts allow RoPE to accumulate more positional "phases," 
stabilizing the trajectory toward the true embedding 
(lower drift, akin to sinc alignment reconstructing the signal accurately). 
Short prompts lead to unstable rotations and higher drift 
(like phase misalignment in undersampled signals causing distortion).
AVERAGE RUN FOR DEMO PURPOUSES:
Drift with sufficient context (should be lower): 6.339801788330078
Drift with insufficient context (should be higher): 9.225069046020508
"""

import torch
import torch.nn as nn
import numpy as np

# Simple RoPE implementation (phase rotation for positions)
def apply_rope(x, positions, dim):
    # x: [seq_len, dim]
    # positions: [seq_len]
    theta = 10000 ** (-2 * (torch.arange(0, dim // 2, dtype=torch.float32) / dim))
    angles = positions[:, None] * theta[None, :]
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    
    x1 = x[:, :dim//2] * cos_angles - x[:, dim//2:] * sin_angles
    x2 = x[:, :dim//2] * sin_angles + x[:, dim//2:] * cos_angles
    return torch.cat([x1, x2], dim=-1)

# Simple Transformer-like layer (with RoPE and residual)
class SimpleLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, x, positions):
        x = apply_rope(x, positions, x.shape[-1])  # Apply RoPE rotations (phase accumulation)
        return self.linear(x) + x  # Residual connection

# Simulate trajectory: Embeddings evolving through L layers
def simulate_trajectory(prompt_emb, true_emb, L=5, dim=32):
    model = nn.ModuleList([SimpleLayer(dim) for _ in range(L)])
    
    # Initialize weights randomly (for demo; in real LLM, trained)
    for layer in model:
        nn.init.normal_(layer.linear.weight, std=0.1)
    
    positions = torch.arange(prompt_emb.shape[0], dtype=torch.float32)  # Positions for RoPE
    traj = [prompt_emb.clone()]
    
    for layer in model:
        next_emb = layer(traj[-1], positions)
        traj.append(next_emb)
    
    # Final "token" as mean of last embedding (simplified "prediction")
    final_token = traj[-1].mean(dim=0)
    
    # "Drift" as norm distance to true_emb (hallucination metric)
    drift = torch.norm(final_token - true_emb).item()
    return drift, traj

# Test the analogy
dim = 32
true_emb = torch.randn(dim)  # "True" target embedding (ground truth token)

# Sufficient context (long prompt, like Nyquist sampling: more "samples" for alignment)
long_prompt = torch.randn(10, dim)  # 10 tokens
drift_long, _ = simulate_trajectory(long_prompt, true_emb)

# Insufficient context (short prompt, like undersampling: less alignment, more drift)
short_prompt = torch.randn(2, dim)  # 2 tokens
drift_short, _ = simulate_trajectory(short_prompt, true_emb)

print("Drift with sufficient context (should be lower):", drift_long)
print("Drift with insufficient context (should be higher):", drift_short)
