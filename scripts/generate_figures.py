"""
Utility script to regenerate key figures used in the manuscript.

Usage:
    python scripts/generate_figures.py

Outputs are written to the `figures/` directory.
"""

from __future__ import annotations

import math
import pathlib
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

FIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)
plt.style.use("seaborn-v0_8")


# ---------------------------------------------------------------------------
# RoPE drift figure
# ---------------------------------------------------------------------------
def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    theta = 10000 ** (-2 * (torch.arange(0, dim // 2, dtype=torch.float32) / dim))
    angles = positions[:, None] * theta[None, :]
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    x_even, x_odd = x[:, : dim // 2], x[:, dim // 2 :]
    rotated_even = x_even * cos_angles - x_odd * sin_angles
    rotated_odd = x_even * sin_angles + x_odd * cos_angles
    return torch.cat([rotated_even, rotated_odd], dim=-1)


class SimpleLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.linear(apply_rope(x, positions)) + x


def simulate_drift(seq_len: int, *, dim: int = 32, layers: int = 6) -> float:
    model = nn.ModuleList([SimpleLayer(dim) for _ in range(layers)])
    for layer in model:
        nn.init.normal_(layer.linear.weight, std=0.1)
        nn.init.zeros_(layer.linear.bias)

    true_emb = torch.randn(dim)
    prompt = torch.randn(seq_len, dim)
    positions = torch.arange(seq_len, dtype=torch.float32)
    state = prompt.clone()
    for layer in model:
        state = layer(state, positions)
    pred = state.mean(dim=0)
    return torch.norm(pred - true_emb).item()


def run_rope_figure():
    torch.manual_seed(0)
    random.seed(0)
    short = [simulate_drift(2) for _ in range(64)]
    long = [simulate_drift(12) for _ in range(64)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(short, bins=15, alpha=0.6, label="Short prompt", color="#d95f02")
    ax.hist(long, bins=15, alpha=0.6, label="Long prompt", color="#1b9e77")
    ax.set_xlabel("Euclidean drift")
    ax.set_ylabel("Frequency")
    ax.set_title("RoPE drift vs. prompt length")
    ax.legend()
    fig.savefig(FIG_DIR / "rope_drift.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Nyquist reconstruction figure
# ---------------------------------------------------------------------------
def generate_signal(t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 3.0 * t)


def reconstruct_signal(t_dense, t_samples, samples, T):
    x = (t_dense[:, None] - t_samples[None, :]) / T
    return np.sum(samples[None, :] * np.sinc(x), axis=1)


def run_sampling_figure():
    t = np.linspace(0, 1, 1000)
    original = generate_signal(t)
    f_max = 5
    fs_nyquist = 2 * f_max
    fs_under = 1.5 * f_max
    T_nyquist = 1 / fs_nyquist
    T_under = 1 / fs_under

    t_ny = np.arange(0, 1 + T_nyquist, T_nyquist)
    t_under = np.arange(0, 1 + T_under, T_under)
    recon_ny = reconstruct_signal(t, t_ny, generate_signal(t_ny), T_nyquist)
    recon_under = reconstruct_signal(t, t_under, generate_signal(t_under), T_under)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, original, color="black", label="Original")
    axes[0].plot(t, recon_ny, "--", color="#1b9e77", label="Nyquist")
    axes[0].set_title("Nyquist-sampled reconstruction")
    axes[0].legend()

    axes[1].plot(t, original, color="black", label="Original")
    axes[1].plot(t, recon_under, "--", color="#d95f02", label="Undersampled")
    axes[1].set_title("Undersampled reconstruction")
    axes[1].legend()
    axes[1].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "nyquist_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Semantic Nyquist heatmap
# ---------------------------------------------------------------------------
def basis_matrix(t: np.ndarray, complexity: int) -> np.ndarray:
    cols = []
    for k in range(1, complexity + 1):
        cols.append(np.sin(2 * np.pi * k * t))
        cols.append(np.cos(2 * np.pi * k * t))
    return np.stack(cols, axis=1)


def simulate_error(complexity: int, sample_count: int, trials: int = 200, noise: float = 0.01) -> float:
    errors: List[float] = []
    for _ in range(trials):
        coeffs = np.random.randn(2 * complexity)
        dense_t = np.linspace(0, 1, 200)
        dense_basis = basis_matrix(dense_t, complexity)
        concept = dense_basis @ coeffs

        sample_t = np.linspace(0, 1, sample_count)
        sample_basis = basis_matrix(sample_t, complexity)
        observations = sample_basis @ coeffs + noise * np.random.randn(sample_count)
        recon_coeffs, *_ = np.linalg.lstsq(sample_basis, observations, rcond=None)
        recon = dense_basis @ recon_coeffs
        errors.append(np.mean((concept - recon) ** 2))
    return float(np.mean(errors))


def run_threshold_figure():
    complexities = np.arange(1, 6)
    sample_counts = np.arange(2, 18)
    error_matrix = np.zeros((len(complexities), len(sample_counts)))
    for i, comp in enumerate(complexities):
        for j, samples in enumerate(sample_counts):
            error_matrix[i, j] = simulate_error(int(comp), int(samples))

    fig, ax = plt.subplots(figsize=(8, 4))
    mesh = ax.imshow(
        np.log10(error_matrix + 1e-8),
        aspect="auto",
        origin="lower",
        extent=[sample_counts[0], sample_counts[-1], complexities[0], complexities[-1]],
        cmap="viridis",
    )
    ax.set_xlabel("Prompt samples (tokens)")
    ax.set_ylabel("Concept complexity (Fourier modes)")
    ax.set_title("Semantic Nyquist threshold (log10 MSE)")
    ax.axline((0, 0), slope=0.5, color="white", linestyle="--", label="Nyquist boundary")
    fig.colorbar(mesh, ax=ax, label="log10 MSE")
    ax.legend(loc="upper right")
    fig.savefig(FIG_DIR / "semantic_threshold.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def simulate_prompt_noise(seq_len: int, noise_ratio: float, dim: int = 64, trials: int = 200) -> float:
    errors: List[float] = []
    for _ in range(trials):
        concept = np.random.randn(dim)
        concept /= np.linalg.norm(concept)
        informative = int(round(seq_len * (1 - noise_ratio)))
        informative = max(informative, 0)
        noisy = max(seq_len - informative, 0)
        tokens = []
        if informative > 0:
            info = concept + 0.1 * np.random.randn(informative, dim)
            tokens.append(info)
        if noisy > 0:
            noise = np.random.randn(noisy, dim)
            noise /= np.linalg.norm(noise, axis=1, keepdims=True)
            tokens.append(noise)
        prompt = np.vstack(tokens)
        reconstruction = prompt.mean(axis=0)
        reconstruction /= np.linalg.norm(reconstruction) + 1e-8
        errors.append(np.linalg.norm(reconstruction - concept) ** 2)
    return float(np.mean(errors))


def run_prompt_noise_figure():
    lengths = np.arange(2, 33)
    noise_levels = np.linspace(0, 0.8, 17)
    heatmap = np.zeros((len(noise_levels), len(lengths)))

    for i, noise in enumerate(noise_levels):
        for j, length in enumerate(lengths):
            heatmap[i, j] = simulate_prompt_noise(length, noise)

    fig, ax = plt.subplots(figsize=(8, 4))
    mesh = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=[lengths[0], lengths[-1], noise_levels[0], noise_levels[-1]],
        cmap="viridis",
    )
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Noise ratio (1 - ρ)")
    ax.set_title("Error vs. prompt length and semantic noise")
    fig.colorbar(mesh, ax=ax, label="Mean squared error")
    fig.savefig(FIG_DIR / "prompt_noise_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    run_rope_figure()
    run_sampling_figure()
    run_threshold_figure()
    run_prompt_noise_figure()
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()

