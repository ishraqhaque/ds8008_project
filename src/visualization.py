"""Plotting helpers for head and MLP importance arrays."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_head_heatmap(
    importance: np.ndarray,
    title: str = "Attention Head Importance (Δ logit diff when ablated)",
):
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = float(np.abs(importance).max()) if importance.size else 1.0
    im = ax.imshow(importance, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    ax.set_xticks(range(importance.shape[1]))
    ax.set_yticks(range(importance.shape[0]))
    fig.colorbar(im, ax=ax, label="baseline LD − ablated LD")
    fig.tight_layout()
    return fig


def plot_mlp_bars(
    importance: np.ndarray,
    title: str = "MLP Layer Importance (Δ logit diff when ablated)",
):
    fig, ax = plt.subplots(figsize=(8, 4))
    layers = np.arange(len(importance))
    ax.bar(layers, importance, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("baseline LD − ablated LD")
    ax.set_title(title)
    ax.set_xticks(layers)
    fig.tight_layout()
    return fig
