"""Zero-ablation hooks and sweep loops for attention heads and MLP layers."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from transformer_lens import HookedTransformer

from src.model_utils import Fact, _single_token_id, logit_diff


def make_head_zero_hook(head_idx: int) -> Callable:
    """Zero the slice of attention `z` corresponding to one head.

    z shape: [batch, seq, n_heads, d_head].
    """
    def hook_fn(z: torch.Tensor, hook=None) -> torch.Tensor:
        z[..., head_idx, :] = 0.0
        return z
    return hook_fn


def make_mlp_zero_hook() -> Callable:
    """Zero an MLP layer's output. Shape: [batch, seq, d_model]."""
    def hook_fn(mlp_out: torch.Tensor, hook=None) -> torch.Tensor:
        mlp_out[...] = 0.0
        return mlp_out
    return hook_fn


@torch.no_grad()
def _ablated_logit_diff(
    model: HookedTransformer,
    prompt: str,
    correct: str,
    counterfactual: str,
    hook_name: str,
    hook_fn: Callable,
) -> float:
    correct_id = _single_token_id(model, correct)
    cf_id = _single_token_id(model, counterfactual)
    tokens = model.to_tokens(prompt)
    logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    final = logits[0, -1]
    return float((final[correct_id] - final[cf_id]).item())


@torch.no_grad()
def head_importance_sweep(
    model: HookedTransformer,
    facts: list[Fact],
    verbose: bool = True,
) -> np.ndarray:
    """Mean importance per (layer, head): baseline_LD - ablated_LD averaged over facts."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    importance = np.zeros((n_layers, n_heads), dtype=np.float32)

    baselines = [
        logit_diff(model, f["prompt"], f["correct"], f["counterfactual"]).item()
        for f in facts
    ]

    for L in range(n_layers):
        if verbose:
            print(f"Head sweep: layer {L}/{n_layers - 1}")
        for H in range(n_heads):
            hook_fn = make_head_zero_hook(H)
            hook_name = f"blocks.{L}.attn.hook_z"
            drops = [
                baseline - _ablated_logit_diff(
                    model, f["prompt"], f["correct"], f["counterfactual"],
                    hook_name, hook_fn,
                )
                for f, baseline in zip(facts, baselines)
            ]
            importance[L, H] = float(np.mean(drops))
    return importance


@torch.no_grad()
def mlp_importance_sweep(
    model: HookedTransformer,
    facts: list[Fact],
    verbose: bool = True,
) -> np.ndarray:
    """Mean importance per MLP layer."""
    n_layers = model.cfg.n_layers
    importance = np.zeros(n_layers, dtype=np.float32)

    baselines = [
        logit_diff(model, f["prompt"], f["correct"], f["counterfactual"]).item()
        for f in facts
    ]

    hook_fn = make_mlp_zero_hook()
    for L in range(n_layers):
        if verbose:
            print(f"MLP sweep: layer {L}/{n_layers - 1}")
        hook_name = f"blocks.{L}.hook_mlp_out"
        drops = [
            baseline - _ablated_logit_diff(
                model, f["prompt"], f["correct"], f["counterfactual"],
                hook_name, hook_fn,
            )
            for f, baseline in zip(facts, baselines)
        ]
        importance[L] = float(np.mean(drops))
    return importance
