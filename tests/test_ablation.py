import pytest
import torch

from src.ablation import (
    head_importance_sweep,
    make_head_zero_hook,
    make_mlp_zero_hook,
    mlp_importance_sweep,
)
from src.model_utils import load_model


@pytest.fixture(scope="module")
def model():
    return load_model(device="cpu")


def test_head_zero_hook_zeroes_correct_slice():
    hook = make_head_zero_hook(head_idx=3)
    z = torch.randn(1, 4, 12, 64)
    out = hook(z.clone())
    assert torch.all(out[..., 3, :] == 0)
    assert torch.any(out[..., 0, :] != 0)
    assert torch.any(out[..., 11, :] != 0)


def test_mlp_zero_hook_zeroes_entire_output():
    hook = make_mlp_zero_hook()
    x = torch.randn(1, 5, 768)
    out = hook(x.clone())
    assert torch.all(out == 0)


def test_head_sweep_shape_and_nonzero(model):
    facts = [
        {"prompt": "The capital of France is", "correct": " Paris", "counterfactual": " Berlin"},
    ]
    imp = head_importance_sweep(model, facts, verbose=False)
    assert imp.shape == (12, 12)
    assert imp.any()


def test_mlp_sweep_shape_and_nonzero(model):
    facts = [
        {"prompt": "The capital of France is", "correct": " Paris", "counterfactual": " Berlin"},
    ]
    imp = mlp_importance_sweep(model, facts, verbose=False)
    assert imp.shape == (12,)
    assert imp.any()
