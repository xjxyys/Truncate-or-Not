"""MuonH + AdamW: reference parity, the fixed-radius sphere, parameter roles and checkpoint checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from looptf.model import LoopedTransformer, ModelConfig
from looptf.optim import MuonHAdamW, build_optimizer, parameter_roles
from looptf.recipe import ADAM_BETAS, BASE_LRS, LearningRateTransfer

FIXTURES = Path(__file__).parent / "fixtures"


def small_model():
    return LoopedTransformer(ModelConfig(vocab_size=19, hidden_size=256, intermediate_size=1024, num_attention_heads=2,
                                         num_prelude_layers=1, num_recurrent_layers=1, num_coda_layers=1,
                                         recurrence_steps=2, backprop_steps=2))


def test_muonh_matches_the_reference_implementation():
    """Updates recorded from muon_step_fused of the released optimizer (CPU, BF16 orthogonalization)."""
    cases = json.loads((FIXTURES / "reference_muonh.json").read_text())["cases"]
    for case in cases:
        p = nn.Parameter(torch.tensor(case["initial"]))
        opt = MuonHAdamW([{"params": [p], "role": "matrix", "lr": 0.014, "weight_decay": 0}])
        for grad in case["gradients"]:
            p.grad = torch.tensor(grad)
            opt.step()
        torch.testing.assert_close(p.detach(), torch.tensor(case["expected"]), atol=1e-6, rtol=1e-6)


def test_matrices_stay_on_their_initial_sphere_without_second_moments():
    torch.manual_seed(0)
    weight = nn.Parameter(torch.randn(16, 8))
    radius = weight.norm().item()
    opt = MuonHAdamW([{"params": [weight], "role": "matrix", "lr": 0.05, "weight_decay": 0}])
    for _ in range(5):
        weight.grad = torch.randn_like(weight)
        opt.step()
        assert weight.norm().item() == pytest.approx(radius, rel=1e-6)
    assert set(opt.state[weight]) == {"momentum_buffer", "radius"}


def test_weight_decay_is_rejected():
    with pytest.raises(ValueError, match="zero weight decay"):
        MuonHAdamW([{"params": [nn.Parameter(torch.ones(2, 2))], "role": "matrix", "lr": 0.1, "weight_decay": 0.1}])


def test_parameter_roles_of_the_looped_transformer():
    model = small_model()
    roles = parameter_roles(model)
    assert roles["embedding"] == [model.embed_tokens.weight]
    assert roles["unembedding"] == [model.lm_head.weight]
    assert not roles["other"]
    norms = [m.weight for m in model.modules() if type(m).__name__ == "RMSNorm"]
    assert {id(p) for p in roles["norm"]} == {id(p) for p in norms}
    assert any(p is model.input_write.proj.weight for p in roles["matrix"])
    assert all(p.ndim == 2 for p in roles["matrix"])
    assert sum(len(v) for v in roles.values()) == len(list(model.parameters()))


def test_groups_start_at_the_transferred_peaks():
    model = small_model()
    transfer = LearningRateTransfer(width=256, total_tokens=2**30, batch_tokens=2**18)
    opt = build_optimizer(model, BASE_LRS, transfer, ADAM_BETAS)
    peaks = transfer.peak_lrs(BASE_LRS)
    assert [g["role"] for g in opt.param_groups] == ["matrix", "embedding", "unembedding", "norm"]
    assert all(g["lr"] == g["_base_lr"] == peaks[g["role"]] and g["weight_decay"] == 0 for g in opt.param_groups)


def test_resume_requires_the_same_groups_and_peaks():
    model = small_model()
    first = build_optimizer(model, BASE_LRS, LearningRateTransfer(256, 2**30), ADAM_BETAS)
    other = build_optimizer(model, BASE_LRS, LearningRateTransfer(256, 2**31), ADAM_BETAS)
    other.load_state_dict(build_optimizer(model, BASE_LRS, LearningRateTransfer(256, 2**31), ADAM_BETAS).state_dict())
    with pytest.raises(ValueError, match="peak learning rates differ"):
        other.load_state_dict(first.state_dict())
