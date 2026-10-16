"""Looped transformer: reference parity, loop schedule, truncated BPTT, untied loops and parameter accounting."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from looptf.model import LoopedTransformer, ModelConfig, parameter_counts
from looptf.model.losses import token_cross_entropy

FIXTURES = Path(__file__).parent / "fixtures"


def config(**overrides):
    values = dict(vocab_size=19, hidden_size=256, intermediate_size=1024, num_attention_heads=2, head_dim=128,
                  max_position_embeddings=16, num_prelude_layers=1, num_recurrent_layers=2, num_coda_layers=1,
                  recurrence_steps=3, backprop_steps=2)
    return ModelConfig(**(values | overrides))


def canonical_weight(name, shape):
    """Deterministic weights of the reference fixture (tests/generate_reference_fixture.py)."""
    generator = torch.Generator().manual_seed(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    x = torch.randn(shape, generator=generator)
    if len(shape) == 1:
        return 1 + 0.05 * x
    return x * (0.8 if name == "transformer.wte.weight" else shape[-1] ** -0.5)


def reference_weights(width=256, layers=2, padded_vocab=64):
    shapes = {"transformer.wte.weight": (padded_vocab, width), "lm_head.weight": (padded_vocab, width),
              **{f"norm_{name}.weight": (width,) for name in ("emb", "recur", "final")}}
    for i in range(layers):
        prefix = f"transformer.recur.{i}."
        shapes.update({prefix + f"attn.c_{name}.weight": (width, width) for name in ("q", "k", "v", "proj")})
        shapes.update({prefix + "mlp.c_fc.weight": (4 * width, width), prefix + "mlp.c_proj.weight": (width, 4 * width),
                       prefix + "n1.weight": (width,), prefix + "n3.weight": (width,)})
    return {name: canonical_weight(name, shape) for name, shape in shapes.items()}


def reference_names(model):
    """Parameter of `model` -> name in the reference implementation."""
    names = {model.embed_tokens.weight: "transformer.wte.weight", model.lm_head.weight: "lm_head.weight",
             model.norm_emb.weight: "norm_emb.weight", model.norm_recur.weight: "norm_recur.weight",
             model.final_norm.weight: "norm_final.weight"}
    for i, block in enumerate(model.recurrent_core):
        prefix = f"transformer.recur.{i}."
        for ours, theirs in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "proj")):
            names[getattr(block.attn, ours).weight] = prefix + f"attn.c_{theirs}.weight"
        names.update({block.mlp.up_proj.weight: prefix + "mlp.c_fc.weight",
                      block.mlp.down_proj.weight: prefix + "mlp.c_proj.weight",
                      block.attn_norm.weight: prefix + "n1.weight", block.mlp_norm.weight: prefix + "n3.weight"})
    return names


def test_matches_the_reference_implementation():
    """A non-looped two-layer model against logits, loss and gradients recorded from the authors' released GPT.

    With one loop the injection [I | 0] passes the prelude output through unchanged, as in the reference.
    """
    fixture = json.loads((FIXTURES / "reference_block.json").read_text())
    model = LoopedTransformer(config(num_prelude_layers=0, num_coda_layers=0, recurrence_steps=1, backprop_steps=1))
    names, weights = reference_names(model), reference_weights()
    with torch.no_grad():
        for param, name in names.items():
            param.copy_(weights[name])
    ids, labels = torch.tensor(fixture["ids"]), torch.tensor(fixture["labels"])
    logits = model(ids).logits
    torch.testing.assert_close(logits, torch.tensor(fixture["logits"]), atol=2e-5, rtol=2e-4)
    loss = token_cross_entropy(logits, labels).mean()
    assert loss.item() == pytest.approx(fixture["loss"], rel=1e-5)
    loss.backward()
    for param, name in names.items():
        grad = param.grad.flatten()
        expected = fixture["gradients"][name]
        assert grad.norm().item() == pytest.approx(expected["norm"], rel=1e-4), name
        samples = grad[torch.linspace(0, grad.numel() - 1, 12).long()]
        torch.testing.assert_close(samples, torch.tensor(expected["samples"]), atol=1e-6, rtol=1e-3, msg=name)


def test_loops_reuse_the_recurrent_block_and_truncation_stops_gradients():
    torch.manual_seed(0)
    model = LoopedTransformer(config(recurrence_steps=3, backprop_steps=1))
    visits = []
    for block in [*model.prelude, *model.recurrent_core, *model.coda]:
        block.register_forward_hook(lambda module, *_: visits.append(module))
    ids = torch.tensor([[1, 5, 3, 8]])
    model(ids, labels=ids).loss.backward()
    assert visits == [model.prelude[0], *list(model.recurrent_core) * 3, model.coda[0]]
    # The prelude feeds every loop through the injection, so it is trained although two loops are truncated.
    assert model.prelude[0].attn.q_proj.weight.grad.norm() > 0
    assert all(p.grad is not None for p in model.parameters())


def test_truncated_loops_do_not_change_the_forward_pass():
    torch.manual_seed(0)
    model = LoopedTransformer(config(recurrence_steps=4))
    ids = torch.randint(0, 19, (2, 8))
    full = model(ids, num_steps=4, backprop_steps=4).logits
    truncated = model(ids, num_steps=4, backprop_steps=1).logits
    torch.testing.assert_close(full, truncated, atol=0, rtol=0)


def test_activation_checkpointing_gives_the_same_gradients():
    torch.manual_seed(0)
    model = LoopedTransformer(config())
    checked = copy.deepcopy(model)
    checked.config.gradient_checkpointing = True
    ids = torch.tensor([[1, 5, 3, 8, 2]])
    model(ids, labels=ids).loss.backward()
    checked(ids, labels=ids).loss.backward()
    for (name, p), q in zip(model.named_parameters(), checked.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)


def test_evaluation_runs_all_loops_without_gradients_by_default():
    torch.manual_seed(0)
    model = LoopedTransformer(config(recurrence_steps=3, backprop_steps=1)).eval()
    assert model._schedule(None, None).no_grad_steps == 3
    model.train()
    assert (model._schedule(None, None).no_grad_steps, model._schedule(None, None).grad_steps) == (2, 1)


def test_causal_softcapped_logits_over_the_active_vocabulary():
    model = LoopedTransformer(config()).eval()
    first = model(torch.tensor([[1, 2, 3, 4]])).logits
    second = model(torch.tensor([[1, 2, 10, 12]])).logits
    assert first.shape == (1, 4, 19) and model.lm_head.weight.shape[0] == 64
    assert first.abs().max() <= 15
    torch.testing.assert_close(first[:, :2], second[:, :2], atol=0, rtol=0)


def test_initialization():
    torch.manual_seed(0)
    model = LoopedTransformer(config())
    width = 256
    injection = model.input_write.proj.weight
    assert torch.equal(injection[:, :width], torch.eye(width)) and not injection[:, width:].any()
    assert all(torch.equal(m.weight, torch.ones(width)) for m in (model.norm_emb, model.norm_recur, model.final_norm))
    bound = (3 / width) ** 0.5
    assert model.recurrent_core[0].attn.q_proj.weight.abs().max() <= bound
    assert model.embed_tokens.weight.std().item() == pytest.approx(1.0, rel=0.05)
    assert model.lm_head.weight.std().item() == pytest.approx(0.001, rel=0.05)
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_untied_loops_run_one_copy_per_loop():
    torch.manual_seed(0)
    model = LoopedTransformer(config(recurrence_steps=3, backprop_steps=3, untie_loops=True))
    tied = LoopedTransformer(config(recurrence_steps=3, backprop_steps=3))
    per_copy = sum(p.numel() for m in (model.recurrent_core, model.input_write, model.norm_recur)
                   for p in m.parameters())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in tied.parameters()) + 2 * per_copy
    assert not torch.equal(model.untied_recurrent_cores[0][0].attn.q_proj.weight,
                           model.recurrent_core[0].attn.q_proj.weight)
    visits = []
    for core in (model.recurrent_core, *model.untied_recurrent_cores):
        core[0].register_forward_hook(lambda module, *_: visits.append(module))
    # At initialization ([I | 0]) every injection ignores the previous state, so only the last loop would receive
    # a gradient; give the state half some weight, as after the first updates.
    with torch.no_grad():
        for write in (model.input_write, *model.untied_input_writes):
            write.proj.weight[:, 256:].normal_(std=0.01)
    ids = torch.tensor([[1, 2, 3]])
    model(ids, labels=ids).loss.backward()
    assert visits == [model.recurrent_core[0], *(core[0] for core in model.untied_recurrent_cores)]
    assert all(p.grad is not None and p.grad.norm() > 0 for p in model.untied_recurrent_cores.parameters())
    with pytest.raises(ValueError, match="all recurrence_steps"):
        model(ids, num_steps=2)


def test_parameter_counts_of_the_60m_model():
    with torch.device("meta"):
        model = LoopedTransformer(ModelConfig(hidden_size=768, intermediate_size=3072, num_attention_heads=6))
    counts = parameter_counts(768, 8)
    assert sum(p.numel() for p in model.parameters()) == counts["total"] + 2 * 768**2
    assert counts["non_embedding"] == 56_637_696
    assert not any(name.endswith("bias") for name, _ in model.named_parameters())


@pytest.mark.parametrize("overrides", [{"intermediate_size": 768}, {"rms_norm_eps": 1e-5}, {"rope_theta": 0},
                                       {"head_dim": 64}, {"untie_loops": True, "recurrence_steps": 1}])
def test_invalid_configurations_are_rejected(overrides):
    with pytest.raises(ValueError):
        config(**overrides)
