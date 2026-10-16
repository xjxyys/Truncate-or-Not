"""Parameter roles and optimizer construction."""

from __future__ import annotations

from torch import nn

from looptf.recipe import ADAM_EPS, ROLES, LearningRateTransfer

from .muonh import MuonHAdamW


def parameter_roles(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Split trainable parameters into matrix (MuonH), embedding, unembedding, norm and other (AdamW) roles."""
    embedding_ids, head_ids, norm_ids = set(), set(), set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1].lower()
        if isinstance(module, nn.Embedding) or leaf == "embed_tokens":
            embedding_ids.update(id(p) for p in module.parameters(recurse=False))
        if leaf == "lm_head":
            head_ids.update(id(p) for p in module.parameters(recurse=False))
        if "norm" in type(module).__name__.lower():
            norm_ids.update(id(p) for p in module.parameters(recurse=False) if p.ndim <= 1)
        for getter, identities in (("get_input_embeddings", embedding_ids), ("get_output_embeddings", head_ids)):
            fn = getattr(module, getter, None)
            if callable(fn):
                try:
                    layer = fn()
                except NotImplementedError:
                    continue
                if isinstance(layer, nn.Module):
                    identities.update(id(p) for p in layer.parameters())
    roles = {role: [] for role in ROLES}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            role = "unembedding"
        elif id(p) in embedding_ids:
            role = "embedding"
        elif id(p) in norm_ids or (p.ndim <= 1 and "norm" in name.lower()):
            role = "norm"
        elif p.ndim == 2:
            role = "matrix"
        else:
            role = "other"
        roles[role].append(p)
    return roles


def build_optimizer(model: nn.Module, base_lrs: dict[str, float], transfer: LearningRateTransfer,
                    betas: tuple[float, float]) -> MuonHAdamW:
    """One group per non-empty role at its transferred peak learning rate; the schedule scales `lr` later."""
    roles = parameter_roles(model)
    peaks = transfer.peak_lrs(base_lrs)
    names = {id(p): name for name, p in model.named_parameters()}
    groups = [
        {
            "params": params,
            "role": role,
            "lr": peaks[role],
            "_base_lr": peaks[role],
            "param_names": [names[id(p)] for p in params],
            "weight_decay": 0.0,
            "_base_weight_decay": 0.0,
            "betas": betas,
            "eps": ADAM_EPS,
        }
        for role, params in roles.items()
        if params
    ]
    return MuonHAdamW(groups)
