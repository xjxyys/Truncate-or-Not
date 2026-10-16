"""Import Megatron-LM from a source checkout and patch the few places where it does not fit this environment.

Training uses Megatron-LM at commit ed6ac270ba as a source checkout (--megatron-lm-path or MEGATRON_LM_PATH).
`ensure_megatron` binds the package name `megatron` to that checkout before anything imports it, so a `megatron`
package installed elsewhere (some NGC PyTorch images ship one) cannot shadow it, and applies two import-time
patches: an alias for torch.float8_e8m0fnu on PyTorch builds that lack it (Megatron's inference-only MXFP8 code
references it at import time; training never uses it), and a re-export of Transformer Engine's attention classes
on builds that do not export them from transformer_engine.pytorch.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any


def _patch_torch() -> None:
    import torch

    if not hasattr(torch, "float8_e8m0fnu"):
        # MXFP8 scales are stored as bytes in the only module that references this dtype.
        torch.float8_e8m0fnu = torch.uint8


def _patch_transformer_engine() -> None:
    try:
        import transformer_engine.pytorch as te
    except ImportError:
        return
    if hasattr(te, "DotProductAttention"):
        return
    from transformer_engine.pytorch.attention import (
        DotProductAttention,
        InferenceParams,
        MultiheadAttention,
        RotaryPositionEmbedding,
    )

    te.DotProductAttention = DotProductAttention
    te.MultiheadAttention = MultiheadAttention
    te.InferenceParams = InferenceParams
    te.RotaryPositionEmbedding = RotaryPositionEmbedding


def ensure_megatron(path: str | None = None) -> Path:
    """Make `import megatron` resolve to the Megatron-LM checkout at `path` (default: MEGATRON_LM_PATH)."""
    candidate = path or os.environ.get("MEGATRON_LM_PATH")
    if not candidate:
        raise RuntimeError("set --megatron-lm-path or MEGATRON_LM_PATH to a Megatron-LM checkout (commit ed6ac270ba)")
    root = Path(candidate).expanduser().resolve()
    package_dir = root / "megatron"
    if not (package_dir / "training" / "__init__.py").is_file():
        raise RuntimeError(f"{root} is not a Megatron-LM source checkout: megatron/training/__init__.py is missing")
    current = sys.modules.get("megatron")
    if current is not None:
        if str(package_dir) not in list(getattr(current, "__path__", [])):
            raise RuntimeError(f"`megatron` was imported from {list(getattr(current, '__path__', []))} "
                               f"before the checkout {root} was selected")
        return root
    _patch_torch()
    _patch_transformer_engine()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # The checkout's megatron/ has no __init__.py; a regular package found anywhere on sys.path would win over it.
    spec = importlib.machinery.ModuleSpec("megatron", None, is_package=True)
    spec.submodule_search_locations = [str(package_dir)]
    sys.modules["megatron"] = importlib.util.module_from_spec(spec)
    importlib.invalidate_caches()
    importlib.import_module("megatron.training")
    return root


def make_wandb_checkpoint_callback_nonfatal() -> None:
    """Report and skip a failure of the W&B artifact callback that runs after every checkpoint save.

    megatron/training/checkpointing.py calls wandb_utils.on_save_checkpoint_success() without exception handling;
    a transient network or authentication error in run.log_artifact() would otherwise kill a run whose checkpoint
    has already been written. The artifact is only a reference, metric logging is unaffected.
    """
    from megatron.training import wandb_utils

    callback = wandb_utils.on_save_checkpoint_success

    @wraps(callback)
    def on_save_checkpoint_success(*args: Any, **kwargs: Any) -> Any:
        try:
            return callback(*args, **kwargs)
        except Exception as error:  # noqa: BLE001
            print(f"W&B checkpoint callback failed after a successful save: {type(error).__name__}: {error}",
                  file=sys.stderr, flush=True)
            return None

    wandb_utils.on_save_checkpoint_success = on_save_checkpoint_success


def stream_wandb_file_hashes() -> Callable[[], None]:
    """Hash W&B artifact files with bounded reads instead of memory maps.

    Memory-mapped hashing of Megatron's checkpoint tracker file on a shared filesystem raised SIGBUS after a
    successful save, which bypasses the SDK's OSError fallback. Returns a function that restores the SDK helpers.
    """
    from wandb.sdk.lib import hashutil

    def hash_files(factory, *paths):
        hasher = factory()
        for path in sorted(map(str, paths)):
            with open(path, "rb") as stream:
                while chunk := stream.read(128 * 1024):
                    hasher.update(chunk)
        return hasher

    originals = {}
    for name, factory_name in (("_md5_file_hasher", "_md5"), ("_xxh128_file_hasher", "_xxh128")):
        if hasattr(hashutil, name):
            originals[name] = getattr(hashutil, name)
            factory = getattr(hashutil, factory_name)
            setattr(hashutil, name, lambda *paths, _factory=factory: hash_files(_factory, *paths))
    if not originals:
        raise RuntimeError("unsupported W&B file-hashing API")

    def restore() -> None:
        for name, original in originals.items():
            setattr(hashutil, name, original)

    return restore


class _ResolvedArguments:
    def __init__(self, module: Any, args: Any) -> None:
        self._module = module
        self._args = args

    def parse_args(self, *positional: Any, **keywords: Any) -> Any:
        return self._args

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


def use_resolved_args_in_ft_setup(args: Any) -> Callable[[], None]:
    """Let Megatron's fault-tolerance setup read the trainer's resolved arguments.

    ft_integration.setup() re-parses sys.argv with Megatron's own parser, which does not know the trainer's options
    (e.g. --optimizer muonh) and misses the resolved recipe values. Only the FT module's reference is rebound, and
    only while setup() runs.
    """
    ft = importlib.import_module("megatron.training.ft_integration")
    original = ft.setup
    if (getattr(original, "__globals__", None) is not vars(ft)
            or not callable(getattr(ft.arguments, "parse_args", None))):
        raise RuntimeError("unsupported Megatron fault-tolerance interface: expected setup() using "
                           "arguments.parse_args")

    @wraps(original)
    def setup(*positional: Any, **keywords: Any) -> Any:
        previous = ft.arguments
        ft.arguments = _ResolvedArguments(previous, args)
        try:
            return original(*positional, **keywords)
        finally:
            ft.arguments = previous

    ft.setup = setup

    def restore() -> None:
        ft.setup = original

    return restore
