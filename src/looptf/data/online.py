"""Training and validation loaders that tokenize FineWeb-Edu while training (Megatron's external-loader path)."""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .streaming import StreamingBatchLoader, stable_validation_assignment

DEFAULT_DATASET = "karpathy/fineweb-edu-100b-shuffle"
DEFAULT_DATASET_SUBSET = "default"
DEFAULT_DATASET_REVISION = "4c8f30d6756da75362432a4d5569e1b229263b71"
DEFAULT_TOKENIZER_REVISION = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
_ACTIVE_LOADERS: list[Any] = []


def close_online_dataloaders() -> None:
    """Release Arrow iterators while Python is still alive, including early exits."""
    import gc

    had_source = any(loader.loader._source is not None for loader in _ACTIVE_LOADERS)
    while _ACTIVE_LOADERS:
        loader = _ACTIVE_LOADERS.pop()
        if not loader.close(timeout=120):
            print(f"online {loader.split_name} producer still blocked during shutdown", flush=True)
    gc.collect()
    if had_source:
        # Match datasets' Parquet/interleave shutdown workaround: Arrow can
        # still have queued native callbacks after generator destruction. Let
        # them acquire the GIL before interpreter finalization (arrow#45214).
        from datasets import config as datasets_config

        time.sleep(getattr(datasets_config, "SLEEP_TIME_ON_THREADS_SHUTDOWN", 5))


def add_online_data_arguments(parser: Any) -> None:
    group = parser.add_argument_group("online data")
    group.add_argument("--online-dataset", default=DEFAULT_DATASET)
    group.add_argument("--online-dataset-subset", default=DEFAULT_DATASET_SUBSET)
    group.add_argument("--online-dataset-revision", default=DEFAULT_DATASET_REVISION)
    group.add_argument("--online-source-split", default="train")
    group.add_argument("--online-text-column", default="text")
    group.add_argument("--online-validation-fraction", type=float, default=0.001)
    group.add_argument("--online-split-seed", type=int, default=260_421_106)
    group.add_argument("--online-tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    group.add_argument("--online-tokenizer-batch-size", type=int, default=256)
    group.add_argument("--online-shuffle-buffer-records", type=int, default=128)
    group.add_argument("--online-data-cache-dir")
    group.add_argument("--online-prefetch-batches", type=int, default=16)
    group.add_argument(
        "--online-source-mode", choices=("auto", "stream", "local"), default="auto",
        help="Read the local shard manifest when it exists (auto), stream from the Hub, or require local files.",
    )
    group.add_argument("--online-raw-manifest", help="Training-shard manifest (scripts/prepare_data.py).")
    group.add_argument("--online-validation-manifest",
                       help="Held-out documents evaluated during training (scripts/prepare_data.py).")
    if "--dataloader-save" not in parser._option_string_actions:
        group.add_argument("--dataloader-save", default=None, help="Directory for the dataloader checkpoint state.")


def validate_online_data_args(args: Any) -> None:
    from .manifest import resolve_source_mode

    resolve_source_mode(args)
    if args.dataloader_type != "external":
        raise ValueError("--online-data requires --dataloader-type external")
    if not 0.0 < args.online_validation_fraction < 1.0:
        raise ValueError("--online-validation-fraction must be in (0, 1)")
    if args.online_tokenizer_batch_size < 1:
        raise ValueError("--online-tokenizer-batch-size must be positive")
    if args.online_shuffle_buffer_records < 0:
        raise ValueError("--online-shuffle-buffer-records must be non-negative")
    if getattr(args, "online_prefetch_batches", 16) < 0:
        raise ValueError("--online-prefetch-batches must be non-negative")
    if getattr(args, "online_source_mode", "stream") == "local":
        if not getattr(args, "online_raw_manifest", None):
            raise ValueError("local online source requires --online-raw-manifest")
        from .manifest import read_manifest, source_identity

        read_manifest(args.online_raw_manifest, source_identity(args))
    if getattr(args, "online_validation_manifest", None):
        from .validation_cache import read_validation_manifest
        read_validation_manifest(args.online_validation_manifest, args)
    if args.num_workers != 0:
        raise ValueError(
            "--online-data requires --num-workers 0; the loader tokenizes in batches itself "
            "and does not use Megatron's DataLoader workers"
        )
    if (getattr(args, "save", None) or getattr(args, "load", None)) and not args.dataloader_save:
        raise ValueError("online data with checkpoint save/load requires --dataloader-save")


def _build_source_factory(args: Any, dp_rank: int, dp_world_size: int, *, local_files=None) -> Callable[[], Any]:
    from .manifest import resolve_source_mode

    resolve_source_mode(args)
    if local_files is None and getattr(args, "online_source_mode", "stream") == "local":
        from .manifest import read_manifest, source_identity

        local_files = read_manifest(args.online_raw_manifest, source_identity(args))

    def build() -> Any:
        from datasets import load_dataset
        from datasets.distributed import split_dataset_by_node
        from pyarrow.dataset import ParquetFragmentScanOptions

        kwargs: dict[str, Any] = {
            "path": args.online_dataset,
            "split": args.online_source_split,
            "revision": args.online_dataset_revision,
            "streaming": True,
        }
        if args.online_dataset_subset:
            kwargs["name"] = args.online_dataset_subset
        if args.online_data_cache_dir:
            Path(args.online_data_cache_dir).expanduser().mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = args.online_data_cache_dir
        if local_files is not None:
            # No dataset-repository resolution or remote fallback in local mode.
            kwargs = {
                "path": "parquet", "data_files": {"train": local_files},
                "split": "train", "streaming": True,
                "cache_dir": args.online_data_cache_dir,
            }
        # The batch producer already reads ahead. Arrow's separate range-prefetch pool can outlive a closed
        # iterator while an HTTP retry is sleeping and then deadlock on the GIL at interpreter exit.
        kwargs["fragment_scan_options"] = ParquetFragmentScanOptions(pre_buffer=False)
        source = load_dataset(**kwargs)
        return split_dataset_by_node(source, rank=dp_rank, world_size=dp_world_size)

    return build


def _dataloader_checkpoint_path(args: Any, dp_rank: int) -> Path | None:
    if not args.dataloader_save or not getattr(args, "load", None) or args.iteration <= 0:
        return None
    from megatron.training.checkpointing import get_checkpoint_name

    return Path(
        get_checkpoint_name(
            args.dataloader_save,
            args.iteration,
            pipeline_rank=0,
            basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
        )
    )


def build_online_dataloaders(args: Any) -> tuple[Any, Any, Any]:
    """Rank-local training and validation loaders (no separate test split)."""
    from megatron.core import mpu
    from transformers import AutoTokenizer

    if mpu.get_tensor_model_parallel_rank() != 0:
        return None, None, None

    dp_rank = mpu.get_data_parallel_rank()
    dp_world_size = mpu.get_data_parallel_world_size()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        revision=args.online_tokenizer_revision,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("online tokenizer has no eos_token_id")
    # The tokenizer's vocab_size excludes added tokens such as <|assistant_end|>.
    active_vocab_size = len(tokenizer)
    if getattr(args, "vocab_size", None) is not None and active_vocab_size != args.vocab_size:
        raise ValueError(
            f"configured vocabulary size {args.vocab_size} does not match "
            f"online tokenizer vocabulary size {active_vocab_size}"
        )

    source_factory = _build_source_factory(args, dp_rank, dp_world_size)

    validation_source_factory = source_factory
    validation_manifest = None
    if getattr(args, "online_validation_manifest", None):
        from .validation_cache import read_validation_manifest
        files, validation_manifest = read_validation_manifest(args.online_validation_manifest, args)
        if validation_manifest["documents"] < dp_world_size:
            raise ValueError("offline validation needs at least one document per data-parallel rank")
        validation_source_factory = _build_source_factory(
            args, dp_rank, dp_world_size, local_files=files,
        )

    from .prefetch import PrefetchBatchLoader

    def make_loader(split_name: str) -> PrefetchBatchLoader:
        train = split_name == "train"
        shuffle_buffer_records = args.online_shuffle_buffer_records if train else 0
        loader = StreamingBatchLoader(
            source_factory=source_factory if train else validation_source_factory,
            tokenizer=tokenizer,
            split_name=split_name,
            text_column=args.online_text_column,
            validation_fraction=args.online_validation_fraction,
            split_seed=args.online_split_seed,
            sequence_length=args.seq_length,
            micro_batch_size=args.micro_batch_size if train else args.eval_micro_batch_size,
            tokenizer_batch_size=args.online_tokenizer_batch_size,
            shuffle_buffer_records=shuffle_buffer_records,
            shuffle_seed=args.seed + 1_000_003 * dp_rank + (0 if train else 1),
            create_attention_mask=args.create_attention_mask_in_dataloader,
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
        )
        loader.source_identity = {
            "mode": getattr(args, "online_source_mode", "stream"),
            "dataset": getattr(args, "online_dataset", None),
            "subset": getattr(args, "online_dataset_subset", None),
            "revision": getattr(args, "online_dataset_revision", None),
            "split": getattr(args, "online_source_split", None),
            "dp_rank": dp_rank, "dp_world_size": dp_world_size,
            "sequence_length": args.seq_length, "micro_batch_size": args.micro_batch_size,
            "tokenizer": args.tokenizer_model,
            "tokenizer_revision": args.online_tokenizer_revision,
            "validation_fraction": args.online_validation_fraction,
            "split_seed": args.online_split_seed,
            "text_column": args.online_text_column,
            "shuffle_buffer_records": shuffle_buffer_records,
            "tokenizer_batch_size": args.online_tokenizer_batch_size,
            "create_attention_mask": args.create_attention_mask_in_dataloader,
            "reset_position_ids": args.reset_position_ids,
            "reset_attention_mask": args.reset_attention_mask,
            "eod_mask_loss": args.eod_mask_loss,
        }
        if not train and validation_manifest is not None:
            loader.source_identity.update(mode="local", validation_cache=validation_manifest)
        capacity = getattr(args, "online_prefetch_batches", 16)
        # Validation accepts few documents, so avoid scanning many extra shards
        # between evaluations just to fill a large unused validation queue.
        if not train:
            capacity = min(capacity, 1)
        return PrefetchBatchLoader(loader, capacity=capacity)

    train = make_loader("train")
    validation = make_loader("validation")
    train.attach_checkpoint_peers(validation=validation)
    _ACTIVE_LOADERS.extend((train, validation))

    checkpoint_path = _dataloader_checkpoint_path(args, dp_rank)
    if checkpoint_path is not None and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train.restore_state(payload["dataloader_state_dict"])
        print(f"restored online dataloader state from {checkpoint_path}", flush=True)

    return train, validation, None


def dataset_provider(train_val_test_num_samples: Any, vp_stage: Any = None) -> Any:
    """Megatron dataset provider: the external online loaders."""
    from megatron.training import get_args, print_rank_0

    print_rank_0("> building external online Hugging Face dataloaders ...")
    loaders = build_online_dataloaders(get_args())
    print_rank_0("> finished building external online dataloaders")
    return loaders


dataset_provider.is_distributed = True

__all__ = [
    "StreamingBatchLoader",
    "add_online_data_arguments",
    "build_online_dataloaders",
    "dataset_provider",
    "stable_validation_assignment",
    "validate_online_data_args",
]
