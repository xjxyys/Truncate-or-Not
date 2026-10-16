"""Where scripts/prepare_data.py puts FineWeb-Edu, its manifests and the held-out sets under DATA_ROOT.

karpathy/fineweb-edu-100b-shuffle at a pinned revision has 1,823 Parquet shards. Shards 0..1820 are the training
split, from which the online loader also holds out 0.1% of the documents by hash (the periodic validation set);
the last two shards are the held-out evaluation set that no run trains on.
"""

from __future__ import annotations

import os
from pathlib import Path

from .online import DEFAULT_DATASET, DEFAULT_DATASET_REVISION, DEFAULT_DATASET_SUBSET

DATASET = {"dataset": DEFAULT_DATASET, "subset": DEFAULT_DATASET_SUBSET, "revision": DEFAULT_DATASET_REVISION,
           "split": "train"}
NUM_SHARDS = 1823
NUM_TRAIN_SHARDS = 1821
HELDOUT_SHARDS = ("shard_01821.parquet", "shard_01822.parquet")


class DataLayout:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.shards = Path(os.environ.get("FINEWEB_EDU_DIR", self.root / "fineweb-edu-100b-shuffle"))
        self.train_manifest = self.root / "raw" / "fineweb-edu-100b-shuffle-train" / "manifest.json"
        self.val4096 = self.root / "validation" / "fineweb-edu-100b-shuffle-val4096"
        self.heldout = self.root / "validation" / "fineweb-edu-100b-shuffle-heldout"
        self.duplicates = self.heldout / "duplicates.npz"
        self.packed = self.root / "validation" / "packed"
        self.hf_cache = self.root / "cache" / "huggingface"
