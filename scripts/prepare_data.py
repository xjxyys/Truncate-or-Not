#!/usr/bin/env python3
"""Prepare the data of the scaling study under DATA_ROOT (default: data/), after downloading the shards:

    hf download karpathy/fineweb-edu-100b-shuffle --repo-type dataset \
        --revision 4c8f30d6756da75362432a4d5569e1b229263b71 --local-dir data/fineweb-edu-100b-shuffle

Steps, in order:
  manifest     manifest of the 1,821 training shards (symlinks next to it), read by the training loader
  validation   the first 4,096 hash-held-out documents of the training stream, evaluated during training
  duplicates   held-out documents (last two shards) whose exact text also occurs in a training shard
  heldout      the 106,496 held-out documents in one Parquet file, in shard order
  pack         tokenize both evaluation sets and cut them into 2,049-token records (scripts/eval_heldout.py)
FINEWEB_EDU_DIR overrides the location of the downloaded shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from looptf.data.layout import DATASET, HELDOUT_SHARDS, NUM_SHARDS, NUM_TRAIN_SHARDS, DataLayout  # noqa: E402

RECORD = 2049


def now():
    return time.strftime("%Y%m%d%H%M%S")


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def doc_hash(text):
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def cmd_manifest(layout, args):
    import pyarrow.parquet as pq

    if layout.train_manifest.exists():
        raise SystemExit(f"{layout.train_manifest} already exists; not overwriting")
    missing = [f"shard_{i:05d}.parquet" for i in range(NUM_SHARDS)
               if not (layout.shards / f"shard_{i:05d}.parquet").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} shards missing under {layout.shards} (first: {missing[0]})")
    directory = layout.train_manifest.parent
    (directory / "shards").mkdir(parents=True, exist_ok=True)
    files, docs = [], 0
    for i in range(NUM_TRAIN_SHARDS):
        name = f"shard_{i:05d}.parquet"
        target, link = layout.shards / name, directory / "shards" / name
        if not link.is_symlink():
            link.symlink_to(target.resolve())
        files.append({"path": f"shards/{name}", "size": target.stat().st_size})
        docs += pq.ParquetFile(target).metadata.num_rows
    manifest = {
        "format_version": 1, "source": DATASET, "files": files,
        "derivation": {
            "kind": "karpathy-shuffle-train", "created": now(), "source_dir": str(layout.shards.resolve()),
            "train_shards": f"shard_00000..shard_{NUM_TRAIN_SHARDS - 1:05d} ({len(files)} symlinks)",
            "excluded_val_shards": list(HELDOUT_SHARDS), "train_docs": docs,
            "note": "the hash-based held-out documents (fraction 0.001, seed 260421106) are excluded online "
                    "by the loader",
            "script": "scripts/prepare_data.py manifest",
        },
    }
    temporary = layout.train_manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, layout.train_manifest)
    print(json.dumps(manifest["derivation"]))


def cmd_validation(layout, args):
    from looptf.data.validation_cache import prepare_validation_cache

    os.environ.setdefault("HF_HOME", str(layout.hf_cache))
    source = SimpleNamespace(
        online_dataset=DATASET["dataset"], online_dataset_subset=DATASET["subset"],
        online_dataset_revision=DATASET["revision"], online_source_split=DATASET["split"],
        online_source_mode="local", online_raw_manifest=str(layout.train_manifest),
        online_data_cache_dir=str(layout.hf_cache), online_text_column="text", online_validation_fraction=0.001,
        online_split_seed=260_421_106,
    )
    print(prepare_validation_cache(source, layout.val4096, args.documents, args.max_source_documents))


_HELDOUT = None


def _init(hashes):
    global _HELDOUT
    _HELDOUT = hashes


def _scan(path):
    import pyarrow.parquet as pq

    shard = pq.ParquetFile(path)
    hits, rows = [], 0
    for group in range(shard.num_row_groups):
        for text in shard.read_row_group(group, columns=["text"]).column("text").to_pylist():
            rows += 1
            digest = doc_hash(text)
            if digest in _HELDOUT:
                hits.append(digest)
    return rows, hits


def cmd_duplicates(layout, args):
    import pyarrow.parquet as pq

    heldout = [doc_hash(text) for name in HELDOUT_SHARDS
               for text in pq.read_table(layout.shards / name, columns=["text"]).column("text").to_pylist()]
    unique = frozenset(heldout)
    first, copies, docs = {h: -1 for h in unique}, {h: 0 for h in unique}, 0
    shards = [layout.shards / f"shard_{i:05d}.parquet" for i in range(NUM_TRAIN_SHARDS)]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(unique,)) as pool:
        for i, (rows, hits) in enumerate(pool.map(_scan, shards, chunksize=4)):
            docs += rows
            for digest in hits:
                copies[digest] += 1
                if first[digest] < 0:
                    first[digest] = i
    first_shard = np.array([first[h] for h in heldout], dtype=np.int32)
    layout.duplicates.parent.mkdir(parents=True, exist_ok=True)
    np.savez(layout.duplicates, first_shard=first_shard, n_copies=np.array([copies[h] for h in heldout], np.int32),
             val_hash=np.frombuffer(b"".join(heldout), dtype="S8"))
    report = {"val_docs": len(heldout), "train_docs_scanned": docs,
              "val_with_copy_in_train": int((first_shard >= 0).sum()),
              "frac_with_copy": round(float((first_shard >= 0).mean()), 4)}
    layout.duplicates.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


def cmd_heldout(layout, args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    target = layout.heldout / "validation.parquet"
    if target.exists():
        raise SystemExit(f"{target} already exists; not overwriting")
    texts, shards, rows = [], [], []
    for name in HELDOUT_SHARDS:
        for i, text in enumerate(pq.read_table(layout.shards / name, columns=["text"]).column("text").to_pylist()):
            texts.append(text)
            shards.append(name)
            rows.append(i)
    duplicates = np.load(layout.duplicates)
    hashes = np.frombuffer(b"".join(doc_hash(t) for t in texts), dtype="S8")
    if not np.array_equal(hashes, duplicates["val_hash"]):
        raise SystemExit(f"document order differs from {layout.duplicates}; rerun the duplicates step")
    temporary = layout.heldout / ".validation.parquet.tmp"
    pq.write_table(pa.table({"text": texts, "source_shard": shards, "source_row": rows}), temporary,
                   row_group_size=1000)
    temporary.replace(target)
    manifest = {"format_version": 1, "kind": "heldout-full-v1", "created": now(),
                "source": {**DATASET, "shards": list(HELDOUT_SHARDS),
                           "note": "all documents of the held-out shards, not deduplicated; identical copies in the "
                                   "training shards are listed in duplicates.npz (same order)"},
                "docs": len(texts), "files": [{"path": "validation.parquet", "sha256": sha256(target)}],
                "script": "scripts/prepare_data.py heldout"}
    (layout.heldout / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(len(texts), manifest["files"][0]["sha256"])


def evaluation_sets(layout):
    return {"val4096": layout.val4096, "heldout": layout.heldout}


def cmd_pack(layout, args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from looptf.data.tokenizer import prepare_tokenizer

    tokenizer = AutoTokenizer.from_pretrained(prepare_tokenizer(data_root=str(layout.root)), use_fast=True)
    layout.packed.mkdir(parents=True, exist_ok=True)
    for name, directory in evaluation_sets(layout).items():
        target = layout.packed / f"{name}_t{RECORD}.npz"
        digest = sha256(directory / "validation.parquet")
        if target.exists() and not args.redo and str(np.load(target)["parquet_sha256"]) == digest:
            print(f"{name}: {target} matches the Parquet file, skipped")
            continue
        texts = pq.read_table(directory / "validation.parquet", columns=["text"]).column(0).to_pylist()
        lengths, chunks = [], []
        for i in range(0, len(texts), 2048):
            for ids in tokenizer(texts[i:i + 2048], add_special_tokens=False)["input_ids"]:
                chunks.append(np.asarray(ids + [tokenizer.eos_token_id], dtype=np.uint16))
                lengths.append(len(ids) + 1)
        stream = np.concatenate(chunks)
        records = len(stream) // RECORD
        temporary = layout.packed / f".{name}_t{RECORD}.tmp.npz"
        np.savez(temporary, records=stream[: records * RECORD].reshape(records, RECORD),
                 doc_lens=np.asarray(lengths, np.int32), parquet_sha256=np.asarray(digest))
        temporary.replace(target)
        print(f"{name}: {len(texts)} documents, {len(stream)} tokens, {records} records -> {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", str(REPO_ROOT / "data")))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest")
    p = sub.add_parser("validation")
    p.add_argument("--documents", type=int, default=4096)
    p.add_argument("--max-source-documents", type=int, default=10_000_000)
    sub.add_parser("duplicates").add_argument("--workers", type=int, default=40)
    sub.add_parser("heldout")
    sub.add_parser("pack").add_argument("--redo", action="store_true")
    args = parser.parse_args()
    commands = {"manifest": cmd_manifest, "validation": cmd_validation, "duplicates": cmd_duplicates,
                "heldout": cmd_heldout, "pack": cmd_pack}
    commands[args.command](DataLayout(args.data_root), args)


if __name__ == "__main__":
    main()
