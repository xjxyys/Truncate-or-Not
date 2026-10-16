"""Cache the first N hash-held-out documents of the training stream; evaluated periodically during training
(built by scripts/prepare_data.py validation)."""
from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import os
import re
import tempfile
from pathlib import Path

from .manifest import read_manifest, source_identity
from .streaming import stable_validation_assignment


def validation_identity(args):
    return {
        **source_identity(args),
        "text_column": args.online_text_column,
        "validation_fraction": args.online_validation_fraction,
        "split_seed": args.online_split_seed,
        "selection": "hash-heldout-prefix-v1",
    }


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_validation_manifest(path, args):
    """Require a matching, complete cache; never fall back to remote validation."""
    files = read_manifest(path, validation_identity(args))
    manifest = json.loads(Path(path).expanduser().read_text())
    if manifest.get("kind") != "online-validation" or manifest.get("documents", 0) < 1:
        raise ValueError("invalid offline validation manifest")
    for entry, local in zip(manifest["files"], files):
        if _sha256(local) != entry.get("sha256"):
            raise ValueError(f"offline validation checksum mismatch: {local}")
    return files, manifest


def prepare_validation_cache(args, directory, documents, max_source_documents, source_factory=None):
    """Save the first N hash-held-out documents, scanning at most the given limit."""
    if not re.fullmatch(r"[0-9a-f]{40}", args.online_dataset_revision):
        raise ValueError("validation caching requires an immutable 40-character dataset commit")
    if documents < 1 or max_source_documents < documents:
        raise ValueError("positive document target must not exceed the source scan limit")
    if not 0 < args.online_validation_fraction < 1:
        raise ValueError("validation fraction must be in (0, 1)")
    directory = Path(directory).expanduser().absolute()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    with (directory / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest_path.exists():
            _, manifest = read_validation_manifest(manifest_path, args)
            if manifest["documents"] != documents:
                raise ValueError("existing validation cache has a different document target")
            return manifest_path
        if source_factory is None:
            from .online import _build_source_factory
            source_factory = _build_source_factory(args, 0, 1)
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Temporary data stays under DATA_ROOT and is removed on failure. The
        # manifest is the publication boundary; readers never see a partial set.
        with tempfile.TemporaryDirectory(prefix=".validation-", dir=directory) as temporary:
            parquet = Path(temporary) / "validation.parquet"
            source = iter(source_factory())
            accepted = seen = 0
            pending = []
            schema = pa.schema([(args.online_text_column, pa.string())])
            try:
                with pq.ParquetWriter(parquet, schema, compression="zstd") as writer:
                    for row in itertools.islice(source, max_source_documents):
                        seen += 1
                        text = row.get(args.online_text_column)
                        if not isinstance(text, str) or not text:
                            continue
                        if not stable_validation_assignment(
                                text, args.online_validation_fraction, args.online_split_seed):
                            continue
                        pending.append(text)
                        accepted += 1
                        if len(pending) >= 256 or accepted == documents:
                            writer.write_table(pa.Table.from_pydict({args.online_text_column: pending}, schema=schema))
                            pending = []
                        if accepted == documents:
                            break
            finally:
                close = getattr(source, "close", None)
                if close is not None:
                    close()
            if accepted != documents:
                raise ValueError(f"found {accepted}/{documents} validation documents after {seen} source rows; "
                                 "no cache published")
            entry = {"path": "validation.parquet", "size": parquet.stat().st_size, "sha256": _sha256(parquet)}
            manifest = {
                "format_version": 1, "kind": "online-validation",
                "source": validation_identity(args), "documents": accepted,
                "source_documents_scanned": seen, "max_source_documents": max_source_documents,
                "files": [entry],
            }
            parquet.replace(directory / entry["path"])
            temporary_manifest = Path(temporary) / "manifest.json"
            with temporary_manifest.open("w") as handle:
                json.dump(manifest, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary_manifest.replace(manifest_path)
    return manifest_path
