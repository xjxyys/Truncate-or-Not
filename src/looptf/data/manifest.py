"""Local shard manifests: which Parquet files a run reads, validated without contacting the Hub."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def source_identity(args: Any) -> dict[str, str]:
    return {
        "dataset": args.online_dataset,
        "subset": args.online_dataset_subset or "",
        "revision": args.online_dataset_revision,
        "split": args.online_source_split,
    }


def resolve_source_mode(args: Any) -> None:
    """Resolve source mode "auto" once: read the local manifest if it exists, otherwise stream from the Hub."""
    if getattr(args, "online_source_mode", "stream") != "auto":
        return
    manifest = Path(args.online_raw_manifest) if getattr(args, "online_raw_manifest", None) else None
    if manifest is not None and manifest.is_file():
        # A published but invalid manifest is an error, not a silent network fallback.
        read_manifest(manifest, source_identity(args))
        args.online_raw_manifest = str(manifest)
        args.online_source_mode = "local"
        print(f"online source: using complete local raw cache {manifest}", flush=True)
    else:
        args.online_source_mode = "stream"
        print(f"online source: no complete raw cache at {manifest}; streaming from Hub", flush=True)


def read_manifest(path: str | Path, expected: dict[str, str]) -> list[str]:
    """Absolute paths of the manifest's files, after checking the source identity and every file size."""
    path = Path(path).expanduser().absolute()
    manifest = json.loads(path.read_text())
    if manifest.get("format_version") != 1 or manifest.get("source") != expected:
        raise ValueError("raw cache manifest does not match the configured dataset/revision/split")
    if not manifest.get("files"):
        raise ValueError("raw cache manifest contains no files")
    files = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("raw cache paths must be relative to the manifest directory")
        local = path.parent / relative
        if not local.is_file() or local.stat().st_size != entry["size"]:
            raise ValueError(f"raw cache file missing or incomplete: {local}")
        files.append(str(local))
    return files
