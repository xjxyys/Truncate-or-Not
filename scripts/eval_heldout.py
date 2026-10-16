#!/usr/bin/env python3
"""Held-out evaluation: per-document loss of the final checkpoint of each training run.

Two sets, prepared by scripts/prepare_data.py (heldout, pack), each concatenated in document order with an EOS
after every document, cut into non-overlapping 2,049-token records and evaluated once per record:
  heldout  the 106,496 documents of the last two shards, which no run trains on
  val4096  the 4,096 documents that training evaluates periodically; used to check this evaluation against the
           validation loss logged during training
The model is rebuilt from the Megatron checkpoint (FP32 weights, BF16 autocast) and runs as many loops as in
training. `read` reports the loss over all held-out documents and over the 57,001 documents without an identical
copy in the training shards; the latter is the loss reported in the paper.

Self-check: the val4096 loss minus the final periodic validation loss logged by training must fall in
[CHECK_LO, CHECK_HI]; both read the same documents, cut into records and weighted differently. The window was
calibrated on single-node runs with 8 data-parallel ranks; runs with another data-parallel size get check=na.

    python scripts/eval_heldout.py eval --list runs.txt --gpus 8     # one run name per line
    python scripts/eval_heldout.py read --list runs.txt

Runs are read from <output-root>/runs/<run>; results go to <output-root>/eval/heldout/{runs,docs}/<run>@<iter>.
Runs that already have results are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from looptf.data.layout import DataLayout  # noqa: E402

RECORD = 2049
# val4096 loss minus the logged final periodic validation loss was -0.0035 to -0.0030 (median -0.0033) on five
# single-node runs (four 60M, one 1B); the window is the median +- 0.0027.
CHECK_LO, CHECK_HI = -0.0060, -0.0006


class Paths:
    def __init__(self, args):
        self.data = DataLayout(args.data_root)
        self.runs = Path(args.output_root) / "runs"
        self.eval = Path(args.output_root) / "eval" / "heldout"
        self.sets = {"val4096": self.data.val4096, "heldout": self.data.heldout}

    def final_iter(self, run):
        path = self.runs / run / "checkpoints" / "latest_checkpointed_iteration.txt"
        return int(path.read_text().strip()) if path.exists() else None

    def results(self, run, iteration):
        return self.eval / "runs" / f"{run}@{iteration}.json", self.eval / "docs" / f"{run}@{iteration}.npz"


def now():
    return time.strftime("%Y%m%d%H%M%S")


def read_list(path):
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip() and not x.startswith("#")]


def parquet_sha(directory):
    with open(directory / "validation.parquet", "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_model(checkpoint):
    """The looped transformer with the weights of a Megatron checkpoint (FP32, eval mode): (model, args, iter)."""
    import torch

    from looptf.model import LoopedTransformer, ModelConfig

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    a = payload["args"]
    model = LoopedTransformer(ModelConfig(
        vocab_size=a.vocab_size, padded_vocab_size=a.padded_vocab_size, hidden_size=a.hidden_size,
        intermediate_size=a.ffn_hidden_size, num_attention_heads=a.num_attention_heads,
        head_dim=a.hidden_size // a.num_attention_heads, max_position_embeddings=a.seq_length,
        rope_theta=float(a.rotary_base), rms_norm_eps=float(a.layernorm_epsilon),
        num_prelude_layers=a.prelude_layers, num_recurrent_layers=a.recurrent_layers, num_coda_layers=a.coda_layers,
        recurrence_steps=a.loops, backprop_steps=a.backprop_loops, untie_loops=a.untie_loops,
    ))
    state = {k[len("model."):]: v for k, v in payload["model"].items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("_extra_state")]
    unexpected = [k for k in unexpected if not k.endswith("_extra_state")]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.float().eval()
    return model, a, int(payload["iteration"])


def load_set(paths, name, device):
    import torch

    packed = np.load(paths.data.packed / f"{name}_t{RECORD}.npz")
    if str(packed["parquet_sha256"]) != parquet_sha(paths.sets[name]):
        raise SystemExit(f"{name}: the packed file does not match the Parquet file; run prepare_data.py pack --redo")
    records, lengths = packed["records"], packed["doc_lens"]
    doc_of = np.ascontiguousarray(
        np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)[: records.size].reshape(records.shape)[:, 1:])
    return {"x": torch.from_numpy(records.astype(np.int64)).to(device), "doc": torch.from_numpy(doc_of).to(device),
            "n_docs": len(lengths)}


def run_set(model, data, loops, batch_size, limit=0):
    """Summed cross-entropy and token count of every document."""
    import torch
    import torch.nn.functional as F

    x, doc = data["x"], data["doc"]
    if limit:
        x, doc = x[:limit], doc[:limit]
    nll = torch.zeros(data["n_docs"], dtype=torch.float64, device=x.device)
    ntok = torch.zeros(data["n_docs"], dtype=torch.float64, device=x.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, x.shape[0], batch_size):
            batch = x[i:i + batch_size]
            logits = model(batch[:, :-1], num_steps=loops).logits
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), batch[:, 1:].reshape(-1),
                                 reduction="none")
            d = doc[i:i + batch_size].reshape(-1).long()
            nll.index_add_(0, d, ce.double())
            ntok.index_add_(0, d, torch.ones_like(ce, dtype=torch.float64))
    return nll.cpu().numpy(), ntok.cpu().numpy()


def ce_of(nll, ntok, mask=None):
    if mask is not None:
        nll, ntok = nll[mask], ntok[mask]
    return round(float(nll.sum() / ntok.sum()), 6), int(round(float(ntok.sum())))


def logged_val(paths, run, iteration):
    """Final periodic validation loss at `iteration` from the run's evaluation log."""
    value = None
    path = paths.runs / run / "evaluation_metrics.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.strip().startswith("{"):
            row = json.loads(line)
            if not row.get("eval/is_final") and row.get("eval/checkpoint_step") == iteration and "eval/val/loss" in row:
                value = row["eval/val/loss"]
    return value


def same_val_cache(paths, manifest):
    """True when a run's training-time validation cache holds the same files as the val4096 set evaluated here."""
    if not manifest:
        return False
    ours = paths.data.val4096 / "manifest.json"
    try:
        return json.loads(Path(manifest).read_text())["files"] == json.loads(ours.read_text())["files"]
    except (OSError, ValueError, KeyError):
        return Path(manifest).resolve() == ours.resolve()


def evaluate_runs(paths, args):
    import torch

    torch.cuda.set_device(args.worker % torch.cuda.device_count())
    device = f"cuda:{torch.cuda.current_device()}"
    mine = [r for i, r in enumerate(read_list(args.list)) if i % args.workers == args.worker]
    data, problems = {}, []
    print(f"[w{args.worker}] host={socket.gethostname()} device={device} runs={len(mine)}", flush=True)
    for k, run in enumerate(mine):
        iteration = paths.final_iter(run)
        if iteration is None:
            print(f"[w{args.worker}] {run}: no checkpoint, skipped", flush=True)
            problems.append(run)
            continue
        json_path, docs_path = paths.results(run, iteration)
        if json_path.exists() and not args.redo and not args.limit:
            continue
        start = time.time()
        checkpoint = paths.runs / run / "checkpoints" / f"iter_{iteration:07d}" / "mp_rank_00" / "model_optim_rng.pt"
        model, a, real_iteration = load_model(checkpoint)
        if real_iteration != iteration or int(a.train_iters) != iteration:
            print(f"[w{args.worker}] {run}: last checkpoint is iteration {real_iteration}, training ran "
                  f"{a.train_iters}; skipped", flush=True)
            problems.append(run)
            continue
        model.to(device)
        load_seconds = time.time() - start
        for name in paths.sets:
            if name not in data:
                data[name] = load_set(paths, name, device)
        results, arrays, walls = {}, {}, {}
        for name in paths.sets:
            start = time.time()
            nll, ntok = run_set(model, data[name], int(a.loops), args.bs, args.limit if name == "heldout" else 0)
            walls[name] = round(time.time() - start, 1)
            arrays[f"{name}_nll"], arrays[f"{name}_ntok"] = nll, ntok.astype(np.int32)
            ce, tokens = ce_of(nll, ntok)
            results[name] = {"ce": ce, "n_tok": tokens, "n_rec": int(data[name]["x"].shape[0]),
                             "n_docs": data[name]["n_docs"]}
        reference = logged_val(paths, run, iteration)
        validation_manifest = getattr(a, "online_validation_manifest", None)
        dp = int(getattr(a, "data_parallel_size", 0) or 0)
        diff = (round(results["val4096"]["ce"] - reference, 5)
                if reference is not None and same_val_cache(paths, validation_manifest) else None)
        check = "na" if diff is None or dp != 8 else ("ok" if CHECK_LO <= diff <= CHECK_HI else "fail")
        row = {"ts": now(), "run": run, "iter": iteration, "T": int(a.loops), "B": int(a.backprop_loops),
               "untie_loops": bool(a.untie_loops), "hidden": int(a.hidden_size), "dp": dp, "logged_val": reference,
               "logged_val_set": validation_manifest, "val4096_minus_logged": diff, "check": check, "sets": results,
               "heldout_manifest": str(paths.data.heldout / "manifest.json"),
               "wall_s": {"load": round(load_seconds, 1), **walls}, "host": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(device), "bs": args.bs}
        if args.limit:
            print(f"[w{args.worker}] self-test (--limit {args.limit}, nothing written): {json.dumps(row)}", flush=True)
        else:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = docs_path.with_name(docs_path.stem + ".tmp.npz")
            np.savez(temporary, **arrays)
            temporary.replace(docs_path)
            temporary = json_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(row) + "\n")
            temporary.replace(json_path)
            print(f"[w{args.worker}] {k + 1}/{len(mine)} {run}@{iteration} heldout={results['heldout']['ce']} "
                  f"val4096={results['val4096']['ce']} delta_logged={diff} check={check}", flush=True)
        if check == "fail":
            problems.append(run)
        del model
        torch.cuda.empty_cache()
    if problems:
        print(f"[w{args.worker}] runs with problems (no checkpoint, last checkpoint not the final step, or "
              f"self-check outside the window): {problems}", flush=True)
        sys.exit(2)


def cmd_eval(paths, args):
    if args.worker is not None or args.gpus <= 1:
        args.worker, args.workers = args.worker or 0, args.workers or 1
        evaluate_runs(paths, args)
        return
    log_dir = paths.eval / "logs" / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    workers = []
    for g in range(args.gpus):
        command = [sys.executable, __file__, "--data-root", args.data_root, "--output-root", args.output_root, "eval",
                   "--list", args.list, "--worker", str(g), "--workers", str(args.gpus), "--bs", str(args.bs),
                   *(["--redo"] if args.redo else [])]
        workers.append(subprocess.Popen(command, env=env, stdout=open(log_dir / f"worker{g}.log", "a"),
                                        stderr=subprocess.STDOUT))
    codes = [w.wait() for w in workers]
    print(f"workers finished with {codes}; {status(paths, args.list)} runs evaluated; logs in {log_dir}")
    if any(codes):
        sys.exit(1)


def status(paths, run_list):
    runs = read_list(run_list)
    done = sum(1 for r in runs if (it := paths.final_iter(r)) is not None and paths.results(r, it)[0].exists())
    return f"{done}/{len(runs)}"


def cmd_read(paths, args):
    copied = np.load(paths.data.duplicates)["first_shard"] >= 0
    runs = set(read_list(args.list)) if args.list else None
    print(f"held-out documents with an identical copy in the training shards: {int(copied.sum())} of {len(copied)} "
          f"({copied.mean():.1%}); dedup = the remaining {int((~copied).sum())} documents")
    print("| run | iter | dedup CE | all-documents CE | duplicated-documents CE | self-check |")
    print("|---|---|---|---|---|---|")
    for path in sorted((paths.eval / "runs").glob("*.json")):
        row = json.loads(path.read_text())
        if runs and row["run"] not in runs:
            continue
        docs = np.load(paths.eval / "docs" / f"{row['run']}@{row['iter']}.npz")
        nll, ntok = docs["heldout_nll"], docs["heldout_ntok"].astype(np.float64)
        full, dedup, dup = (ce_of(nll, ntok, m)[0] for m in (np.ones_like(copied), ~copied, copied))
        print(f"| {row['run']} | {row['iter']} | {dedup:.4f} | {full:.4f} | {dup:.4f} | "
              f"{row['check']} {row.get('val4096_minus_logged')} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", str(REPO_ROOT / "data")))
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", str(REPO_ROOT / "output")))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("eval", help="evaluate the final checkpoints of the listed runs")
    p.add_argument("--list", required=True)
    p.add_argument("--gpus", type=int, default=1, help="one worker per GPU")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--redo", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N held-out records, write nothing")
    p.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    p.add_argument("--workers", type=int, help=argparse.SUPPRESS)
    sub.add_parser("status").add_argument("--list", required=True)
    sub.add_parser("read").add_argument("--list", default="")
    args = parser.parse_args()
    paths = Paths(args)
    if args.command == "eval":
        cmd_eval(paths, args)
    elif args.command == "status":
        print(status(paths, args.list))
    else:
        cmd_read(paths, args)


if __name__ == "__main__":
    main()
