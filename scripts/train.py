#!/usr/bin/env python3
"""Train one configuration of the scaling study on this node (run the same command on every node of a job).

    python scripts/train.py --run-name m240_d4_t4b4 --size 240M --loops 4 --backprop-loops 4 --tokens 4

--tokens D sets the batch, length, warmup, evaluation and learning-rate multiplier of the paper's runs at D x 2^30
training tokens (looptf/scaling.py); every derived value can be overridden. Checkpoints, logs and evaluation
results go to <output-root>/runs/<run-name>/. Launching an unfinished run resumes it from its latest checkpoint;
a finished run (train_done.txt) is not started again.
"""

from __future__ import annotations

import argparse
import os
import resource
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from looptf import scaling  # noqa: E402
from looptf.data.layout import DataLayout  # noqa: E402
from looptf.data.tokenizer import prepare_tokenizer, snapshot_path  # noqa: E402
from looptf.recipe import LR_MULTIPLIERS  # noqa: E402

KEEP_LATEST_CHECKPOINT_ONLY = 2_000_000_000


def load_dotenv(path: Path) -> None:
    """KEY=VALUE lines of .env as defaults for the environment."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), os.path.expandvars(value.strip().strip("'\"")))


def visible_gpus() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, check=False).stdout
        if out.split():
            return len(out.split())
    return 8


def micro_batch(per_rank: int, cap: int = 16) -> int:
    """Largest number of sequences per micro-batch that divides the per-rank batch and does not exceed `cap`."""
    return max(m for m in range(1, min(per_rank, cap) + 1) if per_rank % m == 0)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True, help="Run directory under <output-root>/runs/.")
    model = p.add_argument_group("model")
    model.add_argument("--size", choices=tuple(scaling.SIZES), help="Model size of the scaling grid.")
    model.add_argument("--hidden-size", type=int)
    model.add_argument("--num-attention-heads", type=int, help="Default: hidden size / 128.")
    model.add_argument("--layers", type=int, nargs=3, metavar=("P", "R", "C"),
                       help="Prelude, recurrent and coda layers.")
    model.add_argument("--loops", type=int, required=True, help="T: forward loops of the recurrent block.")
    model.add_argument("--backprop-loops", type=int, help="B: backpropagated loops, B <= T (default T).")
    model.add_argument("--untie-loops", action="store_true",
                       help="Every loop gets its own copy of the recurrent block (needs B = T).")
    model.add_argument("--no-recompute", action="store_true",
                       help="Keep the activations of the backpropagated loops instead of recomputing them.")
    run = p.add_argument_group("training")
    run.add_argument("--tokens", type=float, help="Token budget in units of 2^30; sets the scaling-suite settings.")
    run.add_argument("--train-iters", type=int)
    run.add_argument("--global-batch-size", type=int, help="Sequences of 2,048 tokens per step.")
    run.add_argument("--micro-batch-size", type=int, help="Default: the largest divisor of the per-GPU batch <= 16.")
    run.add_argument("--lr-multiplier", choices=tuple(LR_MULTIPLIERS), metavar="M",
                     help="Multiplier on the recipe's base learning rates, a power of 2^(1/8) written as in "
                          "looptf/recipe.py, e.g. 0.9170 (default: the suite rule with --tokens, otherwise 1).")
    run.add_argument("--schedule", choices=("cosine", "linear"), default="cosine",
                     help="cosine: warmup then cosine decay (the runs of the paper); linear: the recipe's linear "
                          "decay without warmup.")
    run.add_argument("--warmup-iters", type=int, help="Default: 2%% of the steps (cosine schedule).")
    run.add_argument("--seed", type=int, default=42)
    ev = p.add_argument_group("evaluation and checkpoints")
    ev.add_argument("--eval-interval", type=int, help="Default: steps / 8.")
    ev.add_argument("--eval-iters", type=int, help="Default: 4,096 sequences per evaluation.")
    ev.add_argument("--final-eval-iters", type=int, help="Default: 69,120 sequences at the end.")
    ev.add_argument("--eval-global-batch-size", type=int,
                    help="Default: the training batch, or 64 if that does not divide 4,096.")
    ev.add_argument("--eval-micro-batch-size", type=int)
    ev.add_argument("--save-interval", type=int, help="Default: the evaluation interval.")
    ev.add_argument("--log-interval", type=int, default=10)
    ev.add_argument("--exit-interval", type=int, help="Stop (with a checkpoint) after this many steps.")
    env = p.add_argument_group("environment")
    env.add_argument("--data-root", default=os.environ.get("DATA_ROOT", str(REPO_ROOT / "data")))
    env.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", str(REPO_ROOT / "output")))
    env.add_argument("--megatron-lm-path",
                     default=os.environ.get("MEGATRON_LM_PATH", str(REPO_ROOT.parent / "Megatron-LM")))
    env.add_argument("--tokenizer", help="Existing tokenizer snapshot (default: prepared under <data-root>).")
    env.add_argument("--gpus", type=int, default=int(os.environ.get("GPUS_PER_NODE", "0")) or None,
                     help="Processes on this node (default: visible GPUs).")
    env.add_argument("--nnodes", type=int, default=int(os.environ.get("NNODES", "1")))
    env.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0")))
    env.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR"))
    env.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29500")))
    env.add_argument("--wandb-project", help="Log to Weights & Biases (needs WANDB_API_KEY).")
    env.add_argument("--wandb-entity")
    env.add_argument("--dry-run", action="store_true", help="Print the training command and exit.")
    args, extra = p.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def resolve(args: argparse.Namespace) -> None:
    if args.size:
        hidden, *layers = scaling.SIZES[args.size]
        args.hidden_size = args.hidden_size or hidden
        args.layers = args.layers or layers
    if not args.hidden_size or not args.layers:
        raise SystemExit("give --size, or --hidden-size and --layers")
    args.num_attention_heads = args.num_attention_heads or args.hidden_size // 128
    args.backprop_loops = args.backprop_loops or args.loops
    if args.tokens is not None:
        args.global_batch_size = args.global_batch_size or scaling.global_batch(args.tokens)
        args.train_iters = args.train_iters or scaling.train_iters(args.tokens, args.global_batch_size)
        args.lr_multiplier = args.lr_multiplier or scaling.lr_multiplier(args.tokens)
    if not args.train_iters or not args.global_batch_size:
        raise SystemExit("give --tokens, or --train-iters and --global-batch-size")
    args.lr_multiplier = args.lr_multiplier or "1"
    if args.warmup_iters is None:
        args.warmup_iters = scaling.warmup_iters(args.train_iters) if args.schedule == "cosine" else 0
    suite = scaling.evaluation(args.train_iters, args.global_batch_size)
    args.eval_interval = args.eval_interval or suite.interval
    if args.eval_global_batch_size is None and suite.global_batch_size != args.global_batch_size:
        args.eval_global_batch_size = suite.global_batch_size
    eval_batch = args.eval_global_batch_size or args.global_batch_size
    if args.eval_iters is None or args.final_eval_iters is None:
        if scaling.EVAL_SEQUENCES % eval_batch or scaling.FINAL_EVAL_SEQUENCES % eval_batch:
            raise SystemExit(f"the evaluation batch {eval_batch} does not divide the 4,096 / 69,120 evaluation "
                             f"sequences; give --eval-iters and --final-eval-iters")
        args.eval_iters = args.eval_iters or scaling.EVAL_SEQUENCES // eval_batch
        args.final_eval_iters = args.final_eval_iters or scaling.FINAL_EVAL_SEQUENCES // eval_batch
    args.save_interval = args.save_interval or args.eval_interval
    args.gpus = args.gpus or visible_gpus()
    data_parallel = args.gpus * args.nnodes
    if args.global_batch_size % data_parallel:
        raise SystemExit(f"the global batch {args.global_batch_size} must be divisible by the {data_parallel} GPUs")
    args.micro_batch_size = args.micro_batch_size or micro_batch(args.global_batch_size // data_parallel)
    if args.eval_global_batch_size and not args.eval_micro_batch_size:
        if args.eval_global_batch_size % data_parallel:
            raise SystemExit(f"the evaluation batch {args.eval_global_batch_size} must be divisible by the "
                             f"{data_parallel} GPUs")
        args.eval_micro_batch_size = micro_batch(args.eval_global_batch_size // data_parallel)
    if args.nnodes > 1 and not args.master_addr:
        raise SystemExit("multi-node training needs --master-addr (the address of node 0)")


def trainer_command(args: argparse.Namespace, run_dir: Path, tokenizer: str, extra: list[str]) -> list[str]:
    data = DataLayout(args.data_root)
    prelude, recurrent, coda = args.layers
    trainer = [
        "-m", "looptf.megatron.train",
        "--megatron-lm-path", args.megatron_lm_path,
        "--hidden-size", args.hidden_size, "--num-attention-heads", args.num_attention_heads,
        "--prelude-layers", prelude, "--recurrent-layers", recurrent, "--coda-layers", coda,
        "--loops", args.loops, "--backprop-loops", args.backprop_loops,
        *(["--untie-loops"] if args.untie_loops else []),
        *([] if args.no_recompute else ["--recompute-loops"]),
        "--micro-batch-size", args.micro_batch_size, "--global-batch-size", args.global_batch_size,
        "--train-iters", args.train_iters, "--seed", args.seed,
        "--lr-multiplier", args.lr_multiplier, "--lr-schedule", args.schedule,
        "--lr-warmup-iters", args.warmup_iters,
        "--tokenizer-model", tokenizer,
        "--online-raw-manifest", data.train_manifest,
        "--online-validation-manifest", data.val4096 / "manifest.json",
        "--online-data-cache-dir", data.hf_cache,
        "--dataloader-save", run_dir / "dataloader",
        "--save", run_dir / "checkpoints", "--load", run_dir / "checkpoints",
        # Megatron requires a multiple of the save interval.
        "--save-interval", args.save_interval,
        "--save-retain-interval", args.save_interval * (KEEP_LATEST_CHECKPOINT_ONLY // args.save_interval),
        "--eval-interval", args.eval_interval, "--eval-iters", args.eval_iters,
        "--log-interval", args.log_interval,
    ]
    for flag, value in (("--final-eval-iters", args.final_eval_iters),
                        ("--eval-global-batch-size", args.eval_global_batch_size),
                        ("--eval-micro-batch-size", args.eval_micro_batch_size),
                        ("--exit-interval", args.exit_interval)):
        if value:
            trainer += [flag, value]
    if args.wandb_project:
        trainer += ["--wandb-project", args.wandb_project, "--wandb-exp-name", args.run_name,
                    "--wandb-save-dir", run_dir / "wandb", "--tensorboard-dir", run_dir / "tensorboard",
                    "--tensorboard-log-interval", args.log_interval, "--log-timers-to-tensorboard"]
        if args.wandb_entity:
            trainer += ["--wandb-entity", args.wandb_entity]
    if args.nnodes > 1:
        rendezvous = [f"--nnodes={args.nnodes}", f"--node-rank={args.node_rank}",
                      f"--master-addr={args.master_addr}", f"--master-port={args.master_port}"]
    else:
        rendezvous = ["--standalone"]
    torchrun = os.environ.get("TORCHRUN", "torchrun")
    return [torchrun, *rendezvous, f"--nproc-per-node={args.gpus}", *map(str, trainer), *extra]


def worker_environment(args: argparse.Namespace, run_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO_ROOT / "src"), env.get("PYTHONPATH"))))
    env["MEGATRON_LM_PATH"] = args.megatron_lm_path
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env.setdefault("NCCL_NVLS_ENABLE", "0")
    env.setdefault("HF_HOME", str(DataLayout(args.data_root).hf_cache))
    # Ranks import the same sources at once; one rank rewriting __pycache__ can hand another a stale module.
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if args.wandb_project:
        if not env.get("WANDB_API_KEY"):
            raise SystemExit("--wandb-project needs WANDB_API_KEY")
        id_file = run_dir / "wandb_run_id.txt"
        if args.node_rank == 0 and not id_file.exists():
            from wandb.sdk.lib.runid import generate_id

            id_file.write_text(generate_id() + "\n")
        for _ in range(60):
            if id_file.exists():
                break
            time.sleep(1)
        env["WANDB_RUN_ID"] = id_file.read_text().strip()
        env.setdefault("WANDB_RESUME", "allow")
    return env


def prune_dataloader_states(run_dir: Path) -> None:
    """Megatron's --save-retain-interval deletes old checkpoints but not the dataloader states saved with them."""
    latest = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    if not latest.is_file() or not (run_dir / "dataloader").is_dir():
        return
    last = int(latest.read_text().strip())
    for state in (run_dir / "dataloader").glob("iter_*"):
        if int(state.name[5:]) < last and not (run_dir / "checkpoints" / state.name).exists():
            shutil.rmtree(state)


def run(command: list[str], env: dict[str, str], log_path: Path) -> int:
    """Run torchrun in its own process group, copy its output to the log, stop the group on SIGTERM/SIGINT."""
    with log_path.open("a") as log:
        process = subprocess.Popen(command, env=env, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True, text=True, errors="replace", bufsize=1)

        def stop(signum, frame):
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            sys.exit(128 + signum)

        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, stop)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return process.wait()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args, extra = parse_args()
    resolve(args)
    run_dir = Path(args.output_root).resolve() / "runs" / args.run_name
    if (run_dir / "train_done.txt").exists() and not args.dry_run:
        print(f"{args.run_name} already finished ({run_dir / 'train_done.txt'})")
        return
    if args.dry_run:
        print(shlex.join(trainer_command(args, run_dir, args.tokenizer or snapshot_path(args.data_root), extra)))
        return
    tokenizer = prepare_tokenizer(args.tokenizer or "NousResearch/Llama-2-7b-hf", data_root=args.data_root)
    command = trainer_command(args, run_dir, tokenizer, extra)
    log_dir = run_dir / "logs" / ("" if args.node_rank == 0 else f"node{args.node_rank}")
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.node_rank == 0:
        (run_dir / "launch_command.txt").write_text(shlex.join(command) + "\n")
    env = worker_environment(args, run_dir)
    if args.nnodes > 1:
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    log_path = log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    print(f"{args.run_name} on {socket.gethostname()}: {shlex.join(command)}", flush=True)
    rc = run(command, env, log_path)
    if rc != 0:
        sys.exit(rc)
    if args.node_rank == 0:
        prune_dataloader_states(run_dir)
        latest = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
        if latest.is_file() and int(latest.read_text().strip()) == args.train_iters:
            (run_dir / "train_done.txt").write_text(
                f"{args.run_name} finished {time.strftime('%Y-%m-%d %H:%M:%S')} on {socket.gethostname()} "
                f"({args.train_iters} iterations)\n")


if __name__ == "__main__":
    main()
