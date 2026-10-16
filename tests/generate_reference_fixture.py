"""Record tests/fixtures/reference_block.json from the authors' released model code.

    python tests/generate_reference_fixture.py <checkout of kschwethelm/looped-lm-scaling>

Only the released nanochat/gpt.py computes the outputs; this repository supplies the deterministic weights and
inputs (tests/test_model.py: reference_weights).
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_model import reference_weights  # noqa: E402

SOURCE_COMMIT = "122366472a31e57982a2877eeee10bb115477713"

root = Path(sys.argv[1])
for file in ("nanochat/gpt.py", "nanochat/flash_attention.py"):
    expected = subprocess.check_output(["git", "-C", str(root), "show", f"{SOURCE_COMMIT}:{file}"])
    assert expected == (root / file).read_bytes(), f"source differs from {SOURCE_COMMIT}: {file}"
sys.path.insert(0, str(root))
from nanochat.gpt import GPT, GPTConfig  # noqa: E402

torch.set_num_threads(1)
model = GPT(GPTConfig(n_embd=256, n_head=2, n_kv_head=2, n_prelude=0, n_recur_block=2, n_coda=0, num_recur=1,
                      bptt_k=1, input_injection="passthrough", vocab_size=19, sequence_len=8))
model.load_state_dict(reference_weights(), strict=True)
ids = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
labels = torch.tensor([[2, 3, 4, 5], [5, 6, 7, 8]])
logits = model(ids)
loss = model(ids, targets=labels)
loss.backward()


def summary(p):
    grad = p.grad.float().flatten()
    return {"norm": grad.norm().item(), "samples": grad[torch.linspace(0, grad.numel() - 1, 12).long()].tolist()}


fixture = {"source_commit": SOURCE_COMMIT, "ids": ids.tolist(), "labels": labels.tolist(),
           "logits": logits.detach().tolist(), "loss": loss.item(),
           "gradients": {name: summary(p) for name, p in model.named_parameters()}}
path = Path(__file__).parent / "fixtures" / "reference_block.json"
path.write_text(json.dumps(fixture, indent=2) + "\n")
print(path)
