"""The pinned Llama-2 tokenizer plus <|assistant_end|> (32,001 tokens), saved once as a local snapshot."""

from looptf.model.config import VOCAB_SIZE

TOKENIZER_MODEL = "NousResearch/Llama-2-7b-hf"
TOKENIZER_REVISION = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"


def validate_tokenizer(path):
    from pathlib import Path

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
    if tokenizer.get_vocab_size() != VOCAB_SIZE or any(
        tokenizer.token_to_id(token) != index
        for token, index in (("<unk>", 0), ("<s>", 1), ("</s>", 2), ("<|assistant_end|>", 32000))
    ):
        raise ValueError("expected the Llama-2 tokenizer plus <|assistant_end|> (32,001 tokens)")


def _identity(model: str = TOKENIZER_MODEL, revision: str = TOKENIZER_REVISION) -> dict:
    return {"model": model, "revision": revision, "added_token": "<|assistant_end|>", "format_version": 2}


def snapshot_path(data_root: str = "data") -> str:
    """Where prepare_tokenizer saves the pinned tokenizer."""
    import hashlib
    import json
    from pathlib import Path

    key = hashlib.sha256(json.dumps(_identity(), sort_keys=True).encode()).hexdigest()[:16]
    return str(Path(data_root).resolve() / "tokenizers" / f"llama2_32001_{key}")


def prepare_tokenizer(model: str = TOKENIZER_MODEL, revision: str = TOKENIZER_REVISION, data_root: str = "data") -> str:
    """Save the tokenizer under <data_root>/tokenizers once (atomically across nodes) and return its path.

    Megatron and the online loader both read this snapshot; `model` may also be an existing snapshot directory.
    """
    import json
    import tempfile
    from pathlib import Path

    from filelock import FileLock

    if Path(model).is_dir():
        validate_tokenizer(model)
        return str(Path(model).resolve())
    if model != TOKENIZER_MODEL or revision != TOKENIZER_REVISION:
        raise ValueError(f"expected the pinned tokenizer {TOKENIZER_MODEL} at revision {TOKENIZER_REVISION}")
    # The snapshot keeps the raw `tokenizers` pipeline of the reference implementation: AutoTokenizer may convert
    # Llama's normalizer and pre-tokenizer depending on the Transformers version, which changes how leading spaces
    # and embedded special tokens are encoded.
    identity = _identity(model, revision)
    target = Path(snapshot_path(data_root))
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock", timeout=600):
        if target.exists():
            if json.loads((target / "source.json").read_text()) != identity:
                raise ValueError(f"tokenizer source receipt mismatch: {target}")
            validate_tokenizer(target)
            return str(target.resolve())
        from huggingface_hub import hf_hub_download
        from tokenizers import AddedToken, Tokenizer
        from transformers import PreTrainedTokenizerFast

        raw_path = hf_hub_download(
            model, "tokenizer.json", revision=revision,
            cache_dir=str(Path(data_root) / "cache/huggingface"),
        )
        raw = Tokenizer.from_file(raw_path)
        if raw.get_vocab_size() != 32000:
            raise ValueError("expected 32000 tokens before adding assistant_end")
        raw.add_special_tokens([AddedToken("<|assistant_end|>", special=True)])
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
            additional_special_tokens=["<|assistant_end|>"],
        )
        with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=parent) as temp:
            snapshot = Path(temp) / "tokenizer"
            tokenizer.save_pretrained(snapshot)
            # Transformers 5 writes TokenizersBackend here; PreTrainedTokenizerFast is also readable by 4.57.
            config_path = snapshot / "tokenizer_config.json"
            config = json.loads(config_path.read_text())
            config["tokenizer_class"] = "PreTrainedTokenizerFast"
            # 5.x serializes a list here, while 4.57 expects a dictionary for
            # model-specific tokens. The shared additional_special_tokens key
            # represents this ordinary added stop token on both versions.
            config.pop("extra_special_tokens", None)
            config["additional_special_tokens"] = ["<|assistant_end|>"]
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            expected = json.loads(raw.to_str())
            actual = json.loads((snapshot / "tokenizer.json").read_text())
            for field in ("model", "normalizer", "pre_tokenizer", "decoder", "added_tokens"):
                if actual[field] != expected[field]:
                    raise ValueError(f"tokenizer serialization changed the source {field}")
            validate_tokenizer(snapshot)
            (snapshot / "source.json").write_text(json.dumps(identity, indent=2) + "\n")
            try:
                snapshot.rename(target)
            except FileExistsError:
                # Some shared mounts don't coordinate advisory locks across
                # clients. Another node may have atomically published the same
                # complete snapshot while this node prepared its private copy.
                if json.loads((target / "source.json").read_text()) != identity:
                    raise ValueError(f"tokenizer source receipt mismatch: {target}") from None
                validate_tokenizer(target)
    return str(target.resolve())
