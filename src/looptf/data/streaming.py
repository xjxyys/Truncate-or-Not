"""Tokenize a document stream into fixed-length records, with resumable state; no Megatron dependency."""
from __future__ import annotations

import hashlib
import os
import random
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch


def stable_validation_assignment(text: str, validation_fraction: float, seed: int) -> bool:
    """Assign a document to validation independently of stream ordering."""
    payload = seed.to_bytes(8, "little", signed=False) + text.encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return value < int(validation_fraction * (1 << 64))


class StreamingBatchLoader:
    """Tokenize and pack a sharded Hugging Face text stream during training.

    The object is both iterable and iterator because Megatron's external-data
    path calls ``next()`` directly on the value returned by its dataset
    provider. Source examples are tokenized in CPU batches, appended with EOS,
    concatenated into a continuous token stream, and emitted as fixed
    ``sequence_length + 1`` records.

    This core iterator is synchronous; PrefetchBatchLoader owns it during
    asynchronous training. Its ``save_state()`` describes
    exactly the next batch that will be returned: the Hugging Face stream state
    records all source rows already read, while ``token_buffer`` records
    tokenized data read but not yet consumed by the trainer.
    """

    def __init__(
        self,
        *,
        source_factory: Callable[[], Iterable[dict[str, Any]]],
        tokenizer: Any,
        split_name: str,
        text_column: str,
        validation_fraction: float,
        split_seed: int,
        sequence_length: int,
        micro_batch_size: int,
        tokenizer_batch_size: int,
        shuffle_buffer_records: int,
        shuffle_seed: int,
        create_attention_mask: bool,
        reset_position_ids: bool,
        reset_attention_mask: bool,
        eod_mask_loss: bool,
        repeat: bool = True,
    ) -> None:
        if split_name not in {"train", "validation", "all"}:
            raise ValueError(f"unsupported online split: {split_name}")
        if not 0.0 <= validation_fraction <= 1.0:
            raise ValueError("validation_fraction must be in [0, 1]")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive")
        if tokenizer_batch_size < 1:
            raise ValueError("tokenizer_batch_size must be positive")
        if shuffle_buffer_records < 0:
            raise ValueError("shuffle_buffer_records must be non-negative")
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no eos_token_id")

        self.source_factory = source_factory
        self.tokenizer = tokenizer
        self.split_name = split_name
        self.text_column = text_column
        self.validation_fraction = validation_fraction
        self.split_seed = split_seed
        self.sequence_length = sequence_length
        self.record_length = sequence_length + 1
        self.micro_batch_size = micro_batch_size
        self.tokenizer_batch_size = tokenizer_batch_size
        # Consecutive failed source reads tolerated before giving up. Streaming
        # runs for hours over HTTP, so transient faults are expected; env
        # override lets a flaky pod raise it without a code change.
        self.source_retry_limit = max(1, int(os.environ.get("ONLINE_SOURCE_RETRY_LIMIT", "8")))
        self.shuffle_buffer_records = shuffle_buffer_records
        self.create_attention_mask = create_attention_mask
        self.reset_position_ids = reset_position_ids
        self.reset_attention_mask = reset_attention_mask
        self.eod_mask_loss = eod_mask_loss
        self.repeat = repeat

        self._source: Any = None
        self._source_iterator: Iterator[dict[str, Any]] | None = None
        self._token_buffer: deque[int] = deque()
        self._record_shuffle_buffer: list[torch.Tensor] = []
        self._shuffle_rng = random.Random(shuffle_seed)
        self._epoch = 0
        self._source_exhausted = False
        self._accepted_documents_this_epoch = 0
        self._source_documents_seen = 0
        self._records_yielded = 0
        self._batches_yielded = 0
        self._checkpoint_peers: dict[str, StreamingBatchLoader] = {}
        self.source_identity: dict[str, Any] | None = None

    def __iter__(self) -> StreamingBatchLoader:
        return self

    def close(self) -> None:
        iterator, self._source_iterator = self._source_iterator, None
        try:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        finally:
            self._source = None

    def _ensure_source(self) -> None:
        if self._source is None:
            # Metadata resolution can be rate-limited before the first row is
            # read. Give it the same bounded recovery as subsequent reads.
            for attempt in range(self.source_retry_limit):
                try:
                    self._source = self.source_factory()
                    break
                except Exception as error:
                    if attempt + 1 == self.source_retry_limit:
                        raise RuntimeError(
                            f"online {self.split_name} source initialization failed "
                            f"{self.source_retry_limit} times"
                        ) from error
                    delay = min(2**attempt, 30)
                    response = getattr(error, "response", None)
                    retry_after = getattr(response, "headers", {}).get("retry-after", "")
                    if str(retry_after).isdigit():
                        delay = max(delay, min(int(retry_after), 300))
                    print(
                        f"online {self.split_name} source initialization failed "
                        f"({type(error).__name__}: {error}); retrying in {delay}s "
                        f"[attempt {attempt + 1}/{self.source_retry_limit}]",
                        flush=True,
                    )
                    time.sleep(delay)
            if not hasattr(self._source, "state_dict") or not hasattr(
                self._source, "load_state_dict"
            ):
                raise TypeError(
                    "online source must implement state_dict() and load_state_dict() "
                    "for exact checkpoint resume"
                )
            self._source_iterator = iter(self._source)

    def _belongs_to_split(self, text: str) -> bool:
        if self.split_name == "all":
            return True
        is_validation = stable_validation_assignment(
            text,
            self.validation_fraction,
            self.split_seed,
        )
        return is_validation if self.split_name == "validation" else not is_validation

    def _reconnect_source(self) -> None:
        """Rebuild the HTTP stream at the current position after a transport error.

        Streaming reads the dataset over long-lived HTTP range requests, so a
        transient failure (observed: HTTP 499 from the hub after ~200 training
        iterations) raises out of ``next()`` and kills the rank -- and one dead
        rank ends the whole job. The source implements state_dict/load_state_dict
        for checkpoint resume, so the same mechanism can restore position here
        without re-reading from the beginning or skipping records.
        """
        assert self._source is not None
        position = self._source.state_dict()
        self._source = self.source_factory()
        self._source.load_state_dict(position)
        self._source_iterator = iter(self._source)

    def _next_source_row(self) -> Any:
        """Return the next source row, retrying transport errors.

        StopIteration means the split is exhausted and must propagate to the
        caller's epoch handling. Anything else is treated as a transport fault:
        retried with backoff, and only raised once the retries are spent.
        """
        last_error: Exception | None = None
        for attempt in range(self.source_retry_limit):
            try:
                return next(self._source_iterator)
            except StopIteration:
                raise
            except Exception as error:  # noqa: BLE001 -- any transport fault
                last_error = error
                delay = min(2**attempt, 30)
                print(
                    f"online {self.split_name} source read failed "
                    f"({type(error).__name__}: {error}); reconnecting in {delay}s "
                    f"[attempt {attempt + 1}/{self.source_retry_limit}]",
                    flush=True,
                )
                time.sleep(delay)
                try:
                    self._reconnect_source()
                except Exception as reconnect_error:  # noqa: BLE001
                    print(
                        f"online {self.split_name} reconnect failed "
                        f"({type(reconnect_error).__name__}: {reconnect_error})",
                        flush=True,
                    )
        raise RuntimeError(
            f"online {self.split_name} source failed "
            f"{self.source_retry_limit} consecutive reads"
        ) from last_error

    def _tokenize_more(self) -> None:
        self._ensure_source()
        assert self._source_iterator is not None
        texts: list[str] = []
        source_rows_read = 0
        while source_rows_read < self.tokenizer_batch_size:
            try:
                row = self._next_source_row()
            except StopIteration:
                self._source_exhausted = True
                break
            source_rows_read += 1
            self._source_documents_seen += 1
            text = row.get(self.text_column)
            if not isinstance(text, str) or not text or not self._belongs_to_split(text):
                continue
            texts.append(text)

        if not texts:
            return
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        eos_token_id = int(self.tokenizer.eos_token_id)
        for token_ids in encoded:
            self._token_buffer.extend(int(token_id) for token_id in token_ids)
            self._token_buffer.append(eos_token_id)
            self._accepted_documents_this_epoch += 1

    def _restart_source(self) -> None:
        if not self.repeat:
            raise StopIteration
        if self._accepted_documents_this_epoch == 0:
            raise RuntimeError(
                f"online {self.split_name} split produced no documents in one source pass"
            )
        self._token_buffer.clear()
        self._epoch += 1
        self._source = None
        self._source_iterator = None
        self._source_exhausted = False
        self._accepted_documents_this_epoch = 0

    def _next_unshuffled_record(self) -> torch.Tensor:
        while len(self._token_buffer) < self.record_length:
            if self._source_exhausted:
                if (self._records_yielded == 0 and self.source_identity
                        and "validation_cache" in self.source_identity):
                    raise ValueError(
                        "offline validation has too few tokens on this data-parallel rank; "
                        "prepare a larger validation cache"
                    )
                self._restart_source()
            self._tokenize_more()
        self._records_yielded += 1
        return torch.tensor(
            [self._token_buffer.popleft() for _ in range(self.record_length)],
            dtype=torch.long,
        )

    def _next_record(self) -> torch.Tensor:
        if self.shuffle_buffer_records <= 1:
            return self._next_unshuffled_record()
        while len(self._record_shuffle_buffer) < self.shuffle_buffer_records:
            self._record_shuffle_buffer.append(self._next_unshuffled_record())
        selected = self._shuffle_rng.randrange(len(self._record_shuffle_buffer))
        record = self._record_shuffle_buffer[selected]
        self._record_shuffle_buffer[selected] = self._next_unshuffled_record()
        return record

    def _build_position_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(self.sequence_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(self.micro_batch_size, -1).clone()
        if not self.reset_position_ids:
            return position_ids
        eos = int(self.tokenizer.eos_token_id)
        for batch_index in range(self.micro_batch_size):
            previous_index = 0
            for index in torch.nonzero(tokens[batch_index].eq(eos), as_tuple=False).flatten():
                boundary = int(index) + 1
                position_ids[batch_index, boundary:] -= boundary - previous_index
                previous_index = boundary
        return position_ids

    def _build_attention_mask(self, tokens: torch.Tensor) -> torch.Tensor | None:
        if not self.create_attention_mask:
            return None
        attention_mask = torch.triu(
            torch.ones(
                self.sequence_length,
                self.sequence_length,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = attention_mask.expand(self.micro_batch_size, -1, -1, -1).clone()
        if not self.reset_attention_mask:
            return attention_mask
        eos = int(self.tokenizer.eos_token_id)
        for batch_index in range(self.micro_batch_size):
            for index in torch.nonzero(tokens[batch_index].eq(eos), as_tuple=False).flatten():
                boundary = int(index) + 1
                attention_mask[batch_index, 0, boundary:, :boundary] = True
        return attention_mask

    def __next__(self) -> dict[str, torch.Tensor]:
        records = [self._next_record() for _ in range(self.micro_batch_size)]
        packed = torch.stack(records)
        tokens = packed[:, :-1].contiguous()
        labels = packed[:, 1:].contiguous()
        loss_mask = torch.ones_like(tokens, dtype=torch.float32)
        if self.eod_mask_loss:
            loss_mask.masked_fill_(tokens.eq(int(self.tokenizer.eos_token_id)), 0.0)
        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": self._build_position_ids(tokens),
        }
        attention_mask = self._build_attention_mask(tokens)
        if attention_mask is not None:
            batch["attention_mask"] = attention_mask
        self._batches_yielded += 1
        return batch

    def attach_checkpoint_peers(
        self,
        **peers: StreamingBatchLoader,
    ) -> None:
        """Include validation/test loader state in the train-loader sidecar."""
        self._checkpoint_peers = dict(peers)

    def state_dict(self) -> dict[str, Any]:
        source_state = None
        if self._source is not None and hasattr(self._source, "state_dict"):
            source_state = self._source.state_dict()
        return {
            "format_version": 1,
            "source_identity": self.source_identity,
            "split_name": self.split_name,
            "epoch": self._epoch,
            "source_state": source_state,
            "source_exhausted": self._source_exhausted,
            "accepted_documents_this_epoch": self._accepted_documents_this_epoch,
            "source_documents_seen": self._source_documents_seen,
            "records_yielded": self._records_yielded,
            "batches_yielded": self._batches_yielded,
            "token_buffer": list(self._token_buffer),
            "record_shuffle_buffer": [
                record.clone() for record in self._record_shuffle_buffer
            ],
            "shuffle_rng_state": self._shuffle_rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_identity = state.get("source_identity")
        if saved_identity is not None and saved_identity != self.source_identity:
            raise ValueError("online checkpoint source/layout differs from the configured loader")
        if saved_identity is None and self.source_identity and self.source_identity["mode"] == "local":
            raise ValueError("legacy streaming checkpoints cannot be restored into the local cache")
        if int(state.get("format_version", 1)) != 1:
            raise ValueError("unsupported online dataloader checkpoint format")
        if state.get("split_name", self.split_name) != self.split_name:
            raise ValueError(
                f"checkpoint split {state.get('split_name')!r} does not match "
                f"loader split {self.split_name!r}"
            )
        self._epoch = int(state.get("epoch", 0))
        self._source_exhausted = bool(state.get("source_exhausted", False))
        self._accepted_documents_this_epoch = int(
            state.get("accepted_documents_this_epoch", 0)
        )
        self._source_documents_seen = int(state.get("source_documents_seen", 0))
        self._records_yielded = int(state.get("records_yielded", 0))
        self._batches_yielded = int(state.get("batches_yielded", 0))
        self._token_buffer = deque(int(value) for value in state.get("token_buffer", ()))
        self._record_shuffle_buffer = [
            torch.as_tensor(record, dtype=torch.long).clone()
            for record in state.get("record_shuffle_buffer", ())
        ]
        if state.get("shuffle_rng_state") is not None:
            self._shuffle_rng.setstate(state["shuffle_rng_state"])
        self._source = None
        self._source_iterator = None
        source_state = state.get("source_state")
        if source_state is not None:
            self._ensure_source()
            if not hasattr(self._source, "load_state_dict"):
                raise TypeError("online source does not implement load_state_dict()")
            self._source.load_state_dict(source_state)
            self._source_iterator = iter(self._source)

    def save_state(self) -> dict[str, Any]:
        """Return the state format expected by Megatron's dataloader sidecar."""
        return {
            "train": self.state_dict(),
            "peers": {
                name: loader.state_dict()
                for name, loader in self._checkpoint_peers.items()
            },
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore this loader and any attached validation/test peers."""
        if "train" not in state:
            self.load_state_dict(state)
            return
        self.load_state_dict(state["train"])
        peer_states = state.get("peers", {})
        for name, loader in self._checkpoint_peers.items():
            if name in peer_states:
                loader.load_state_dict(peer_states[name])


