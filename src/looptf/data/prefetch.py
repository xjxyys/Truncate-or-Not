"""Bounded CPU batch read-ahead with exact external-loader checkpoints."""
from __future__ import annotations

import copy
import threading
import time
from collections import deque
from typing import Any


class PrefetchBatchLoader:
    """One producer owns the source; the training thread only dequeues batches.

    Snapshots pause the producer at a batch boundary and save both its advanced
    source state and every unconsumed batch. No per-batch source-state copies
    are needed. A checkpoint can wait for an in-flight source read to finish.
    Only one consumer/checkpoint caller is supported, as in Megatron.
    """

    def __init__(self, loader: Any, capacity: int = 16) -> None:
        if capacity < 0:
            raise ValueError("prefetch capacity must be non-negative")
        self.loader = loader
        self.capacity = capacity
        self._queue: deque[Any] = deque()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._active = False
        self._paused = False
        self._closed = False
        self._exhausted = False
        self._error: BaseException | None = None
        self._peers: dict[str, PrefetchBatchLoader] = {}
        self._wait_seconds = 0.0
        self._empty_waits = 0
        self._consumed = 0
        self._last_wait_log = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)

    def __iter__(self) -> PrefetchBatchLoader:
        return self

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("prefetch loader is closed")
            if self.capacity and self._thread is None and not self._exhausted:
                self._thread = threading.Thread(
                    target=self._produce, name=f"online-{self.loader.split_name}", daemon=True
                )
                self._thread.start()

    def _produce(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed
                    or (not self._paused and len(self._queue) < self.capacity)
                )
                if self._closed:
                    return
                self._active = True
            # Never hold the queue lock over I/O or tokenization: ready batches
            # must remain consumable while the producer is slow or reconnects.
            batch = None
            error = None
            exhausted = False
            try:
                batch = next(self.loader)
            except StopIteration:
                exhausted = True
            except BaseException as exc:
                error = exc
            with self._condition:
                self._active = False
                self._error = error
                self._exhausted = exhausted
                if not exhausted and error is None and not self._closed:
                    self._queue.append(batch)
                self._condition.notify_all()
                if exhausted or error is not None or self._closed:
                    return

    def __next__(self) -> Any:
        self.start()
        with self._condition:
            if not self.capacity and not self._queue:
                if self._exhausted:
                    raise StopIteration
                batch = next(self.loader)
                self._consumed += 1
                return batch
            if not self._queue and not self._exhausted and self._error is None:
                started = time.monotonic()
                self._empty_waits += 1
                self._condition.wait_for(
                    lambda: bool(self._queue) or self._error is not None
                    or self._exhausted or self._closed
                )
                waited = time.monotonic() - started
                self._wait_seconds += waited
                if waited >= 1.0 and time.monotonic() - self._last_wait_log >= 60:
                    print(
                        f"online {self.loader.split_name} data wait: {waited:.3f}s; "
                        f"total={self._wait_seconds:.3f}s, empty_waits={self._empty_waits}",
                        flush=True,
                    )
                    self._last_wait_log = time.monotonic()
            if self._closed:
                raise RuntimeError("prefetch loader is closed")
            if self._queue:
                batch = self._queue.popleft()
                self._consumed += 1
                self._condition.notify_all()
                return batch
            if self._error is not None:
                raise RuntimeError("online background data producer failed") from self._error
            raise StopIteration

    def stats(self) -> dict[str, float | int]:
        with self._condition:
            return {
                "queued_batches": len(self._queue), "consumed_batches": self._consumed,
                "empty_waits": self._empty_waits, "wait_seconds": self._wait_seconds,
            }

    def state_dict(self) -> dict[str, Any]:
        with self._condition:
            self._paused = True
            try:
                self._condition.wait_for(lambda: not self._active)
                if self._error is not None:
                    raise RuntimeError("cannot checkpoint a failed data producer") from self._error
                return {
                    "prefetch_format_version": 1,
                    "source": copy.deepcopy(self.loader.state_dict()),
                    "pending_batches": copy.deepcopy(list(self._queue)),
                    "exhausted": self._exhausted,
                }
            finally:
                self._paused = False
                self._condition.notify_all()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with self._condition:
            if self._thread is not None or self._closed:
                raise RuntimeError("restore data state before starting the prefetch loader")
            if "prefetch_format_version" in state:
                if state["prefetch_format_version"] != 1:
                    raise ValueError("unsupported prefetch checkpoint format")
                self.loader.load_state_dict(copy.deepcopy(state["source"]))
                self._queue = deque(copy.deepcopy(state["pending_batches"]))
                self._exhausted = state["exhausted"]
            else:
                # Existing synchronous-loader checkpoints remain supported.
                self.loader.load_state_dict(copy.deepcopy(state))
                self._queue.clear()
                self._exhausted = False

    def attach_checkpoint_peers(self, **peers: PrefetchBatchLoader) -> None:
        self._peers = dict(peers)

    def save_state(self) -> dict[str, Any]:
        return {
            "train": self.state_dict(),
            "peers": {name: peer.state_dict() for name, peer in self._peers.items()},
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self.load_state_dict(state.get("train", state))
        for name, peer in self._peers.items():
            if name in state.get("peers", {}):
                peer.load_state_dict(state["peers"][name])

    def close(self, timeout: float = 5.0) -> bool:
        """Stop read-ahead; return False if a network call has not returned yet."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                return False
        close = getattr(self.loader, "close", None)
        if close is not None:
            close()
        return True
