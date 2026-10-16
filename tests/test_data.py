"""Online data pipeline: tokenization into records, exact resume, prefetching, local shard and validation caches."""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from looptf.data import online
from looptf.data.manifest import read_manifest, resolve_source_mode, source_identity
from looptf.data.online import _build_source_factory, validate_online_data_args
from looptf.data.prefetch import PrefetchBatchLoader
from looptf.data.streaming import StreamingBatchLoader, stable_validation_assignment
from looptf.data.validation_cache import prepare_validation_cache, read_validation_manifest


class FakeTokenizer:
    eos_token_id = 99
    vocab_size = 128

    def __call__(self, texts: list[str], **_: Any) -> dict[str, list[list[int]]]:
        return {"input_ids": [[int(value) for value in text.split()] for text in texts]}


class StatefulRows:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows
        self.position = 0

    def __iter__(self) -> StatefulRows:
        return self

    def __next__(self) -> dict[str, str]:
        if self.position >= len(self.rows):
            raise StopIteration
        row = self.rows[self.position]
        self.position += 1
        return row

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = state["position"]


def make_loader(rows, *, sequence_length=4, batch_size=1, shuffle_buffer=0, shuffle_seed=11, tokenizer_batch=2):
    return StreamingBatchLoader(
        source_factory=lambda: StatefulRows(rows), tokenizer=FakeTokenizer(), split_name="all", text_column="text",
        validation_fraction=0.1, split_seed=7, sequence_length=sequence_length, micro_batch_size=batch_size,
        tokenizer_batch_size=tokenizer_batch, shuffle_buffer_records=shuffle_buffer, shuffle_seed=shuffle_seed,
        create_attention_mask=False, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False,
    )


def shuffled_loader(rows):
    return make_loader(rows, sequence_length=2, shuffle_buffer=3, shuffle_seed=29, tokenizer_batch=1)


def same_batches(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------- streaming loader

def test_documents_are_concatenated_with_eos_into_next_token_records():
    batch = next(make_loader([{"text": "1 2"}, {"text": "3 4 5"}, {"text": "6 7 8 9"}], batch_size=2))
    assert torch.equal(batch["tokens"], torch.tensor([[1, 2, 99, 3], [5, 99, 6, 7]]))
    assert torch.equal(batch["labels"], torch.tensor([[2, 99, 3, 4], [99, 6, 7, 8]]))
    assert torch.equal(batch["loss_mask"], torch.ones(2, 4))
    assert torch.equal(batch["position_ids"], torch.arange(4).expand(2, -1))
    assert "attention_mask" not in batch


def test_resume_restores_source_position_and_unconsumed_tokens():
    rows = [{"text": "1 2 3"}, {"text": "4 5 6"}, {"text": "7 8 9"}, {"text": "10 11 12"}]
    continuous = make_loader(rows)
    assert torch.equal(next(continuous)["tokens"], torch.tensor([[1, 2, 3, 99]]))
    state = continuous.save_state()
    resumed = make_loader(rows)
    resumed.restore_state(state)
    assert same_batches(next(resumed), next(continuous))


def test_resume_restores_shuffle_buffer_and_rng():
    rows = [{"text": f"{index}"} for index in range(1, 12)]
    continuous = shuffled_loader(rows)
    next(continuous)
    state = continuous.save_state()
    resumed = shuffled_loader(rows)
    resumed.restore_state(state)
    for _ in range(5):
        assert torch.equal(next(resumed)["tokens"], next(continuous)["tokens"])


def test_source_failures_are_retried_without_skipping_rows(monkeypatch):
    rows = [{"text": "1 2 3 4"}, {"text": "5 6 7 8"}]
    loader, attempts, delays = make_loader(rows), [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            error = OSError("rate limited")
            error.response = SimpleNamespace(headers={"retry-after": "7"})
            raise error
        return StatefulRows(rows)

    loader.source_factory = flaky
    monkeypatch.setattr("looptf.data.streaming.time.sleep", delays.append)
    assert same_batches(next(loader), next(make_loader(rows)))
    assert len(attempts) == 3 and delays == [7, 7]

    broken = make_loader(rows)
    broken.source_retry_limit = 2
    broken.source_factory = lambda: (_ for _ in ()).throw(OSError("unavailable"))
    monkeypatch.setattr("looptf.data.streaming.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="initialization failed 2 times"):
        next(broken)


def test_validation_assignment_depends_only_on_the_text():
    texts = [f"document {i}" for i in range(20000)]
    selected = [t for t in texts if stable_validation_assignment(t, 0.01, 260421106)]
    assert 120 < len(selected) < 280
    assert selected == [t for t in reversed(texts) if stable_validation_assignment(t, 0.01, 260421106)][::-1]
    assert selected != [t for t in texts if stable_validation_assignment(t, 0.01, 1)]


def test_online_loader_requires_external_dataloader_and_no_workers():
    base = dict(online_validation_fraction=0.001, online_tokenizer_batch_size=8, online_shuffle_buffer_records=0,
                save=None, load=None, dataloader_save=None)
    with pytest.raises(ValueError, match="dataloader-type external"):
        validate_online_data_args(SimpleNamespace(**base, dataloader_type="single", num_workers=0))
    with pytest.raises(ValueError, match="num-workers 0"):
        validate_online_data_args(SimpleNamespace(**base, dataloader_type="external", num_workers=1))


# ---------------------------------------------------------------- prefetching

def wait_full(loader):
    with loader._condition:
        assert loader._condition.wait_for(lambda: len(loader._queue) >= loader.capacity, timeout=5)


@pytest.mark.parametrize("resume_capacity", [0, 1, 4])
def test_prefetched_batches_are_part_of_the_checkpoint(resume_capacity):
    rows = [{"text": str(i)} for i in range(1, 40)]
    loader = PrefetchBatchLoader(shuffled_loader(rows), 4)
    resumed = PrefetchBatchLoader(shuffled_loader(rows), resume_capacity)
    try:
        next(loader)
        wait_full(loader)
        state = loader.save_state()
        assert len(state["train"]["pending_batches"]) == 4
        expected = [next(loader) for _ in range(20)]
        resumed.restore_state(state)
        assert all(same_batches(next(resumed), batch) for batch in expected)
    finally:
        assert loader.close() and resumed.close()


class BlockingSource:
    split_name = "test"

    def __init__(self):
        self.position = 0
        self.entered, self.release = threading.Event(), threading.Event()

    def __next__(self):
        if self.position == 1:
            self.entered.set()
            assert self.release.wait(timeout=10)
        self.position += 1
        return self.position

    def state_dict(self):
        return {"position": self.position}


def test_ready_batch_is_served_while_the_producer_blocks():
    source = BlockingSource()
    loader = PrefetchBatchLoader(source, 2)
    try:
        loader.start()
        assert source.entered.wait(timeout=5)
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(next, loader).result(timeout=2) == 1
        source.release.set()
        assert next(loader) == 2
    finally:
        source.release.set()
        assert loader.close()


def test_checkpoint_waits_for_the_batch_in_flight():
    source = BlockingSource()
    loader = PrefetchBatchLoader(source, 2)
    try:
        loader.start()
        assert source.entered.wait(timeout=5)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(loader.state_dict)
            with loader._condition:
                assert loader._condition.wait_for(lambda: loader._paused, timeout=2)
            assert not future.done()
            source.release.set()
            state = future.result(timeout=5)
        assert state["source"] == {"position": 2} and state["pending_batches"] == [1, 2]
    finally:
        source.release.set()
        assert loader.close()


@pytest.mark.parametrize("fail", [False, True])
def test_end_of_data_and_errors_reach_the_consumer_after_ready_batches(fail):
    class Source:
        split_name = "test"
        index = 0

        def __next__(self):
            self.index += 1
            if self.index <= 2:
                return self.index
            if fail:
                raise OSError("download failed")
            raise StopIteration

    loader = PrefetchBatchLoader(Source(), 4)
    try:
        assert [next(loader), next(loader)] == [1, 2]
        with pytest.raises(RuntimeError if fail else StopIteration):
            next(loader)
    finally:
        assert loader.close()


def test_validation_loader_resumes_with_the_training_loader():
    rows = [{"text": str(i)} for i in range(1, 40)]
    train, val = PrefetchBatchLoader(shuffled_loader(rows), 3), PrefetchBatchLoader(shuffled_loader(rows), 2)
    resumed, resumed_val = PrefetchBatchLoader(shuffled_loader(rows), 1), PrefetchBatchLoader(shuffled_loader(rows), 1)
    try:
        train.attach_checkpoint_peers(validation=val)
        next(train)
        next(val)
        wait_full(train)
        wait_full(val)
        state = train.save_state()
        resumed.attach_checkpoint_peers(validation=resumed_val)
        resumed.restore_state(state)
        for _ in range(8):
            assert torch.equal(next(train)["tokens"], next(resumed)["tokens"])
            assert torch.equal(next(val)["tokens"], next(resumed_val)["tokens"])
    finally:
        for loader in (train, val, resumed, resumed_val):
            assert loader.close()


def test_close_stops_the_producer_and_releases_the_source():
    core = shuffled_loader([{"text": "1 2 3"}])
    loader = PrefetchBatchLoader(core, 2)
    loader.start()
    wait_full(loader)
    assert core._source is not None
    assert loader.close()
    assert core._source is None and core._source_iterator is None
    with pytest.raises(RuntimeError, match="closed"):
        next(loader)


# ---------------------------------------------------------------- local shards

def shard_args(tmp_path, mode="local"):
    return SimpleNamespace(
        online_dataset="org/corpus", online_dataset_subset="sample", online_dataset_revision="a" * 40,
        online_source_split="train", online_source_mode=mode, online_raw_manifest=str(tmp_path / "manifest.json"),
        online_data_cache_dir=str(tmp_path / "metadata"),
    )


def write_manifest(tmp_path, args, files):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"format_version": 1, "source": source_identity(args),
                                "files": [{"path": f.name, "size": f.stat().st_size} for f in files]}))
    return path


@pytest.mark.parametrize("mode", ["local", "auto"])
def test_local_shards_are_split_across_ranks_offline_and_resumable(tmp_path, monkeypatch, mode):
    import datasets
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    args = shard_args(tmp_path, mode)
    files = [tmp_path / f"{i}.parquet" for i in range(4)]
    for i, path in enumerate(files):
        pq.write_table(pa.table({"text": [f"{i}:a", f"{i}:b"]}), path)
    write_manifest(tmp_path, args, files)
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", lambda *a, **k: pytest.fail("contacted the Hub"))
    original_load = datasets.load_dataset

    def local_only(path, **kwargs):
        assert path == "parquet" and kwargs["fragment_scan_options"].pre_buffer is False
        return original_load(path, **kwargs)

    monkeypatch.setattr(datasets, "load_dataset", local_only)
    factories = [_build_source_factory(args, i, 2) for i in range(2)]
    rows = [[row["text"] for row in factory()] for factory in factories]
    assert not set(rows[0]) & set(rows[1]) and len(set(rows[0]) | set(rows[1])) == 8
    source = factories[0]()
    iterator = iter(source)
    next(iterator)
    state, expected = source.state_dict(), list(iterator)
    resumed = factories[0]()
    resumed.load_state_dict(state)
    assert list(resumed) == expected


def test_manifest_must_match_the_dataset_revision_and_file_sizes(tmp_path):
    args = shard_args(tmp_path)
    shard = tmp_path / "one.parquet"
    shard.write_bytes(b"example")
    path = write_manifest(tmp_path, args, [shard])
    assert read_manifest(path, source_identity(args)) == [str(shard)]
    args.online_dataset_revision = "b" * 40
    with pytest.raises(ValueError, match="does not match"):
        _build_source_factory(args, 0, 1)
    args.online_dataset_revision = "a" * 40
    shard.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="missing or incomplete"):
        _build_source_factory(args, 0, 1)


def test_auto_mode_streams_without_a_manifest_and_resolves_once(tmp_path):
    args = shard_args(tmp_path, "auto")
    resolve_source_mode(args)
    assert args.online_source_mode == "stream"
    shard = tmp_path / "one.parquet"
    shard.write_bytes(b"example")
    write_manifest(tmp_path, args, [shard])
    resolve_source_mode(args)
    assert args.online_source_mode == "stream"
    args.online_source_mode = "auto"
    resolve_source_mode(args)
    assert args.online_source_mode == "local"


# ---------------------------------------------------------------- validation cache

def cache_args(tmp_path):
    return SimpleNamespace(**vars(shard_args(tmp_path, "stream")), online_text_column="text",
                           online_validation_fraction=0.2, online_split_seed=7)


def corpus():
    return [{"text": f"{i} 1 2 3 4 5 6 7"} for i in range(1000)]


def prepare(tmp_path, target=12):
    args, source = cache_args(tmp_path), StatefulRows(corpus())
    return args, prepare_validation_cache(args, tmp_path / "validation", target, 1000, lambda: source), source


def test_validation_cache_holds_the_first_held_out_documents(tmp_path):
    import pyarrow.parquet as pq

    args, path, source = prepare(tmp_path)
    files, manifest = read_validation_manifest(path, args)
    expected = [r["text"] for r in corpus() if stable_validation_assignment(r["text"], 0.2, 7)][:12]
    assert pq.read_table(files[0]).to_pydict()["text"] == expected
    assert source.position == manifest["source_documents_scanned"] < 1000 and manifest["documents"] == 12
    assert prepare_validation_cache(args, path.parent, 12, 1000, lambda: pytest.fail("reread the source")) == path


def test_validation_cache_is_not_published_when_incomplete(tmp_path):
    source = StatefulRows(corpus())
    with pytest.raises(ValueError, match="no cache published"):
        prepare_validation_cache(cache_args(tmp_path), tmp_path / "validation", 10, 10, lambda: source)
    assert not (tmp_path / "validation/manifest.json").exists()


@pytest.mark.parametrize("field,value", [("online_split_seed", 8), ("online_validation_fraction", 0.1),
                                         ("online_dataset_revision", "b" * 40)])
def test_validation_cache_rejects_another_split(tmp_path, field, value):
    args, path, _ = prepare(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="does not match"):
        read_validation_manifest(path, args)


def test_validation_cache_rejects_corrupted_files(tmp_path):
    args, path, _ = prepare(tmp_path)
    shard = path.parent / "validation.parquet"
    content = bytearray(shard.read_bytes())
    content[20] ^= 1
    shard.write_bytes(content)
    with pytest.raises(ValueError, match="checksum"):
        read_validation_manifest(path, args)


def test_training_streams_while_validation_reads_the_cache(tmp_path, monkeypatch):
    import datasets
    from transformers import AutoTokenizer

    args, path, _ = prepare(tmp_path)
    args.__dict__.update(
        online_validation_manifest=str(path), tokenizer_model="fake", online_tokenizer_revision="pinned",
        seq_length=4, micro_batch_size=1, eval_micro_batch_size=1, online_tokenizer_batch_size=2,
        online_shuffle_buffer_records=0, seed=42, create_attention_mask_in_dataloader=False,
        reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False, dataloader_save=None,
        online_prefetch_batches=0,
    )
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0, get_data_parallel_rank=lambda: 0,
                               get_data_parallel_world_size=lambda: 1)
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)

    class Tokenizer(FakeTokenizer):
        def __len__(self):
            return 128

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    original_load, requests = datasets.load_dataset, []

    def load(path, **kwargs):
        requests.append(path)
        if path == "parquet":
            return original_load(path, **kwargs)
        return datasets.Dataset.from_list(corpus()).to_iterable_dataset()

    monkeypatch.setattr(datasets, "load_dataset", load)
    train, validation, _ = online.build_online_dataloaders(args)
    try:
        next(validation)
        assert requests == ["parquet"]
        next(train)
        assert requests == ["parquet", "org/corpus"]
        state = train.save_state()
        expected = next(train), next(validation)
        train.restore_state(state)
        assert same_batches(expected[0], next(train)) and same_batches(expected[1], next(validation))
    finally:
        train.close()
        validation.close()
        online._ACTIVE_LOADERS.remove(train)
        online._ACTIVE_LOADERS.remove(validation)


def test_tokenizer_snapshot_path_is_stable(tmp_path):
    from looptf.data.tokenizer import snapshot_path

    assert Path(snapshot_path(str(tmp_path))).name == "llama2_32001_3cfa8358de44148a"
