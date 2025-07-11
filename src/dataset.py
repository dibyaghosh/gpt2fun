import concurrent.futures
from dataclasses import dataclass

import numpy as np
from etils import epath, epy

with epy.lazy_imports():
    import jax


@dataclass
class Worker:
    worker_id: int
    num_workers: int

    @classmethod
    def current(cls):
        return cls(jax.process_index(), jax.process_count())


class DataShard:
    def __init__(self, file: epath.PathLike):
        self.file = epath.Path(file)
        self.f = None

    def __enter__(self):
        self.f = self.file.open(mode="rb")
        header = np.frombuffer(self.f.read(3 * 4), dtype=np.uint32)
        assert header[0] == 20240520
        assert header[1] == 1
        self.num_tokens = header[2]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()
        self.f = None

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, idx: slice | int):
        assert self.f is not None, "File not open. Use with context manager."
        if isinstance(idx, int):
            return self[slice(idx, idx + 1)][0]
        assert isinstance(idx, slice)
        start = idx.start or 0
        stop = idx.stop or len(self)
        step = idx.step or 1
        num_to_read = stop - start
        self.f.seek(256 * 4 + start * 2)
        return np.frombuffer(self.f.read(num_to_read * 2), dtype=np.uint16)[::step]


def _load_data_shard(file: epath.Path, *, worker: Worker, chunk_size: int):
    with DataShard(file) as shard:
        tokens = []
        num_rounds = len(shard) // (chunk_size * worker.num_workers)
        current_token = worker.worker_id * chunk_size
        for round in range(num_rounds):
            tokens.append(shard[current_token : current_token + chunk_size])
            current_token += chunk_size * worker.num_workers
        return np.concatenate(tokens)


def load(
    directory: epath.Path, worker: Worker | None, num_tokens: int, chunk_size: int
):
    if worker is None:
        worker = Worker(0, 1)
    num_tokens = num_tokens // worker.num_workers

    per_shard = (100_000_000 // (chunk_size * worker.num_workers)) * chunk_size
    num_shards = int(np.ceil(num_tokens / per_shard))
    assert num_tokens <= per_shard * num_shards
    buffer = np.zeros(num_tokens, dtype=np.uint16)
    assert num_shards <= 1020, "Too many shards."

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                _load_data_shard,
                directory / f"fineweb_train_{shard_id:06d}.bin",
                worker=worker,
                chunk_size=chunk_size,
            )
            for shard_id in range(1, num_shards + 1)
        ]
        start = 0
        for n, future in enumerate(futures):
            shard_data = future.result()
            buffer[start : start + shard_data.shape[0]] = shard_data[
                : len(buffer) - start
            ]
            start += shard_data.shape[0]
            del shard_data
            print(f"Loaded shard {n + 1} of {len(futures)}: {start} tokens")

    return buffer[:start]


class ShardedDataset:
    def __init__(
        self,
        directory: epath.Path,
        num_tokens: int,
        *,
        batch_size: int,
        seq_len: int,
        sharding,
        shuffle: bool = True,
    ):
        self.worker = Worker.current()
        self.batch_size = batch_size // self.worker.num_workers
        self.seq_len = seq_len
        self.buffer = load(
            directory,
            worker=self.worker,
            num_tokens=num_tokens,
            chunk_size=self.batch_size * self.seq_len,
        )

        self.sharding = sharding
        self.ordering = np.arange(self.buffer.shape[0] // self.seq_len)
        if shuffle:
            rng = np.random.default_rng(0)
            rng.shuffle(self.ordering)

    def __len__(self):
        return self.buffer.shape[0] // (self.batch_size * self.seq_len)

    def __getitem__(self, idx):
        seq_ids = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch = np.stack(
            [
                self.buffer[
                    self.ordering[seq_id] * self.seq_len : (self.ordering[seq_id] + 1)
                    * self.seq_len
                ]
                for seq_id in seq_ids
            ]
        )
        return jax.make_array_from_process_local_data(self.sharding, batch)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
