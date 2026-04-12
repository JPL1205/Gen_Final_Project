from typing import *

import os
import random

import pyarrow.parquet as pq


def _normalize_cell_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        # Keep "__key__" as string for compatibility with existing comments/usages.
        return value if key == "__key__" else value.encode("utf-8")
    return value


class ParquetChunkDataSource:
    """
    Lightweight parquet row iterator used by ChunkedDataset.

    Args:
        file_dir: Directory that contains parquet files.
        file_name: Prefix used to filter shards (e.g. "GObjaverse-train-280k-83k").
    """
    def __init__(self, file_dir: str, file_name: Optional[str] = None):
        self.file_dir = file_dir
        self.file_name = file_name
        self._parquet_files = self._discover_files(file_dir, file_name)
        self._num_rows: Optional[int] = None

        if len(self._parquet_files) == 0:
            raise FileNotFoundError(
                f"No parquet shards found in [{file_dir}] for prefix [{file_name}]"
            )

    def __len__(self) -> int:
        if self._num_rows is None:
            self._num_rows = 0
            for path in self._parquet_files:
                self._num_rows += pq.ParquetFile(path).metadata.num_rows
        return self._num_rows

    @property
    def parquet_files(self) -> List[str]:
        return self._parquet_files

    @staticmethod
    def _discover_files(file_dir: str, file_name: Optional[str]) -> List[str]:
        if os.path.isfile(file_dir):
            return [file_dir]

        if not os.path.isdir(file_dir):
            raise FileNotFoundError(f"Data directory does not exist: [{file_dir}]")

        files: List[str] = []
        for name in sorted(os.listdir(file_dir)):
            path = os.path.join(file_dir, name)
            if not os.path.isfile(path):
                continue

            # Prefer explicit parquet files, but also accept exact/prefix matches
            # for legacy shard names.
            is_parquet = name.endswith(".parquet")
            prefix_match = (file_name is None) or name.startswith(file_name)
            exact_match = (file_name is not None) and (name == file_name)
            if (is_parquet and prefix_match) or exact_match:
                files.append(path)

        return files

    def iter_rows(
        self,
        shard_id: int = 0,
        num_shards: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        epoch: int = 0,
    ) -> Iterator[Dict[str, Any]]:
        if num_shards <= 0:
            raise ValueError(f"Invalid num_shards: {num_shards}")
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(f"Invalid shard_id [{shard_id}] for num_shards [{num_shards}]")

        files = list(self._parquet_files)
        if shuffle:
            rng = random.Random(seed + epoch)
            rng.shuffle(files)

        files = files[shard_id::num_shards]

        for path in files:
            parquet_file = pq.ParquetFile(path)
            for rg_idx in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(rg_idx)
                rows = table.to_pylist()
                for row_idx, row in enumerate(rows):
                    sample: Dict[str, Any] = {}
                    for key, value in row.items():
                        sample[key] = _normalize_cell_value(key, value)

                    if "__key__" not in sample:
                        sample["__key__"] = f"{os.path.basename(path)}:{rg_idx}:{row_idx}"
                    yield sample
