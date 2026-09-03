from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_mib: int
    free_mib: int
    used_mib: int
    utilization: int

    def to_dict(self) -> dict:
        return asdict(self)


def query_gpus() -> list[GPUInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    rows = csv.reader(io.StringIO(completed.stdout))
    result: list[GPUInfo] = []
    for row in rows:
        if len(row) != 6:
            continue
        result.append(
            GPUInfo(
                index=int(row[0].strip()),
                name=row[1].strip(),
                total_mib=int(row[2].strip()),
                free_mib=int(row[3].strip()),
                used_mib=int(row[4].strip()),
                utilization=int(row[5].strip()),
            )
        )
    if not result:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return result


def choose_gpu_candidate(
    inventory: Iterable[GPUInfo],
    candidates: Iterable[Iterable[int]],
    minimum_free_mib: Iterable[int],
    leased: dict[int, str],
) -> list[int] | None:
    by_index = {gpu.index: gpu for gpu in inventory}
    minimum = list(minimum_free_mib)
    for raw_candidate in candidates:
        candidate = [int(index) for index in raw_candidate]
        if len(candidate) != len(minimum):
            continue
        if any(index in leased for index in candidate):
            continue
        if any(index not in by_index for index in candidate):
            continue
        if all(by_index[index].free_mib >= required for index, required in zip(candidate, minimum)):
            return candidate
    return None
