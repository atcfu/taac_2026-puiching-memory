from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(slots=True)
class GpuDevice:
    index: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int


def query_gpu_devices(gpu_indices: set[int] | None = None) -> list[GpuDevice]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    devices: list[GpuDevice] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        device = GpuDevice(
            index=int(parts[0]),
            name=parts[1],
            total_memory_mb=int(parts[2]),
            used_memory_mb=int(parts[3]),
            free_memory_mb=int(parts[4]),
        )
        if gpu_indices is not None and device.index not in gpu_indices:
            continue
        devices.append(device)
    return devices


def parse_gpu_indices(raw_value: str | None) -> set[int] | None:
    if raw_value is None or not raw_value.strip():
        return None
    return {
        int(part.strip())
        for part in raw_value.split(",")
        if part.strip()
    }


def launchable_devices(
    devices: list[GpuDevice],
    running_jobs_by_gpu: dict[int, int],
    *,
    min_free_memory_mb: int,
    max_jobs_per_gpu: int,
) -> list[GpuDevice]:
    launchable: list[GpuDevice] = []
    for device in sorted(devices, key=lambda item: (item.free_memory_mb, -item.index), reverse=True):
        running_jobs = running_jobs_by_gpu.get(device.index, 0)
        if running_jobs >= max_jobs_per_gpu:
            continue
        if min_free_memory_mb <= 0:
            additional_slots = max_jobs_per_gpu - running_jobs
        else:
            additional_slots = min(device.free_memory_mb // min_free_memory_mb, max_jobs_per_gpu - running_jobs)
        if additional_slots <= 0:
            continue
        launchable.extend([device] * int(additional_slots))
    return launchable


__all__ = [
    "GpuDevice",
    "launchable_devices",
    "parse_gpu_indices",
    "query_gpu_devices",
]
