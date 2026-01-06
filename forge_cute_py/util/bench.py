import math
from typing import Iterable, Dict, List, Sequence

import torch


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def do_bench(fn, warmup: int = 10, rep: int = 100) -> List[float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking")
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms: List[float] = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def summarize_times(times_ms: Iterable[float]) -> Dict[str, float]:
    times = list(times_ms)
    if not times:
        return {"mean_ms": float("nan"), "p50_ms": float("nan"), "p90_ms": float("nan")}
    times_sorted = sorted(times)
    mean = sum(times_sorted) / len(times_sorted)
    return {
        "mean_ms": mean,
        "p50_ms": _percentile(times_sorted, 0.5),
        "p90_ms": _percentile(times_sorted, 0.9),
    }


def bytes_per_ms_to_gbps(bytes_per_ms: float) -> float:
    return bytes_per_ms / 1e6


def estimate_bandwidth(bytes_moved: int, time_ms: float) -> float:
    if time_ms <= 0:
        return float("inf")
    return bytes_per_ms_to_gbps(bytes_moved / time_ms)
