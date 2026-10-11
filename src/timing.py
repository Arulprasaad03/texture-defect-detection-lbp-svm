"""Timing utilities: Timer context manager, TimingLog, peak RSS."""
from __future__ import annotations

import time

import pandas as pd
import psutil


class TimingLog:
    def __init__(self):
        self.rows = []

    def add(self, stage: str, seconds: float, **tags) -> None:
        self.rows.append({"stage": stage, "seconds": seconds, **tags})

    def to_csv(self, path) -> None:
        pd.DataFrame(self.rows).to_csv(path, index=False, float_format="%.6f")


class Timer:
    """with Timer('stage', log, category='carpet'): ...  -> .seconds"""

    def __init__(self, stage: str, log: TimingLog | None = None, **tags):
        self.stage, self.log, self.tags = stage, log, tags
        self.seconds = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        self.seconds = (time.perf_counter_ns() - self._t0) / 1e9
        if self.log is not None:
            self.log.add(self.stage, self.seconds, **self.tags)
        return False


def peak_rss_mb() -> float:
    p = psutil.Process()
    info = p.memory_info()
    return getattr(info, "peak_wset", info.rss) / 1024 ** 2
