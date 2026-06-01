"""Small shared utilities."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def memory_usage_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


@contextmanager
def timer(name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.getLogger(__name__).info("%s completed in %.2fs", name, elapsed)


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

