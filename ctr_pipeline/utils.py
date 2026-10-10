"""Small helpers used everywhere: math, timing, seeding, JSON and device selection."""

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def set_seed(seed):
    """Seed python, numpy and torch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


@contextmanager
def timer(store, key):
    """Record the wall-clock seconds of a `with` block into store[key]."""
    start = time.perf_counter()
    yield
    store[key] = round(time.perf_counter() - start, 3)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
