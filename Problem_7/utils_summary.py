from __future__ import annotations

import random
import math
from typing import Iterable

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rolling_mean(values: Iterable[float], window: int = 100) -> np.ndarray:
    values_arr = np.asarray(list(values), dtype=np.float32)
    if len(values_arr) < window:
        return values_arr
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values_arr, kernel, mode="valid")


def best_rolling_average(rewards: np.ndarray, window: int = 100) -> float:
    curve = rolling_mean(rewards, window)
    if len(curve) == 0:
        return float(np.mean(rewards))
    return float(np.max(curve))


def optimal_reference(reward_sets: list[np.ndarray], window: int = 100) -> float:
    return max(best_rolling_average(rewards, window=window) for rewards in reward_sets)


def first_window_at_least(curve: np.ndarray, threshold: float) -> int | None:
    # curve is expected to be a rolling-mean sequence; return first index where
    # the rolling-mean meets threshold.
    for idx, val in enumerate(curve):
        if val >= threshold:
            return idx
    return None


def print_summary(name: str, rewards: np.ndarray, param_count: int, optimal: float) -> None:
    recent = float(np.mean(rewards[-200:]))
    threshold = 0.8 * optimal
    curve = rolling_mean(rewards, 100)
    episode = first_window_at_least(curve, threshold)
    print(f"{name}: final-200 average reward = {recent:.4f}")
    print(f"{name}: global optimal reference = {optimal:.4f}")
    print(f"{name}: 80% of optimal = {threshold:.4f}")
    print(f"{name}: first 100-episode window above threshold = {episode}")
    print(f"{name}: parameter count = {param_count}")
