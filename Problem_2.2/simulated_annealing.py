from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List

from rack_utils import Pos, objective, random_neighbor

@dataclass
class LSResult:
    best_state: List[Pos]
    best_value: float
    history: List[float]     
    iterations: int

def simulated_annealing(
    initial: List[Pos],
    seed: int = 0,
    T0: float = 5.0,
    alpha: float = 0.995,
    Tmin: float = 1e-3,
    steps: int = 8000,
) -> LSResult:
    rng = random.Random(seed)
    cur = list(initial)
    cur_val = objective(cur)
    best = list(cur)
    best_val = cur_val
    history = [best_val]

    T = T0
    for _ in range(steps):
        nb = random_neighbor(cur, rng)
        nb_val = objective(nb)
        delta = nb_val - cur_val  

        if delta <= 0:
            cur, cur_val = nb, nb_val
        else:
            if T > 0 and rng.random() < math.exp(-delta / T):
                cur, cur_val = nb, nb_val

        if cur_val < best_val:
            best, best_val = list(cur), cur_val

        history.append(best_val)
        T *= alpha
        if T < Tmin:
            break

    return LSResult(best_state=best, best_value=best_val, history=history, iterations=len(history)-1)
