from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from rack_utils import Pos, objective, neighbors

@dataclass
class LSResult:
    best_state: List[Pos]
    best_value: float
    history: List[float]     
    iterations: int

def hill_climb_steepest(initial: List[Pos], max_iters: int = 3000) -> LSResult:
    cur = list(initial)
    cur_val = objective(cur)
    best = list(cur)
    best_val = cur_val
    history = [best_val]

    for _ in range(max_iters):
        nbs = neighbors(cur)
        if not nbs:
            break
        next_state = None
        next_val = best_val
        for nb in nbs:
            v = objective(nb)
            if v < next_val:
                next_val = v
                next_state = nb
        if next_state is None:
            break  # local optimum
        cur = next_state
        cur_val = next_val
        if cur_val < best_val:
            best, best_val = list(cur), cur_val
        history.append(best_val)

    return LSResult(best_state=best, best_value=best_val, history=history, iterations=len(history)-1)
