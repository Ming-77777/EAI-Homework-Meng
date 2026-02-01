from __future__ import annotations

import random
from typing import Dict, List, Tuple

N, E, S, W = "N", "E", "S", "W"
WAIT, PICK, DROP = "WAIT", "PICK", "DROP"
MOVE_ACTIONS = [N, E, S, W]

MOVE_DELTAS = {
    N: (-1, 0),
    E: (0, 1),
    S: (1, 0),
    W: (0, -1),
}


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbor_is_wall(local_grid: List[str], dr: int, dc: int) -> bool:
    size = len(local_grid)
    center = size // 2
    rr, cc = center + dr, center + dc
    if rr < 0 or cc < 0 or rr >= size or cc >= len(local_grid[rr]):
        return True
    return local_grid[rr][cc] == "#"


def valid_moves(obs: Dict[str, object]) -> List[str]:
    local_grid = obs["local_grid"]
    moves: List[str] = []
    for a in MOVE_ACTIONS:
        dr, dc = MOVE_DELTAS[a]
        if not _neighbor_is_wall(local_grid, dr, dc):
            moves.append(a)
    return moves


class WarehouseWeightedAgent:

    def __init__(self, seed: int = 0, w_close: float = 0.4, w_far: float = 0.1, w_neutral: float = 0.2):
        self.rng = random.Random(seed)
        self.w_close = w_close
        self.w_far = w_far
        self.w_neutral = w_neutral

    def act(self, obs: Dict[str, object]) -> str:
        robot_pos = obs["robot_pos"]
        has_item = bool(obs["has_item"])
        pickup_pos = obs["pickup_pos"]
        dropoff_pos = obs["dropoff_pos"]

        if robot_pos == pickup_pos and (not has_item):
            return PICK
        if robot_pos == dropoff_pos and has_item:
            return DROP

        goal = dropoff_pos if has_item else pickup_pos
        moves = valid_moves(obs)

        if not moves:
            return WAIT

        d0 = manhattan(robot_pos, goal)

        actions: List[str] = []
        weights: List[float] = []
        for a in moves:
            dr, dc = MOVE_DELTAS[a]
            nxt = (robot_pos[0] + dr, robot_pos[1] + dc)
            d1 = manhattan(nxt, goal)
            if d1 < d0:
                w = self.w_close
            elif d1 > d0:
                w = self.w_far
            else:
                w = self.w_neutral
            actions.append(a)
            weights.append(w)

        total = sum(weights)
        if total <= 0:
            return self.rng.choice(actions)
        r = self.rng.random() * total
        acc = 0.0
        for a, w in zip(actions, weights):
            acc += w
            if r <= acc:
                return a
        return actions[-1]
