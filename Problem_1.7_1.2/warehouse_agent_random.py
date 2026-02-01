from __future__ import annotations

import random
from typing import Dict, List, Tuple

N, E, S, W = "N", "E", "S", "W"
WAIT, PICK, DROP = "WAIT", "PICK", "DROP"
MOVE_ACTIONS = [N, E, S, W]
BASE_ACTIONS = [N, E, S, W, PICK, DROP]

MOVE_DELTAS = {
    N: (-1, 0),
    E: (0, 1),
    S: (1, 0),
    W: (0, -1),
}


def _neighbor_is_wall(local_grid: List[str], dr: int, dc: int) -> bool:
    size = len(local_grid)
    center = size // 2
    rr, cc = center + dr, center + dc
    if rr < 0 or cc < 0 or rr >= size or cc >= len(local_grid[rr]):
        return True
    return local_grid[rr][cc] == "#"


def valid_actions(obs: Dict[str, object]) -> List[str]:
    local_grid = obs["local_grid"]
    has_item = bool(obs["has_item"])
    robot_pos = obs["robot_pos"]
    pickup_pos = obs["pickup_pos"]
    dropoff_pos = obs["dropoff_pos"]

    valid: List[str] = []

    for a in MOVE_ACTIONS:
        dr, dc = MOVE_DELTAS[a]
        if not _neighbor_is_wall(local_grid, dr, dc):
            valid.append(a)

    if robot_pos == pickup_pos and (not has_item):
        valid.append(PICK)
    if robot_pos == dropoff_pos and has_item:
        valid.append(DROP)
    if not valid:
        valid = [WAIT]
    return valid


class WarehouseRandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs: Dict[str, object]) -> str:
        va = valid_actions(obs)
        return self.rng.choice(va)
