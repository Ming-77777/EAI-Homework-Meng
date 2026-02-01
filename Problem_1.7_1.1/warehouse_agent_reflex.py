from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

N, E, S, W = "N", "E", "S", "W"
WAIT, PICK, DROP = "WAIT", "PICK", "DROP"
MOVE_ACTIONS = [N, E, S, W]
ALL_ACTIONS = [N, E, S, W, WAIT, PICK, DROP]

MOVE_DELTAS = {
    N: (-1, 0),
    E: (0, 1),
    S: (1, 0),
    W: (0, -1),
}


def _neighbor_is_wall(local_grid: List[str], dr: int, dc: int) -> bool:

    if not local_grid:
        return False
    size = len(local_grid)
    center = size // 2
    rr = center + dr
    cc = center + dc
    if rr < 0 or cc < 0 or rr >= size or cc >= len(local_grid[rr]):
        return True
    return local_grid[rr][cc] == "#"


def _valid_actions_from_obs(obs: Dict[str, object]) -> List[str]:

    local_grid = obs["local_grid"]  # list[str]
    has_item = bool(obs["has_item"])


    robot_pos = obs["robot_pos"]
    pickup_pos = obs["pickup_pos"]
    dropoff_pos = obs["dropoff_pos"]

    valid: List[str] = [WAIT]


    for a in MOVE_ACTIONS:
        dr, dc = MOVE_DELTAS[a]
        if not _neighbor_is_wall(local_grid, dr, dc):
            valid.append(a)


    if (robot_pos == pickup_pos) and (not has_item):
        valid.append(PICK)
    if (robot_pos == dropoff_pos) and has_item:
        valid.append(DROP)

    return valid


def _direction_preference(pos: Tuple[int, int], goal: Tuple[int, int]) -> List[str]:
    r, c = pos
    gr, gc = goal
    prefs: List[str] = []

    dr = gr - r
    dc = gc - c

    if dr < 0:
        prefs.append(N)
    elif dr > 0:
        prefs.append(S)

    if dc > 0:
        prefs.append(E)
    elif dc < 0:
        prefs.append(W)

    return prefs


class WarehouseReflexAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs: Dict[str, object]) -> str:
        robot_pos = obs["robot_pos"]
        has_item = bool(obs["has_item"])
        pickup_pos = obs["pickup_pos"]
        dropoff_pos = obs["dropoff_pos"]

        valid_actions = _valid_actions_from_obs(obs)

        # Rule 1: PICK
        if (robot_pos == pickup_pos) and (not has_item) and (PICK in valid_actions):
            return PICK

        # Rule 2: DROP
        if (robot_pos == dropoff_pos) and has_item and (DROP in valid_actions):
            return DROP

        # Rule 3: move toward goal
        goal = dropoff_pos if has_item else pickup_pos
        prefs = _direction_preference(robot_pos, goal)
        for a in prefs:
            if a in valid_actions:
                return a

        # Rule 4: random valid action
        return self.rng.choice(valid_actions)
