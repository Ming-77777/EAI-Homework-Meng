from __future__ import annotations
from collections import deque
import random
from typing import Dict, List, Tuple

N, E, S, W = "N", "E", "S", "W"
WAIT, PICK, DROP = "WAIT", "PICK", "DROP"
MOVE_ACTIONS = [N, E, S, W]

MOVE_DELTAS = {N: (-1, 0), E: (0, 1), S: (1, 0), W: (0, -1)}


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbor_is_wall(local_grid: List[str], dr: int, dc: int) -> bool:
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

    va: List[str] = [WAIT]

    for a in MOVE_ACTIONS:
        dr, dc = MOVE_DELTAS[a]
        if not neighbor_is_wall(local_grid, dr, dc):
            va.append(a)

    if robot_pos == pickup_pos and (not has_item):
        va.append(PICK)
    if robot_pos == dropoff_pos and has_item:
        va.append(DROP)

    return va


def simulate_move(pos: Tuple[int, int], act: str) -> Tuple[int, int]:
    if act not in MOVE_DELTAS:
        return pos
    dr, dc = MOVE_DELTAS[act]
    return (pos[0] + dr, pos[1] + dc)


class PasterAgent:

    def __init__(self, seed: int = 0, loop_k: int = 6):
        self.rng = random.Random(seed)
        self.recent_positions = deque(maxlen=loop_k)

        self.loop_penalty = 3.0
        self.wait_penalty = 0.5

    def reset(self):
        self.recent_positions.clear()

    def act(self, obs: Dict[str, object]) -> str:
        pos = obs["robot_pos"]
        has_item = bool(obs["has_item"])
        pickup = obs["pickup_pos"]
        dropoff = obs["dropoff_pos"]
        local_grid = obs["local_grid"]

        va = valid_actions(obs)

        if PICK in va:
            self.recent_positions.append(pos)
            return PICK
        if DROP in va:
            self.recent_positions.append(pos)
            return DROP

        goal = dropoff if has_item else pickup

        best_score = None
        best_actions: List[str] = []

        for a1 in va:
            if a1 in (PICK, DROP):
                continue

            if a1 in MOVE_DELTAS and neighbor_is_wall(local_grid, *MOVE_DELTAS[a1]):
                continue
            p1 = simulate_move(pos, a1)

            loop1 = self.loop_penalty if (p1 in self.recent_positions) else 0.0

            best2 = None
            for a2 in MOVE_ACTIONS + [WAIT]:
                p2 = simulate_move(p1, a2)
                d2 = manhattan(p2, goal)
                loop2 = self.loop_penalty if (p2 in self.recent_positions) else 0.0
                s2 = -d2 - loop2
                if best2 is None or s2 > best2:
                    best2 = s2

            d1 = manhattan(p1, goal)
            score = -d1 - loop1 + 0.5 * (best2 if best2 is not None else 0.0)

            if a1 == WAIT:
                score -= self.wait_penalty

            if best_score is None or score > best_score:
                best_score = score
                best_actions = [a1]
            elif score == best_score:
                best_actions.append(a1)

        chosen = self.rng.choice(best_actions) if best_actions else self.rng.choice(va)
        self.recent_positions.append(pos)
        return chosen
