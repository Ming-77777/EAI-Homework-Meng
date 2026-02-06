from __future__ import annotations
import random
from typing import List, Tuple, Set, Optional

Pos = Tuple[int, int]  

GRID_W = 20
GRID_H = 20
N_RACKS = 20
DEPOT: Pos = (10, 10)
LAMBDA = 2.0
CONGEST_RADIUS = 5  

def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def objective(state: List[Pos], lam: float = LAMBDA) -> float:
    avg_dist = sum(manhattan(DEPOT, p) for p in state) / float(N_RACKS)
    congest = sum(1 for p in state if manhattan(DEPOT, p) < CONGEST_RADIUS)
    return avg_dist + lam * congest

def random_state(rng: random.Random) -> List[Pos]:
    used: Set[Pos] = {DEPOT}
    out: List[Pos] = []
    while len(out) < N_RACKS:
        p = (rng.randrange(GRID_W), rng.randrange(GRID_H))
        if p in used:
            continue
        used.add(p)
        out.append(p)
    return out

def neighbors(state: List[Pos]) -> List[List[Pos]]:
    used = set(state)
    out: List[List[Pos]] = []
    for i, (x, y) in enumerate(state):
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                continue
            np = (nx, ny)
            if np == DEPOT:
                continue
            if np in used:
                continue
            new_state = list(state)
            new_state[i] = np
            out.append(new_state)
    return out

def random_neighbor(state: List[Pos], rng: random.Random) -> List[Pos]:
    used = set(state)
    idxs = list(range(len(state)))
    rng.shuffle(idxs)
    for i in idxs:
        x, y = state[i]
        moves = [(1,0), (-1,0), (0,1), (0,-1)]
        rng.shuffle(moves)
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                continue
            np = (nx, ny)
            if np == DEPOT or np in used:
                continue
            ns = list(state)
            ns[i] = np
            return ns
    return list(state)

def render_grid(state: List[Pos]) -> List[str]:
    grid = [["." for _ in range(GRID_W)] for _ in range(GRID_H)]
    dx, dy = DEPOT
    grid[dy][dx] = "*"
    for (x, y) in state:
        grid[y][x] = "R"
    return ["".join(row) for row in grid]
