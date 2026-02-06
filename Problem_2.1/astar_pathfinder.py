from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Pos = Tuple[int, int]  


@dataclass
class SearchStats:
    path: List[Pos]
    path_length: int
    nodes_expanded: int
    max_frontier_size: int
    time_sec: float


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(parent: Dict[Pos, Optional[Pos]], goal: Pos) -> List[Pos]:
    cur = goal
    rev = [cur]
    while parent[cur] is not None:
        cur = parent[cur]  
        rev.append(cur)
    rev.reverse()
    return rev


def astar(grid: List[str], start: Pos, goal: Pos) -> SearchStats:
    t0 = time.perf_counter()

    H = len(grid)
    W = len(grid[0]) if H > 0 else 0

    def in_bounds(p: Pos) -> bool:
        r, c = p
        return 0 <= r < H and 0 <= c < W

    def passable(p: Pos) -> bool:
        r, c = p
        return grid[r][c] != "#"

    def neighbors(p: Pos) -> List[Pos]:
        r, c = p
        cand = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [q for q in cand if in_bounds(q) and passable(q)]

    pq: List[Tuple[int, int, int, Pos]] = []
    tie = 0
    g0 = 0
    f0 = g0 + manhattan(start, goal)
    heapq.heappush(pq, (f0, tie, g0, start))

    best_g: Dict[Pos, int] = {start: 0}
    parent: Dict[Pos, Optional[Pos]] = {start: None}

    nodes_expanded = 0
    max_frontier = 1

    while pq:
        max_frontier = max(max_frontier, len(pq))
        f, _, g, cur = heapq.heappop(pq)

        if g != best_g.get(cur, None):
            continue

        nodes_expanded += 1

        if cur == goal:
            path = reconstruct_path(parent, goal)
            t1 = time.perf_counter()
            return SearchStats(
                path=path,
                path_length=len(path) - 1,
                nodes_expanded=nodes_expanded,
                max_frontier_size=max_frontier,
                time_sec=t1 - t0,
            )

        for nb in neighbors(cur):
            ng = g + 1
            if ng < best_g.get(nb, 10**18):
                best_g[nb] = ng
                parent[nb] = cur
                tie += 1
                nf = ng + manhattan(nb, goal)
                heapq.heappush(pq, (nf, tie, ng, nb))

    t1 = time.perf_counter()
    return SearchStats(
        path=[],
        path_length=10**9,
        nodes_expanded=nodes_expanded,
        max_frontier_size=max_frontier,
        time_sec=t1 - t0,
    )
