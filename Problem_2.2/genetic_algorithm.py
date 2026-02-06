from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Set

from rack_utils import Pos, objective, random_state, random_neighbor, GRID_W, GRID_H, DEPOT, N_RACKS

@dataclass
class GAResult:
    best_state: List[Pos]
    best_value: float
    history: List[float]     # best-so-far per generation
    generations: int

def tournament_select(pop: List[List[Pos]], rng: random.Random, k: int = 3) -> List[Pos]:
    cand = rng.sample(pop, k)
    cand.sort(key=objective)
    return cand[0]

def crossover(p1: List[Pos], p2: List[Pos], rng: random.Random) -> List[Pos]:
    n = N_RACKS
    take = rng.randint(6, n - 6)
    idx = set(rng.sample(range(n), take))
    child: List[Pos] = [p1[i] for i in idx]
    used: Set[Pos] = set(child)
    used.add(DEPOT)

    for q in p2:
        if q not in used:
            child.append(q)
            used.add(q)
        if len(child) == n:
            break

    while len(child) < n:
        p = (rng.randrange(GRID_W), rng.randrange(GRID_H))
        if p in used:
            continue
        child.append(p)
        used.add(p)

    return child

def mutate(ind: List[Pos], rng: random.Random, pmove: float = 0.7) -> List[Pos]:
    if rng.random() < pmove:
        return random_neighbor(ind, rng)
    a, b = rng.sample(range(len(ind)), 2)
    out = list(ind)
    out[a], out[b] = out[b], out[a]
    return out

def genetic_algorithm(
    initial: List[Pos],
    seed: int = 0,
    pop_size: int = 60,
    generations: int = 250,
    elite: int = 4,
    mutation_rate: float = 0.25,
) -> GAResult:
    rng = random.Random(seed)

    pop: List[List[Pos]] = [list(initial)]
    while len(pop) < pop_size:
        pop.append(random_state(rng))

    best = min(pop, key=objective)
    best_val = objective(best)
    history = [best_val]

    for _ in range(generations):
        pop.sort(key=objective)
        new_pop = [list(pop[i]) for i in range(elite)]  # elitism

        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, rng)
            p2 = tournament_select(pop, rng)
            child = crossover(p1, p2, rng)
            if rng.random() < mutation_rate:
                child = mutate(child, rng)
            new_pop.append(child)

        pop = new_pop
        cur_best = min(pop, key=objective)
        cur_val = objective(cur_best)
        if cur_val < best_val:
            best, best_val = list(cur_best), cur_val

        history.append(best_val)

    return GAResult(best_state=best, best_value=best_val, history=history, generations=generations)
