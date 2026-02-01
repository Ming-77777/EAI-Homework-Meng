from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from warehouse_env import WarehouseEnv
from warehouse_agent_random import WarehouseRandomAgent
from warehouse_agent_weighted import WarehouseWeightedAgent

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

    valid: List[str] = []
    for a in MOVE_ACTIONS:
        dr, dc = MOVE_DELTAS[a]
        if not neighbor_is_wall(local_grid, dr, dc):
            valid.append(a)
    if robot_pos == pickup_pos and (not has_item):
        valid.append(PICK)
    if robot_pos == dropoff_pos and has_item:
        valid.append(DROP)
    if not valid:
        valid = [WAIT]
    return valid


class GreedyAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs: Dict[str, object]) -> str:
        robot_pos = obs["robot_pos"]
        has_item = bool(obs["has_item"])
        pickup_pos = obs["pickup_pos"]
        dropoff_pos = obs["dropoff_pos"]

        va = valid_actions(obs)

        if robot_pos == pickup_pos and (not has_item) and (PICK in va):
            return PICK
        if robot_pos == dropoff_pos and has_item and (DROP in va):
            return DROP

        goal = dropoff_pos if has_item else pickup_pos
        best = None
        cands: List[str] = []
        for a in MOVE_ACTIONS:
            if a not in va:
                continue
            dr, dc = MOVE_DELTAS[a]
            nxt = (robot_pos[0] + dr, robot_pos[1] + dc)
            d = manhattan(nxt, goal)
            if best is None or d < best:
                best = d
                cands = [a]
            elif d == best:
                cands.append(a)
        if cands:
            return self.rng.choice(cands)
        return WAIT


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    final_battery: int
    total_reward: float


def run_one_episode(env: WarehouseEnv, agent, randomize: bool, seed: int) -> EpisodeResult:
    random.seed(seed)
    obs = env.reset(randomize=randomize)

    total_reward = 0.0
    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            return EpisodeResult(
                success=bool(terminated),
                steps=int(obs["steps"]),
                final_battery=int(obs["battery"]),
                total_reward=total_reward,
            )


def run_n(env: WarehouseEnv, agent, n: int, seed_sequence: List[int], randomize: bool) -> List[EpisodeResult]:
    results: List[EpisodeResult] = []
    for i in range(n):
        results.append(run_one_episode(env, agent, randomize=randomize, seed=seed_sequence[i]))
    return results


def summarize(results: List[EpisodeResult]) -> Dict[str, float | List[int]]:
    n = len(results)
    succ = [r for r in results if r.success]
    steps = [r.steps for r in results]

    mean_steps = sum(steps) / n if n else 0.0
    var = sum((x - mean_steps) ** 2 for x in steps) / n if n else 0.0
    std_steps = var ** 0.5

    return {
        "n": n,
        "success_rate": (len(succ) / n) if n else 0.0,
        "mean_steps": mean_steps,
        "std_steps": std_steps,
        "steps": steps,
        "battery": [r.final_battery for r in results],
        "reward": [r.total_reward for r in results],
    }


def plot_spectrum(stats_by_name: Dict[str, Dict[str, float]]):
    names = list(stats_by_name.keys())
    means = [stats_by_name[k]["mean_steps"] for k in names]
    stds = [stats_by_name[k]["std_steps"] for k in names]
    rates = [stats_by_name[k]["success_rate"] for k in names]

    plt.figure(figsize=(7.5, 4.8))
    bars = plt.bar(names, means, yerr=stds, capsize=6)
    plt.ylabel("mean episode length (steps)")
    plt.title("Performance Spectrum: Random to Intelligent Agents")

    for bar, r in zip(bars, rates):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{r*100:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


def main():
    n = 100
    randomize = True
    seed_sequence = [1000 + i for i in range(n)]

    env_a = WarehouseEnv()
    env_b = WarehouseEnv()
    env_c = WarehouseEnv()

    agent_random = WarehouseRandomAgent(seed=0)
    agent_weighted = WarehouseWeightedAgent(seed=0)
    agent_greedy = GreedyAgent(seed=0)

    res_random = run_n(env_a, agent_random, n, seed_sequence, randomize)
    res_weighted = run_n(env_b, agent_weighted, n, seed_sequence, randomize)
    res_greedy = run_n(env_c, agent_greedy, n, seed_sequence, randomize)

    st_random = summarize(res_random)
    st_weighted = summarize(res_weighted)
    st_greedy = summarize(res_greedy)

    stats = {
        "random": st_random,
        "weighted": st_weighted,
        "greedy": st_greedy,
    }

    for name in ["random", "weighted", "greedy"]:
        s = stats[name]
        print(f"=== {name} ===")
        print(f"success_rate: {s['success_rate']*100:.1f}%")
        print(f"mean_steps:   {s['mean_steps']:.2f} ± {s['std_steps']:.2f}")
        print()

    plot_spectrum(stats)


if __name__ == "__main__":
    main()
