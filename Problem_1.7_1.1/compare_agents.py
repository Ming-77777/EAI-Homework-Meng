from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from warehouse_env import WarehouseEnv
from warehouse_agent_reflex import WarehouseReflexAgent, MOVE_DELTAS, MOVE_ACTIONS, WAIT, PICK, DROP


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    final_battery: int
    total_reward: float


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

    valid: List[str] = [WAIT]

    for a in MOVE_ACTIONS:
        dr, dc = MOVE_DELTAS[a]
        if not neighbor_is_wall(local_grid, dr, dc):
            valid.append(a)

    if (robot_pos == pickup_pos) and (not has_item):
        valid.append(PICK)
    if (robot_pos == dropoff_pos) and has_item:
        valid.append(DROP)

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

        # 先处理 pick/drop
        if (robot_pos == pickup_pos) and (not has_item) and (PICK in va):
            return PICK
        if (robot_pos == dropoff_pos) and has_item and (DROP in va):
            return DROP

        goal = dropoff_pos if has_item else pickup_pos

        # 在所有合法移动里选使距离最小的
        candidates = []
        best = None
        for a in MOVE_ACTIONS:
            if a not in va:
                continue
            dr, dc = MOVE_DELTAS[a]
            nr, nc = robot_pos[0] + dr, robot_pos[1] + dc
            d = manhattan((nr, nc), goal)
            if best is None or d < best:
                best = d
                candidates = [a]
            elif d == best:
                candidates.append(a)

        if candidates:
            return self.rng.choice(candidates)

        return WAIT


def run_one_episode(env: WarehouseEnv, agent, randomize: bool = False) -> EpisodeResult:
    obs = env.reset(randomize=randomize)
    total_reward = 0.0

    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)

        if terminated or truncated:
            success = bool(terminated)
            return EpisodeResult(
                success=success,
                steps=int(obs["steps"]),
                final_battery=int(obs["battery"]),
                total_reward=total_reward,
            )


def run_n_episodes(env: WarehouseEnv, agent, n: int, seed0: int = 1234, randomize: bool = False) -> List[EpisodeResult]:
    results: List[EpisodeResult] = []
    for i in range(n):
        random.seed(seed0 + i)
        results.append(run_one_episode(env, agent, randomize=randomize))
    return results


def summarize(results: List[EpisodeResult]) -> Dict[str, object]:
    n = len(results)
    succ = [r for r in results if r.success]
    steps_succ = [r.steps for r in succ]

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    def median(xs):
        if not xs:
            return None
        ys = sorted(xs)
        m = len(ys) // 2
        return ys[m] if len(ys) % 2 else (ys[m - 1] + ys[m]) / 2

    return {
        "n": n,
        "success_rate": (len(succ) / n) if n else 0.0,
        "mean_steps_success": mean(steps_succ),
        "median_steps_success": median(steps_succ),
        "steps_all": [r.steps for r in results],
        "battery_all": [r.final_battery for r in results],
        "reward_all": [r.total_reward for r in results],
    }


def visualize(reflex_stats: Dict[str, object], greedy_stats: Dict[str, object]):
    labels = ["reflex", "greedy"]
    success_rates = [reflex_stats["success_rate"], greedy_stats["success_rate"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (1) success rate bar
    axes[0].bar(labels, success_rates)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Success rate")
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 0.02, f"{v*100:.1f}%", ha="center")

    # (2) box plot episode lengths
    axes[1].boxplot([reflex_stats["steps_all"], greedy_stats["steps_all"]], labels=labels)
    axes[1].set_title("Episode length (steps)")

    # (3) histogram final battery
    axes[2].hist(reflex_stats["battery_all"], alpha=0.5, label="reflex")
    axes[2].hist(greedy_stats["battery_all"], alpha=0.5, label="greedy")
    axes[2].set_title("Final battery")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def main():
    n = 50

    randomize = True

    env1 = WarehouseEnv()
    env2 = WarehouseEnv()

    reflex = WarehouseReflexAgent(seed=0)
    greedy = GreedyAgent(seed=0)

    reflex_results = run_n_episodes(env1, reflex, n=n, seed0=1000, randomize=randomize)
    greedy_results = run_n_episodes(env2, greedy, n=n, seed0=1000, randomize=randomize)

    rs = summarize(reflex_results)
    gs = summarize(greedy_results)

    print("=== Reflex Agent ===")
    print(f"success_rate: {rs['success_rate']*100:.1f}% ({int(rs['success_rate']*n)}/{n})")
    print(f"mean_steps_success: {rs['mean_steps_success']}")
    print(f"median_steps_success: {rs['median_steps_success']}")
    print(f"mean_reward: {sum(rs['reward_all'])/len(rs['reward_all']):.3f}")

    print("\n=== Greedy Agent ===")
    print(f"success_rate: {gs['success_rate']*100:.1f}% ({int(gs['success_rate']*n)}/{n})")
    print(f"mean_steps_success: {gs['mean_steps_success']}")
    print(f"median_steps_success: {gs['median_steps_success']}")
    print(f"mean_reward: {sum(gs['reward_all'])/len(gs['reward_all']):.3f}")

    visualize(rs, gs)


if __name__ == "__main__":
    main()
