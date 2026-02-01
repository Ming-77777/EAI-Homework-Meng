from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from warehouse_env import WarehouseEnv
from warehouse_agent_paster import PasterAgent

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


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def reset(self):
        pass

    def act(self, obs: Dict[str, object]) -> str:
        va = valid_actions(obs)
        non_wait = [a for a in va if a != WAIT]
        choices = non_wait if non_wait else va
        return self.rng.choice(choices)


class GreedyAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def reset(self):
        pass

    def act(self, obs: Dict[str, object]) -> str:
        pos = obs["robot_pos"]
        has_item = bool(obs["has_item"])
        pickup = obs["pickup_pos"]
        dropoff = obs["dropoff_pos"]

        va = valid_actions(obs)

        if PICK in va:
            return PICK
        if DROP in va:
            return DROP

        goal = dropoff if has_item else pickup

        best = None
        cands: List[str] = []
        for a in MOVE_ACTIONS:
            if a not in va:
                continue
            dr, dc = MOVE_DELTAS[a]
            nxt = (pos[0] + dr, pos[1] + dc)
            d = manhattan(nxt, goal)
            if best is None or d < best:
                best = d
                cands = [a]
            elif d == best:
                cands.append(a)

        return self.rng.choice(cands) if cands else WAIT


LAYOUTS: Dict[str, List[str]] = {
    "layout_0_default": [
        "############",
        "#..P....#..#",
        "#..##...#..#",
        "#......##..#",
        "#..#.......#",
        "#..#..D....#",
        "############",
    ],
    "layout_1_more_walls": [
        "############",
        "#..P..#.#..#",
        "#.###.#.#..#",
        "#....##.#..#",
        "#.##....#..#",
        "#..#..D....#",
        "############",
    ],
    "layout_2_corridors": [
        "############",
        "#..P....#..#",
        "#.####..#..#",
        "#......##..#",
        "###..#.....#",
        "#..#..D....#",
        "############",
    ],
    "layout_3_open_middle": [
        "############",
        "#..P.......#",
        "#..##..##..#",
        "#..........#",
        "#..##..##..#",
        "#.......D..#",
        "############",
    ],
    "layout_4_mazeish": [
        "############",
        "#..P..#....#",
        "#.##..#.##.#",
        "#....##....#",
        "#.##..#.##.#",
        "#....#..D..#",
        "############",
    ],
}


@dataclass
class Row:
    agent: str
    layout: str
    episode: int
    success: int
    steps: int
    battery: int
    reward: float


def run_one_episode(env: WarehouseEnv, agent, seed: int, randomize_pd: bool = False) -> Row:
    random.seed(seed)

    obs = env.reset(randomize=randomize_pd)
    agent.reset() if hasattr(agent, "reset") else None

    total_reward = 0.0
    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            return Row(
                agent=type(agent).__name__,
                layout="",
                episode=seed,
                success=1 if terminated else 0,
                steps=int(obs["steps"]),
                battery=int(obs["battery"]),
                reward=total_reward,
            )


def evaluate_all(episodes_per_layout: int = 200, randomize_pd: bool = False) -> List[Row]:
    agents = {
        "random": RandomAgent(seed=0),
        "greedy": GreedyAgent(seed=0),
        "paster": PasterAgent(seed=0),
    }

    rows: List[Row] = []

    for layout_name, grid in LAYOUTS.items():
        for agent_name, agent in agents.items():
            env = WarehouseEnv(grid=grid, start_pos=(1, 1), max_steps=200, battery=200, view_radius=2)

            for ep in range(episodes_per_layout):
                seed = 1000 + ep 
                r = run_one_episode(env, agent, seed=seed, randomize_pd=randomize_pd)
                r.agent = agent_name
                r.layout = layout_name
                r.episode = ep
                rows.append(r)

    return rows


def aggregate(rows: List[Row]) -> Dict[str, Dict[str, float]]:
    by_agent: Dict[str, List[Row]] = {}
    for r in rows:
        by_agent.setdefault(r.agent, []).append(r)

    out: Dict[str, Dict[str, float]] = {}
    for agent, rs in by_agent.items():
        n = len(rs)
        succ = sum(x.success for x in rs)
        out[agent] = {
            "success_rate": succ / n if n else 0.0,
            "mean_steps": sum(x.steps for x in rs) / n if n else 0.0,
            "mean_battery": sum(x.battery for x in rs) / n if n else 0.0,
            "mean_reward": sum(x.reward for x in rs) / n if n else 0.0,
        }
    return out


def plot_dashboard(stats: Dict[str, Dict[str, float]], out_png: str = "dashboard_1_3.png"):
    agents = ["random", "greedy", "paster"]

    success = [stats[a]["success_rate"] for a in agents]
    steps = [stats[a]["mean_steps"] for a in agents]
    battery = [stats[a]["mean_battery"] for a in agents]
    reward = [stats[a]["mean_reward"] for a in agents]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # 1) success rate
    ax = axes[0, 0]
    bars = ax.bar(agents, success)
    ax.set_ylim(0, 1)
    ax.set_title("Success rate")
    for b, v in zip(bars, success):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v*100:.1f}%", ha="center")

    # 2) mean steps
    ax = axes[0, 1]
    ax.bar(agents, steps)
    ax.set_title("Mean episode length (steps)")
    ax.set_ylabel("steps")

    # 3) mean battery remaining
    ax = axes[1, 0]
    ax.bar(agents, battery)
    ax.set_title("Mean final battery")
    ax.set_ylabel("battery")

    # 4) mean reward
    ax = axes[1, 1]
    ax.bar(agents, reward)
    ax.set_title("Mean total reward")
    ax.set_ylabel("reward")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved dashboard to: {out_png}")


def main():
    rows = evaluate_all(episodes_per_layout=200, randomize_pd=False)

    stats = aggregate(rows)

    print("=== Aggregated across all layouts ===")
    for a in ["random", "greedy", "paster"]:
        s = stats[a]
        print(f"{a:6s} | success={s['success_rate']*100:5.1f}% | steps={s['mean_steps']:7.2f} | battery={s['mean_battery']:7.2f} | reward={s['mean_reward']:8.2f}")

    plot_dashboard(stats, out_png="dashboard_1_3.png")


if __name__ == "__main__":
    main()
