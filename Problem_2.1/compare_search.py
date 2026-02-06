from __future__ import annotations

import csv
import statistics
from dataclasses import asdict
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt

from warehouse_env import WarehouseEnv
from ucs_pathfinder import ucs as ucs_search
from astar_pathfinder import astar as astar_search

Pos = Tuple[int, int]  


def run_two_stage(
    grid: List[str],
    start: Pos,
    pickup: Pos,
    dropoff: Pos,
) -> Dict[str, Dict[str, Any]]:
    """
    Run start->pickup and pickup->dropoff, then merge stats.
    """
    u1 = ucs_search(grid, start, pickup)
    u2 = ucs_search(grid, pickup, dropoff)

    a1 = astar_search(grid, start, pickup)
    a2 = astar_search(grid, pickup, dropoff)

    def merge(s1, s2) -> Dict[str, Any]:
        return {
            "path_length": s1.path_length + s2.path_length,
            "nodes_expanded": s1.nodes_expanded + s2.nodes_expanded,
            "max_frontier_size": max(s1.max_frontier_size, s2.max_frontier_size),
            "time_sec": s1.time_sec + s2.time_sec,
        }

    u = merge(u1, u2)
    a = merge(a1, a2)
    return {"ucs": u, "astar": a, "optimality_ok": {"ok": (u["path_length"] == a["path_length"])}}


def main():
    env = WarehouseEnv()

    trials: List[Dict[str, Any]] = []

    for t in range(10):
        obs = env.reset(randomize=True)
        grid = list(env.grid)  

        start: Pos = tuple(obs["robot_pos"])          
        pickup: Pos = tuple(obs["pickup_pos"])        
        dropoff: Pos = tuple(obs["dropoff_pos"])      

        res = run_two_stage(grid, start, pickup, dropoff)

        row = {
            "trial": t,
            "start_r": start[0], "start_c": start[1],
            "pickup_r": pickup[0], "pickup_c": pickup[1],
            "dropoff_r": dropoff[0], "dropoff_c": dropoff[1],
            "ucs_path_length": res["ucs"]["path_length"],
            "ucs_nodes_expanded": res["ucs"]["nodes_expanded"],
            "ucs_max_frontier_size": res["ucs"]["max_frontier_size"],
            "ucs_time_sec": res["ucs"]["time_sec"],
            "astar_path_length": res["astar"]["path_length"],
            "astar_nodes_expanded": res["astar"]["nodes_expanded"],
            "astar_max_frontier_size": res["astar"]["max_frontier_size"],
            "astar_time_sec": res["astar"]["time_sec"],
            "optimality_ok": res["optimality_ok"]["ok"],
        }
        trials.append(row)

    with open("compare_search_trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trials[0].keys()))
        w.writeheader()
        w.writerows(trials)

    def col(name: str) -> List[float]:
        return [float(r[name]) for r in trials]

    summary = []
    for algo in ["ucs", "astar"]:
        summary.append({
            "algo": algo,
            "mean_path_length": statistics.mean(col(f"{algo}_path_length")),
            "mean_nodes_expanded": statistics.mean(col(f"{algo}_nodes_expanded")),
            "mean_max_frontier_size": statistics.mean(col(f"{algo}_max_frontier_size")),
            "mean_time_sec": statistics.mean(col(f"{algo}_time_sec")),
            "std_path_length": statistics.pstdev(col(f"{algo}_path_length")),
            "std_nodes_expanded": statistics.pstdev(col(f"{algo}_nodes_expanded")),
            "std_time_sec": statistics.pstdev(col(f"{algo}_time_sec")),
        })

    optimal_ok_count = sum(1 for r in trials if r["optimality_ok"])
    summary.append({
        "algo": "optimality_check",
        "mean_path_length": "",
        "mean_nodes_expanded": "",
        "mean_max_frontier_size": "",
        "mean_time_sec": "",
        "std_path_length": "",
        "std_nodes_expanded": "",
        "std_time_sec": f"{optimal_ok_count}/10",
    })

    with open("compare_search_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    mean_ucs = statistics.mean(col("ucs_nodes_expanded"))
    mean_astar = statistics.mean(col("astar_nodes_expanded"))

    plt.figure()
    plt.bar(["UCS", "A*"], [mean_ucs, mean_astar])
    plt.ylabel("Mean nodes expanded (10 trials)")
    plt.title("UCS vs A*: Mean nodes expanded")
    plt.tight_layout()
    plt.savefig("compare_search_mean_nodes_expanded.png", dpi=200)

    print("Wrote:")
    print("  compare_search_trials.csv")
    print("  compare_search_summary.csv")
    print("  compare_search_mean_nodes_expanded.png")
    print(f"Optimality (same path length): {optimal_ok_count}/10")


if __name__ == "__main__":
    main()
