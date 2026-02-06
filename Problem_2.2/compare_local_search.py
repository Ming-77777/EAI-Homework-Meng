from __future__ import annotations
import csv
import time
import random
import statistics
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

from rack_utils import random_state, objective, render_grid, GRID_W, GRID_H
from hill_climbing import hill_climb_steepest
from simulated_annealing import simulated_annealing
from genetic_algorithm import genetic_algorithm

def pad_history(h: List[float], L: int) -> List[float]:
    if len(h) >= L:
        return h[:L]
    return h + [h[-1]] * (L - len(h))

def plot_best_layout(state, title: str, out_png: str):
    grid = render_grid(state)  # list[str], y-major
    arr = np.zeros((GRID_H, GRID_W), dtype=int)
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            arr[y, x] = {"." : 0, "R": 1, "*": 2}.get(ch, 0)

    plt.figure(figsize=(6, 6))
    plt.imshow(arr)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

def main():
    rng = random.Random(42)
    trials = 20

    rows: List[Dict[str, Any]] = []
    curves: Dict[str, List[List[float]]] = {"hill": [], "sa": [], "ga": []}
    best_run: Dict[str, Any] = {
        "hill": (None, float("inf")),
        "sa": (None, float("inf")),
        "ga": (None, float("inf")),
    }

    for t in range(trials):
        init = random_state(rng)
        init_val = objective(init)

        t0 = time.perf_counter()
        r_h = hill_climb_steepest(init, max_iters=3000)
        dt = time.perf_counter() - t0
        curves["hill"].append(r_h.history)
        if r_h.best_value < best_run["hill"][1]:
            best_run["hill"] = (r_h.best_state, r_h.best_value)
        rows.append({
            "trial": t, "algo": "hill",
            "init_value": init_val,
            "best_value": r_h.best_value,
            "iterations": r_h.iterations,
            "time_sec": dt
        })

        t0 = time.perf_counter()
        r_sa = simulated_annealing(init, seed=1000 + t, T0=5.0, alpha=0.995, Tmin=1e-3, steps=8000)
        dt = time.perf_counter() - t0
        curves["sa"].append(r_sa.history)
        if r_sa.best_value < best_run["sa"][1]:
            best_run["sa"] = (r_sa.best_state, r_sa.best_value)
        rows.append({
            "trial": t, "algo": "sa",
            "init_value": init_val,
            "best_value": r_sa.best_value,
            "iterations": r_sa.iterations,
            "time_sec": dt
        })

        t0 = time.perf_counter()
        r_ga = genetic_algorithm(init, seed=2000 + t, pop_size=60, generations=250, elite=4, mutation_rate=0.25)
        dt = time.perf_counter() - t0
        curves["ga"].append(r_ga.history)
        if r_ga.best_value < best_run["ga"][1]:
            best_run["ga"] = (r_ga.best_state, r_ga.best_value)
        rows.append({
            "trial": t, "algo": "ga",
            "init_value": init_val,
            "best_value": r_ga.best_value,
            "iterations": r_ga.generations,
            "time_sec": dt
        })

    with open("problem2_2_trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def summarize(algo: str) -> Dict[str, Any]:
        r = [x for x in rows if x["algo"] == algo]
        return {
            "algo": algo,
            "mean_best": statistics.mean(x["best_value"] for x in r),
            "std_best": statistics.pstdev(x["best_value"] for x in r),
            "mean_time_sec": statistics.mean(x["time_sec"] for x in r),
            "mean_iters": statistics.mean(x["iterations"] for x in r),
        }

    summary = [summarize("hill"), summarize("sa"), summarize("ga")]
    with open("problem2_2_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    maxlen = max(max(len(h) for h in curves[k]) for k in curves)
    plt.figure()
    for k, label in [("hill", "Hill Climbing"), ("sa", "Simulated Annealing"), ("ga", "Genetic Algorithm")]:
        hs = [pad_history(h, maxlen) for h in curves[k]]
        mean_curve = [sum(h[i] for h in hs) / len(hs) for i in range(maxlen)]
        plt.plot(mean_curve, label=label)
    plt.xlabel("Iteration / Generation")
    plt.ylabel("Objective f(s) (lower is better)")
    plt.title("Problem 2.2 Convergence (mean over 20 initial states)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("problem2_2_convergence.png", dpi=200)

    for k, out in [("hill", "problem2_2_best_hill.png"),
                   ("sa", "problem2_2_best_sa.png"),
                   ("ga", "problem2_2_best_ga.png")]:
        state, val = best_run[k]
        plot_best_layout(state, f"{k} best f(s) = {val:.3f}", out)

    print("Wrote:")
    print("  problem2_2_trials.csv")
    print("  problem2_2_summary.csv")
    print("  problem2_2_convergence.png")
    print("  problem2_2_best_hill.png / problem2_2_best_sa.png / problem2_2_best_ga.png")

if __name__ == "__main__":
    main()
