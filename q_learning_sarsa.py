from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple
import numpy as np
import matplotlib.pyplot as plt

from warehouse_env_rl import State, WarehouseEnvRL


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 1000
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_floor: float = 0.01
    seed: int = 42


def state_to_idx(s: State) -> Tuple[int, int]:
    return s[0], s[1]


def select_action(q: np.ndarray, state: State, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, q.shape[-1]))
    x, y = state_to_idx(state)
    best = np.flatnonzero(q[x, y] == np.max(q[x, y]))
    return int(rng.choice(best))


def train_q_learning(env: WarehouseEnvRL, cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    q = np.zeros((env.GRID_W, env.GRID_H, env.n_actions), dtype=float)
    returns = np.zeros(cfg.episodes, dtype=float)
    epsilon = cfg.epsilon_start

    for ep in range(cfg.episodes):
        state = env.reset()
        done = False
        total = 0.0

        while not done:
            action = select_action(q, state, epsilon, rng)
            transition = env.step(action)
            next_state, reward, done = transition.next_state, transition.reward, transition.done
            total += reward

            x, y = state_to_idx(state)
            if done:
                td_target = reward
            else:
                nx, ny = state_to_idx(next_state)
                td_target = reward + cfg.gamma * np.max(q[nx, ny])

            q[x, y, action] += cfg.alpha * (td_target - q[x, y, action])
            state = next_state

        returns[ep] = total
        epsilon = max(cfg.epsilon_floor, epsilon * cfg.epsilon_decay)

    return q, returns


def train_sarsa(env: WarehouseEnvRL, cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    q = np.zeros((env.GRID_W, env.GRID_H, env.n_actions), dtype=float)
    returns = np.zeros(cfg.episodes, dtype=float)
    epsilon = cfg.epsilon_start

    for ep in range(cfg.episodes):
        state = env.reset()
        action = select_action(q, state, epsilon, rng)
        done = False
        total = 0.0

        while not done:
            transition = env.step(action)
            next_state, reward, done = transition.next_state, transition.reward, transition.done
            total += reward

            x, y = state_to_idx(state)
            if done:
                td_target = reward
            else:
                next_action = select_action(q, next_state, epsilon, rng)
                nx, ny = state_to_idx(next_state)
                td_target = reward + cfg.gamma * q[nx, ny, next_action]

            q[x, y, action] += cfg.alpha * (td_target - q[x, y, action])

            if not done:
                state, action = next_state, next_action

        returns[ep] = total
        epsilon = max(cfg.epsilon_floor, epsilon * cfg.epsilon_decay)

    return q, returns


def rolling_mean(values: np.ndarray, window: int = 50) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def value_iteration_reference(env: WarehouseEnvRL, gamma: float, tol: float = 1e-10) -> Tuple[np.ndarray, Dict[State, int]]:
    v = np.zeros((env.GRID_W, env.GRID_H), dtype=float)

    while True:
        delta = 0.0
        for x in range(env.GRID_W):
            for y in range(env.GRID_H):
                state = (x, y)
                if state in env.TERMINALS:
                    continue

                old = v[x, y]
                q_values = []
                for a in range(env.n_actions):
                    expected = 0.0
                    for prob, ns, reward, done in env.transition_distribution(state, a):
                        nx, ny = state_to_idx(ns)
                        bootstrap = 0.0 if done else gamma * v[nx, ny]
                        expected += prob * (reward + bootstrap)
                    q_values.append(expected)
                v[x, y] = max(q_values)
                delta = max(delta, abs(old - v[x, y]))
        if delta < tol:
            break

    policy: Dict[State, int] = {}
    for state in env.all_non_terminal_states():
        best_action = 0
        best_score = -np.inf
        for a in range(env.n_actions):
            score = 0.0
            for prob, ns, reward, done in env.transition_distribution(state, a):
                nx, ny = state_to_idx(ns)
                score += prob * (reward + (0.0 if done else gamma * v[nx, ny]))
            if score > best_score:
                best_score = score
                best_action = a
        policy[state] = best_action

    return v, policy


def greedy_policy_from_q(env: WarehouseEnvRL, q: np.ndarray) -> Dict[State, int]:
    policy: Dict[State, int] = {}
    for state in env.all_non_terminal_states():
        x, y = state_to_idx(state)
        policy[state] = int(np.argmax(q[x, y]))
    return policy


def count_matches(reference: Dict[State, int], learned: Dict[State, int], states: Iterable[State]) -> int:
    return sum(1 for s in states if reference.get(s, -1) == learned.get(s, -2))


def plot_learning_curves(
    q_returns: np.ndarray,
    sarsa_returns: np.ndarray,
    optimal_expected_return: float,
    save_path: Path,
    window: int = 50,
) -> None:
    q_smoothed = rolling_mean(q_returns, window=window)
    s_smoothed = rolling_mean(sarsa_returns, window=window)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(q_smoothed, label="Q-learning", lw=2)
    ax.plot(s_smoothed, label="SARSA", lw=2)
    ax.axhline(optimal_expected_return, color="black", ls="--", lw=1.2, label="Value-iteration target")
    ax.set_title("Learning Curves (rolling average reward)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_policies(env: WarehouseEnvRL, policies: Dict[str, Dict[State, int]], save_path: Path) -> None:
    action_vectors = {
        0: (0.0, 0.35),
        1: (0.35, 0.0),
        2: (0.0, -0.35),
        3: (-0.35, 0.0),
    }

    titles = list(policies.keys())
    fig, axes = plt.subplots(1, len(titles), figsize=(4.6 * len(titles), 4.5), sharex=True, sharey=True)
    if len(titles) == 1:
        axes = [axes]

    for ax, title in zip(axes, titles):
        policy = policies[title]
        ax.set_xlim(-0.5, env.GRID_W - 0.5)
        ax.set_ylim(-0.5, env.GRID_H - 0.5)
        ax.set_xticks(range(env.GRID_W))
        ax.set_yticks(range(env.GRID_H))
        ax.grid(alpha=0.35)
        ax.set_aspect("equal")
        ax.set_title(title)

        gx, gy = env.GOAL
        hx, hy = env.HAZARD
        ax.add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1, color="#8fd19e", alpha=0.7))
        ax.add_patch(plt.Rectangle((hx - 0.5, hy - 0.5), 1, 1, color="#f5a3a3", alpha=0.8))
        ax.text(gx, gy, "+1", ha="center", va="center", fontsize=12, weight="bold", color="#2c6e3e")
        ax.text(hx, hy, "-1", ha="center", va="center", fontsize=12, weight="bold", color="#aa2222")

        for s, a in policy.items():
            dx, dy = action_vectors[a]
            ax.annotate(
                "",
                xy=(s[0] + dx, s[1] + dy),
                xytext=(s[0], s[1]),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#2f2f2f"},
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def convergence_episode(curve: np.ndarray, target: float, tol: float = 0.08, sustain: int = 25) -> int | None:
    if len(curve) < sustain:
        return None
    for i in range(len(curve) - sustain + 1):
        segment = curve[i:i + sustain]
        if np.all(np.abs(segment - target) <= tol):
            return i
    return None


def run_hyperparameter_sensitivity(
    base_cfg: TrainConfig,
    output_dir: Path,
    optimal_expected_return: float,
) -> Tuple[dict, dict]:
    alphas = [0.01, 0.1, 0.5]
    decays = [0.99, 0.995, 0.999]

    fig, axes = plt.subplots(len(alphas), len(decays), figsize=(14, 10), sharex=True, sharey=True)
    summary_rows = []

    for i, alpha in enumerate(alphas):
        for j, decay in enumerate(decays):
            cfg = TrainConfig(
                episodes=base_cfg.episodes,
                alpha=alpha,
                gamma=base_cfg.gamma,
                epsilon_start=base_cfg.epsilon_start,
                epsilon_decay=decay,
                epsilon_floor=base_cfg.epsilon_floor,
                seed=base_cfg.seed,
            )
            env_q = WarehouseEnvRL(seed=cfg.seed)
            env_s = WarehouseEnvRL(seed=cfg.seed)
            _, q_rewards = train_q_learning(env_q, cfg)
            _, s_rewards = train_sarsa(env_s, cfg)

            q_curve = rolling_mean(q_rewards, window=50)
            s_curve = rolling_mean(s_rewards, window=50)

            ax = axes[i, j]
            ax.plot(q_curve, label="Q", color="#1f77b4", alpha=0.9)
            ax.plot(s_curve, label="S", color="#ff7f0e", alpha=0.9)
            ax.set_title(f"alpha={alpha}, rho={decay}", fontsize=10)
            ax.grid(alpha=0.3)
            if i == len(alphas) - 1:
                ax.set_xlabel("Episode")
            if j == 0:
                ax.set_ylabel("Return")

            summary_rows.append(
                {
                    "alpha": alpha,
                    "rho": decay,
                    "q_last50_mean": float(np.mean(q_rewards[-50:])),
                    "sarsa_last50_mean": float(np.mean(s_rewards[-50:])),
                    "q_conv": convergence_episode(q_curve, target=optimal_expected_return, tol=0.1, sustain=30),
                    "s_conv": convergence_episode(s_curve, target=optimal_expected_return, tol=0.1, sustain=30),
                }
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Hyperparameter Sensitivity (Q-learning and SARSA)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "hyperparameter_sensitivity.png", dpi=180)
    plt.close(fig)

    header = "alpha,rho,q_last50_mean,sarsa_last50_mean,q_conv,s_conv"
    lines = [header]
    for row in summary_rows:
        lines.append(
            f"{row['alpha']},{row['rho']},{row['q_last50_mean']:.6f},"
            f"{row['sarsa_last50_mean']:.6f},{row['q_conv']},{row['s_conv']}"
        )
    (output_dir / "hyperparameter_summary.csv").write_text("\n".join(lines), encoding="utf-8")

    best_q = max(summary_rows, key=lambda r: r["q_last50_mean"])
    best_s = max(summary_rows, key=lambda r: r["sarsa_last50_mean"])
    return best_q, best_s


def print_policy_differences(
    env: WarehouseEnvRL,
    optimal: Dict[State, int],
    q_policy: Dict[State, int],
    sarsa_policy: Dict[State, int],
) -> None:
    names = env.ACTIONS
    print("\nState-wise differences (optimal / Q-learning / SARSA):")
    for s in env.all_non_terminal_states():
        o = optimal[s]
        q = q_policy[s]
        sa = sarsa_policy[s]
        if q != o or sa != o:
            print(f"  {s}: {names[o]} / {names[q]} / {names[sa]}")


def main() -> None:
    output_dir = Path("Assignment 4") / "rl_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig()
    env_ref = WarehouseEnvRL(seed=cfg.seed)
    values, optimal_policy = value_iteration_reference(env_ref, gamma=cfg.gamma)
    optimal_expected_return = values[state_to_idx(env_ref.START)]

    q_env = WarehouseEnvRL(seed=cfg.seed)
    s_env = WarehouseEnvRL(seed=cfg.seed)
    q_table, q_rewards = train_q_learning(q_env, cfg)
    s_table, sarsa_rewards = train_sarsa(s_env, cfg)

    q_policy = greedy_policy_from_q(q_env, q_table)
    sarsa_policy = greedy_policy_from_q(s_env, s_table)

    states = env_ref.all_non_terminal_states()
    q_matches = count_matches(optimal_policy, q_policy, states)
    s_matches = count_matches(optimal_policy, sarsa_policy, states)

    q_curve = rolling_mean(q_rewards, window=50)
    s_curve = rolling_mean(sarsa_rewards, window=50)
    q_conv = convergence_episode(q_curve, optimal_expected_return)
    s_conv = convergence_episode(s_curve, optimal_expected_return)

    print(f"Value iteration expected return at start: {optimal_expected_return:.4f}")
    print(f"Q-learning match count: {q_matches}/{len(states)}")
    print(f"SARSA match count: {s_matches}/{len(states)}")
    print(f"Q-learning convergence episode (rolling): {q_conv}")
    print(f"SARSA convergence episode (rolling): {s_conv}")
    print_policy_differences(env_ref, optimal_policy, q_policy, sarsa_policy)

    plot_learning_curves(
        q_returns=q_rewards,
        sarsa_returns=sarsa_rewards,
        optimal_expected_return=float(optimal_expected_return),
        save_path=output_dir / "learning_curves.png",
        window=50,
    )

    plot_policies(
        env=env_ref,
        policies={
            "Value Iteration": optimal_policy,
            "Q-learning": q_policy,
            "SARSA": sarsa_policy,
        },
        save_path=output_dir / "policy_comparison.png",
    )

    best_q, best_s = run_hyperparameter_sensitivity(
        cfg,
        output_dir,
        float(optimal_expected_return),
    )

    (output_dir / "run_summary.txt").write_text(
        "\n".join(
            [
                f"optimal_expected_return={float(optimal_expected_return):.6f}",
                f"q_match={q_matches}/{len(states)}",
                f"sarsa_match={s_matches}/{len(states)}",
                f"q_convergence_episode={q_conv}",
                f"sarsa_convergence_episode={s_conv}",
                (
                    "best_q_hparam="
                    f"(alpha={best_q['alpha']},rho={best_q['rho']},last50={best_q['q_last50_mean']:.4f})"
                ),
                (
                    "best_sarsa_hparam="
                    f"(alpha={best_s['alpha']},rho={best_s['rho']},last50={best_s['sarsa_last50_mean']:.4f})"
                ),
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
