from __future__ import annotations

from pathlib import Path
import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from continuous_warehouse import ContinuousWarehouseEnv, BonusWarehouseEnv6D, ACTION_DELTAS
from discretized_q import DiscretizedQLearner, DenseDiscretizedQLearner6D
from tile_coded_q import TileCodedQLearner
from dqn_agent import DQNAgent
from utils_summary import (
    seed_everything,
    rolling_mean,
    optimal_reference,
    print_summary,
)


def rolling_episode_axis(series: np.ndarray, window: int) -> np.ndarray:
    if len(series) < window:
        return np.arange(len(series))
    return np.arange(window - 1, window - 1 + len(series) - window + 1)


def action_to_vector(action: int) -> tuple[float, float]:
    return float(ACTION_DELTAS[action][0]), float(ACTION_DELTAS[action][1])


def plot_policy_field(
    ax: plt.Axes,
    policy_fn,
    title: str,
    fixed_theta: float = 0.0,
    fixed_speed: float = 0.5,
    grid_size: int = 9,
) -> None:
    xs = np.linspace(0.25, 3.75, grid_size)
    ys = np.linspace(0.25, 3.75, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    u = np.zeros_like(xx)
    v = np.zeros_like(yy)
    for i in range(grid_size):
        for j in range(grid_size):
            state = np.array([xx[i, j], yy[i, j], fixed_theta, fixed_speed], dtype=np.float32)
            action = int(policy_fn(state))
            dx, dy = action_to_vector(action)
            u[i, j] = dx
            v[i, j] = dy

    ax.quiver(xx, yy, u, v, color="#2f4f4f", angles="xy", scale_units="xy", scale=1.8, width=0.006)
    goal = plt.Circle((3.5, 3.5), 0.5, color="#70c17a", alpha=0.25)
    hazard = plt.Circle((2.0, 2.0), 0.5, color="#dd6b6b", alpha=0.28)
    ax.add_patch(goal)
    ax.add_patch(hazard)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.grid(alpha=0.25)
    ax.set_title(title)


def save_policy_comparison(tabular_agent: DiscretizedQLearner, tile_agent: TileCodedQLearner, dqn_agent: DQNAgent, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True)

    def tabular_policy(state: np.ndarray) -> int:
        q = tabular_agent.q_values(state)
        if isinstance(q, torch.Tensor):
            return int(torch.argmax(q).item())
        return int(np.argmax(q))

    def tile_policy(state: np.ndarray) -> int:
        q = tile_agent.q_values(state)
        if isinstance(q, torch.Tensor):
            return int(torch.argmax(q).item())
        return int(np.argmax(q))

    def dqn_policy(state: np.ndarray) -> int:
        with np.errstate(all="ignore"):
            norm = dqn_agent.normalize_state(state)
            state_t = np.array(norm, dtype=np.float32)
        import torch

        with torch.no_grad():
            state_torch = torch.tensor(state_t, dtype=torch.float32, device=dqn_agent.device).unsqueeze(0)
            return int(torch.argmax(dqn_agent.q_net(state_torch), dim=1).item())

    plot_policy_field(axes[0], tabular_policy, "Discretized Q-learning")
    plot_policy_field(axes[1], tile_policy, "Tile-coded linear Q-learning")
    plot_policy_field(axes[2], dqn_policy, "DQN")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def train_discretized_q_learning(episodes: int, seed: int, **kwargs):
    seed_everything(seed)
    device = kwargs.get("device", None)
    env = ContinuousWarehouseEnv()
    agent = DiscretizedQLearner(device=device)
    rng = np.random.default_rng(seed)
    rewards = np.zeros(episodes, dtype=np.float32)
    epsilon = kwargs.get("epsilon_start", 1.0)
    epsilon_decay = kwargs.get("epsilon_decay", 0.995)
    epsilon_min = kwargs.get("epsilon_min", 0.01)

    for episode in range(episodes):
        state = env.reset(seed + episode)
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            action = agent.select_action(state, epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
        rewards[episode] = total_reward
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return agent, rewards


def train_tile_coding_q_learning(episodes: int, seed: int, **kwargs):
    seed_everything(seed)
    device = kwargs.get("device", None)
    env = ContinuousWarehouseEnv()
    agent = TileCodedQLearner(device=device)
    rng = np.random.default_rng(seed + 17)
    rewards = np.zeros(episodes, dtype=np.float32)
    epsilon = kwargs.get("epsilon_start", 1.0)
    epsilon_decay = kwargs.get("epsilon_decay", 0.995)
    epsilon_min = kwargs.get("epsilon_min", 0.01)

    for episode in range(episodes):
        state = env.reset(seed + 1000 + episode)
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            action = agent.select_action(state, epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
        rewards[episode] = total_reward
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return agent, rewards


def train_dqn(episodes: int, seed: int, use_replay=True, use_target=True, **kwargs):
    seed_everything(seed)
    env = ContinuousWarehouseEnv()
    agent = DQNAgent(use_replay=use_replay, use_target=use_target)
    rewards = np.zeros(episodes, dtype=np.float32)
    epsilon = kwargs.get("epsilon_start", 1.0)
    epsilon_decay = kwargs.get("epsilon_decay", 0.985)
    epsilon_min = kwargs.get("epsilon_min", 0.01)
    warmup = kwargs.get("warmup", 500)

    if use_replay:
        state = env.reset(seed)
        for _ in range(warmup):
            action = random.randrange(env.n_actions)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.push(env.normalize_state(state), action, reward, env.normalize_state(next_state), done)
            if done:
                state = env.reset()
            else:
                state = next_state

    for episode in range(episodes):
        state = env.reset(seed + 2000 + episode)
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < env.max_steps:
            norm_state = env.normalize_state(state)
            action = agent.select_action(norm_state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.update(norm_state, action, reward, env.normalize_state(next_state), done)
            state = next_state
            total_reward += reward
            steps += 1

        rewards[episode] = total_reward
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return agent, rewards


def plot_learning_comparison(tabular_rewards, tile_rewards, dqn_rewards, save_path: Path, window: int = 100):
    fig, ax = plt.subplots(figsize=(9, 5))
    tabular_curve = rolling_mean(tabular_rewards, window)
    tile_curve = rolling_mean(tile_rewards, window)
    dqn_curve = rolling_mean(dqn_rewards, window)
    ax.plot(rolling_episode_axis(tabular_rewards, window), tabular_curve, label="Discretized Q-learning", alpha=0.9, lw=2)
    ax.plot(rolling_episode_axis(tile_rewards, window), tile_curve, label="Tile-coded linear Q-learning", alpha=0.9, lw=2)
    ax.plot(rolling_episode_axis(dqn_rewards, window), dqn_curve, label="DQN", alpha=0.9, lw=2)
    if len(tabular_rewards) >= window:
        ax.set_xlim(window - 1, len(tabular_rewards) - 1)
    ax.set_title("Learning Curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling average reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_dqn_ablation(full_rewards, no_replay_rewards, no_target_rewards, save_path: Path, window: int = 100):
    fig, ax = plt.subplots(figsize=(9, 5))
    full_curve = rolling_mean(full_rewards, window)
    no_replay_curve = rolling_mean(no_replay_rewards, window)
    no_target_curve = rolling_mean(no_target_rewards, window)
    ax.plot(rolling_episode_axis(full_rewards, window), full_curve, label="Full DQN", alpha=0.9, lw=2)
    ax.plot(rolling_episode_axis(no_replay_rewards, window), no_replay_curve, label="No replay", alpha=0.9, lw=2)
    ax.plot(rolling_episode_axis(no_target_rewards, window), no_target_curve, label="No target net", alpha=0.9, lw=2)
    if len(full_rewards) >= window:
        ax.set_xlim(window - 1, len(full_rewards) - 1)
    ax.set_title("DQN Ablation Study")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling average reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_bonus_comparison(tabular_rewards, dqn_rewards, save_path: Path, window: int = 100):
    fig, ax = plt.subplots(figsize=(9, 5))
    tabular_curve = rolling_mean(tabular_rewards, window)
    dqn_curve = rolling_mean(dqn_rewards, window)
    ax.plot(rolling_episode_axis(tabular_rewards, window), tabular_curve, label="6D discretized Q-learning", alpha=0.9, lw=2)
    ax.plot(rolling_episode_axis(dqn_rewards, window), dqn_curve, label="6D DQN", alpha=0.9, lw=2)
    if len(tabular_rewards) >= window:
        ax.set_xlim(window - 1, len(tabular_rewards) - 1)
    ax.set_title("6D Bonus Comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling average reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def run_main_experiment(episodes: int, seed: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training discretized Q-learning...")
    tabular_agent, tabular_rewards = train_discretized_q_learning(episodes=episodes, seed=seed)
    print(f"  Device: {tabular_agent.device}")

    print("Training tile-coded linear Q-learning...")
    tile_agent, tile_rewards = train_tile_coding_q_learning(episodes=episodes, seed=seed)
    print(f"  Device: {tile_agent.device}")

    print("Training full DQN...")
    dqn_agent, dqn_rewards = train_dqn(episodes=episodes, seed=seed, use_replay=True, use_target=True)
    print(f"  Device: {dqn_agent.device}")

    optimal_4d = optimal_reference([tabular_rewards, tile_rewards, dqn_rewards])

    plot_learning_comparison(
        tabular_rewards,
        tile_rewards,
        dqn_rewards,
        output_dir / "learning_comparison.png",
    )
    save_policy_comparison(tabular_agent, tile_agent, dqn_agent, output_dir / "policy_comparison.png")

    print_summary("Discretized Q-learning", tabular_rewards, int(tabular_agent.q.numel()), optimal_4d)
    print_summary("Tile-coded Q-learning", tile_rewards, int(tile_agent.weights.numel()), optimal_4d)
    print_summary("DQN", dqn_rewards, int(sum(p.numel() for p in dqn_agent.q_net.parameters())), optimal_4d)

    print("Training DQN ablation variants...")
    _, dqn_no_replay = train_dqn(episodes=episodes, seed=seed + 11, use_replay=False, use_target=True)
    _, dqn_no_target = train_dqn(episodes=episodes, seed=seed + 23, use_replay=True, use_target=False)
    plot_dqn_ablation(
        dqn_rewards,
        dqn_no_replay,
        dqn_no_target,
        output_dir / "dqn_ablation.png",
    )

    print_summary("DQN no replay", dqn_no_replay, int(sum(p.numel() for p in dqn_agent.q_net.parameters())), optimal_4d)
    print_summary("DQN no target", dqn_no_target, int(sum(p.numel() for p in dqn_agent.q_net.parameters())), optimal_4d)


def run_bonus_experiment(episodes: int, seed: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training 6D discretized Q-learning baseline...")
    # implement light-weight 6D training here using the splitted modules
    def train_bonus_discretized_q_learning(episodes: int, seed: int):
        random_seed = seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        env = BonusWarehouseEnv6D()
        agent = DenseDiscretizedQLearner6D()
        rng = np.random.default_rng(seed + 101)
        rewards = np.zeros(episodes, dtype=np.float32)
        epsilon = 1.0
        for episode in range(episodes):
            state = env.reset(seed + 3000 + episode)
            total_reward = 0.0
            done = False
            steps = 0
            while not done and steps < env.max_steps:
                action = agent.select_action(state, epsilon, rng)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
            rewards[episode] = total_reward
            epsilon = max(0.01, epsilon * 0.995)
        return agent, rewards

    def train_bonus_dqn(episodes: int, seed: int):
        seed_everything(seed)
        env = BonusWarehouseEnv6D()
        agent = DQNAgent(state_dim=6)
        rewards = np.zeros(episodes, dtype=np.float32)
        epsilon = 1.0
        # warmup
        state = env.reset(seed)
        for _ in range(min(200, 500)):
            action = random.randrange(env.n_actions)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.push(agent.normalize_state(state), action, reward, agent.normalize_state(next_state), done)
            if done:
                state = env.reset()
            else:
                state = next_state

        for episode in range(episodes):
            state = env.reset(seed + 4000 + episode)
            total_reward = 0.0
            done = False
            steps = 0
            while not done and steps < env.max_steps:
                norm_state = agent.normalize_state(state)
                action = agent.select_action(norm_state, epsilon)
                next_state, reward, done, _ = env.step(action)
                agent.update(norm_state, action, reward, agent.normalize_state(next_state), done)
                state = next_state
                total_reward += reward
                steps += 1
            rewards[episode] = total_reward
            epsilon = max(0.01, epsilon * 0.985)
        return agent, rewards

    bonus_tabular_agent, bonus_tabular_rewards = train_bonus_discretized_q_learning(episodes=episodes, seed=seed)
    print("Training 6D DQN...")
    bonus_dqn_agent, bonus_dqn_rewards = train_bonus_dqn(episodes=episodes, seed=seed)

    optimal_6d = optimal_reference([bonus_tabular_rewards, bonus_dqn_rewards])

    plot_bonus_comparison(
        bonus_tabular_rewards,
        bonus_dqn_rewards,
        output_dir / "bonus_6d_comparison.png",
    )

    print("6D bonus uses a dense 10 bins-per-dimension table with 10^6 x 4 Q-values.")
    print("This makes the dimensionality cost explicit and keeps the comparison faithful to the assignment.")
    print_summary("6D discretized Q-learning", bonus_tabular_rewards, int(bonus_tabular_agent.parameter_count()), optimal_6d)
    if bonus_dqn_agent is not None:
        print_summary("6D DQN", bonus_dqn_rewards, int(sum(p.numel() for p in bonus_dqn_agent.q_net.parameters())), optimal_6d)


def parse_args():
    parser = argparse.ArgumentParser(description="Continuous warehouse RL experiments")
    parser.add_argument("--episodes", type=int, default=2000, help="Training episodes for each run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_main_experiment(args.episodes, args.seed, args.output_dir)
    run_bonus_experiment(args.episodes, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
