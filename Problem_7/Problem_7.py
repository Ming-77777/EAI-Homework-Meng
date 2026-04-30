from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import math
import random
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ACTION_NAMES = ("North", "South", "East", "West")
ACTION_DELTAS = np.array(
    [
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [-1.0, 0.0],
    ],
    dtype=np.float32,
)


@dataclass
class ContinuousWarehouseEnv:
    width: float = 4.0
    height: float = 4.0
    goal_center: tuple[float, float] = (3.5, 3.5)
    hazard_center: tuple[float, float] = (2.0, 2.0)
    goal_radius: float = 0.5
    hazard_radius: float = 0.5
    step_size: float = 0.22
    xy_noise_std: float = 0.03
    theta_noise_std: float = 0.12
    speed_noise_std: float = 0.03
    max_steps: int = 200

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self.state = np.zeros(4, dtype=np.float32)

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def n_actions(self) -> int:
        return 4

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        x = self._rng.uniform(0.2, 0.8)
        y = self._rng.uniform(0.2, 0.8)
        theta = self._rng.uniform(0.0, 2.0 * math.pi)
        speed = self._rng.uniform(0.0, 0.2)
        self.state = np.array([x, y, theta, speed], dtype=np.float32)
        self._steps = 0
        return self.state.copy()

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        normalized = state.astype(np.float32).copy()
        normalized[0] /= self.width
        normalized[1] /= self.height
        normalized[2] /= 2.0 * math.pi
        normalized[3] = np.clip(normalized[3], 0.0, 1.0)
        return np.clip(normalized, 0.0, 1.0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float | bool]]:
        action = int(action)
        prev_xy = self.state[:2].copy()
        prev_goal_dist = np.linalg.norm(prev_xy - np.array(self.goal_center, dtype=np.float32))
        prev_hazard_dist = np.linalg.norm(prev_xy - np.array(self.hazard_center, dtype=np.float32))

        direction = ACTION_DELTAS[action]
        target_theta = math.atan2(float(direction[1]), float(direction[0])) % (2.0 * math.pi)

        theta = 0.68 * float(self.state[2]) + 0.32 * target_theta
        theta += float(self._rng.normal(0.0, self.theta_noise_std))
        theta = theta % (2.0 * math.pi)

        speed = 0.72 * float(self.state[3]) + 0.38
        speed += float(self._rng.normal(0.0, self.speed_noise_std))
        speed = float(np.clip(speed, 0.0, 1.0))

        move = self.step_size * (0.45 + 0.55 * speed) * direction
        xy_noise = self._rng.normal(0.0, self.xy_noise_std, size=2).astype(np.float32)
        xy = np.clip(prev_xy + move + xy_noise, 0.0, self.width)

        self.state = np.array([xy[0], xy[1], theta, speed], dtype=np.float32)
        self._steps += 1

        goal_dist = np.linalg.norm(xy - np.array(self.goal_center, dtype=np.float32))
        hazard_dist = np.linalg.norm(xy - np.array(self.hazard_center, dtype=np.float32))
        goal_reached = goal_dist <= self.goal_radius
        hazard_hit = hazard_dist <= self.hazard_radius

        reward = -0.01
        reward += 0.05 * (prev_goal_dist - goal_dist)
        reward -= 0.03 * (prev_hazard_dist - hazard_dist)

        done = False
        if goal_reached:
            reward = 1.0
            done = True
        elif hazard_hit:
            reward = -1.0
            done = True
        elif self._steps >= self.max_steps:
            done = True

        return self.state.copy(), float(reward), done, {"goal": goal_reached, "hazard": hazard_hit}


class DiscretizedQLearner:
    def __init__(
        self,
        n_bins: int = 10,
        n_actions: int = 4,
        alpha: float = 0.15,
        gamma: float = 0.99,
    ) -> None:
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q = np.zeros((n_bins, n_bins, n_actions), dtype=np.float32)
        self.x_edges = np.linspace(0.0, 4.0, n_bins + 1)[1:-1]
        self.y_edges = np.linspace(0.0, 4.0, n_bins + 1)[1:-1]

    def encode(self, state: np.ndarray) -> tuple[int, int]:
        x = int(np.digitize(state[0], self.x_edges, right=False))
        y = int(np.digitize(state[1], self.y_edges, right=False))
        return min(max(x, 0), self.n_bins - 1), min(max(y, 0), self.n_bins - 1)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        x, y = self.encode(state)
        return self.q[x, y]

    def select_action(self, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, self.n_actions))
        values = self.q_values(state)
        best = np.flatnonzero(values == np.max(values))
        return int(rng.choice(best))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        x, y = self.encode(state)
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.q_values(next_state)))
        self.q[x, y, action] += self.alpha * (target - self.q[x, y, action])


class TileCoder:
    def __init__(self, n_tilings: int = 8, tiles_per_dim: int = 4) -> None:
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.extra = 1
        self.tiles_per_tiling = (tiles_per_dim + self.extra) ** 2
        self.n_features = self.n_tilings * self.tiles_per_tiling
        self.tile_width = 4.0 / tiles_per_dim
        self.offsets = np.array(
            [(k / self.n_tilings) * self.tile_width for k in range(self.n_tilings)],
            dtype=np.float32,
        )

    def features(self, state: np.ndarray) -> np.ndarray:
        phi = np.zeros(self.n_features, dtype=np.float32)
        x, y = float(state[0]), float(state[1])
        for tiling in range(self.n_tilings):
            shift = float(self.offsets[tiling])
            x_bin = int(np.floor((x + shift) / self.tile_width))
            y_bin = int(np.floor((y + shift) / self.tile_width))
            x_bin = min(max(x_bin, 0), self.tiles_per_dim)
            y_bin = min(max(y_bin, 0), self.tiles_per_dim)
            feature_index = tiling * self.tiles_per_tiling + x_bin * (self.tiles_per_dim + self.extra) + y_bin
            phi[feature_index] = 1.0
        return phi


class TileCodedQLearner:
    def __init__(
        self,
        n_tilings: int = 8,
        tiles_per_dim: int = 4,
        n_actions: int = 4,
        alpha: float = 0.08,
        gamma: float = 0.99,
    ) -> None:
        self.encoder = TileCoder(n_tilings=n_tilings, tiles_per_dim=tiles_per_dim)
        self.n_actions = n_actions
        self.alpha = alpha / n_tilings
        self.gamma = gamma
        self.weights = np.zeros((n_actions, self.encoder.n_features), dtype=np.float32)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        phi = self.encoder.features(state)
        return self.weights @ phi

    def select_action(self, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, self.n_actions))
        values = self.q_values(state)
        best = np.flatnonzero(values == np.max(values))
        return int(rng.choice(best))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        phi = self.encoder.features(state)
        q_sa = float(self.weights[action] @ phi)
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.q_values(next_state)))
        self.weights[action] += self.alpha * (target - q_sa) * phi


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 4, n_actions: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000) -> None:
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int = 4,
        n_actions: int = 4,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 32,
        target_update: int = 100,
        buffer_size: int = 10_000,
        use_replay: bool = True,
        use_target: bool = True,
        device: torch.device | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_replay = use_replay
        self.use_target = use_target
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    @staticmethod
    def normalize_state(state: np.ndarray) -> np.ndarray:
        normalized = state.astype(np.float32).copy()
        normalized[0] /= 4.0
        normalized[1] /= 4.0
        normalized[2] /= 2.0 * math.pi
        normalized[3] = np.clip(normalized[3], 0.0, 1.0)
        return np.clip(normalized, 0.0, 1.0)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def _optimize_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_source = self.target_net if self.use_target else self.q_net
            next_q = next_source(next_states_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        if self.use_replay:
            self.buffer.push(state, action, reward, next_state, done)
            if len(self.buffer) < self.batch_size:
                return 0.0
            batch = self.buffer.sample(self.batch_size)
            loss = self._optimize_batch(*batch)
        else:
            loss = self._optimize_batch(
                np.asarray([state], dtype=np.float32),
                np.asarray([action], dtype=np.int64),
                np.asarray([reward], dtype=np.float32),
                np.asarray([next_state], dtype=np.float32),
                np.asarray([done], dtype=np.float32),
            )

        self.steps += 1
        if self.use_target and self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss


@dataclass
class BonusWarehouseEnv6D:
    width: float = 4.0
    height: float = 4.0
    goal_center: tuple[float, float] = (3.5, 3.5)
    hazard_center: tuple[float, float] = (2.0, 2.0)
    goal_radius: float = 0.5
    hazard_radius: float = 0.5
    step_size: float = 0.22
    xy_noise_std: float = 0.03
    theta_noise_std: float = 0.12
    speed_noise_std: float = 0.03
    max_steps: int = 220

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self.state = np.zeros(6, dtype=np.float32)

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def n_actions(self) -> int:
        return 4

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        x = self._rng.uniform(0.2, 0.8)
        y = self._rng.uniform(0.2, 0.8)
        theta = self._rng.uniform(0.0, 2.0 * math.pi)
        speed = self._rng.uniform(0.0, 0.2)
        load = self._rng.uniform(0.0, 1.0)
        battery = self._rng.uniform(0.7, 1.0)
        self.state = np.array([x, y, theta, speed, load, battery], dtype=np.float32)
        self._steps = 0
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float | bool]]:
        action = int(action)
        prev_xy = self.state[:2].copy()
        load = float(self.state[4])
        battery = float(self.state[5])
        prev_goal_dist = np.linalg.norm(prev_xy - np.array(self.goal_center, dtype=np.float32))
        prev_hazard_dist = np.linalg.norm(prev_xy - np.array(self.hazard_center, dtype=np.float32))

        direction = ACTION_DELTAS[action]
        target_theta = math.atan2(float(direction[1]), float(direction[0])) % (2.0 * math.pi)

        theta = 0.68 * float(self.state[2]) + 0.32 * target_theta
        theta += float(self._rng.normal(0.0, self.theta_noise_std))
        theta = theta % (2.0 * math.pi)

        speed = 0.72 * float(self.state[3]) + 0.38
        speed *= 1.0 - 0.45 * load
        speed += float(self._rng.normal(0.0, self.speed_noise_std))
        speed = float(np.clip(speed, 0.0, 1.0))

        move = self.step_size * (0.45 + 0.55 * speed) * direction
        xy_noise = self._rng.normal(0.0, self.xy_noise_std, size=2).astype(np.float32)
        xy = np.clip(prev_xy + move + xy_noise, 0.0, self.width)

        battery = float(np.clip(battery - (0.015 + 0.04 * load), 0.0, 1.0))
        self.state = np.array([xy[0], xy[1], theta, speed, load, battery], dtype=np.float32)
        self._steps += 1

        goal_dist = np.linalg.norm(xy - np.array(self.goal_center, dtype=np.float32))
        hazard_dist = np.linalg.norm(xy - np.array(self.hazard_center, dtype=np.float32))
        goal_reached = goal_dist <= self.goal_radius
        hazard_hit = hazard_dist <= self.hazard_radius

        reward = -0.01 - 0.01 * load
        reward += 0.05 * (prev_goal_dist - goal_dist)
        reward -= 0.03 * (prev_hazard_dist - hazard_dist)

        done = False
        if goal_reached:
            reward = 1.0 + 0.5 * load
            done = True
        elif hazard_hit:
            reward = -1.0 - 0.3 * load
            done = True
        elif battery <= 0.0:
            reward = -0.5
            done = True
        elif self._steps >= self.max_steps:
            done = True

        return self.state.copy(), float(reward), done, {"goal": goal_reached, "hazard": hazard_hit, "battery": battery}


class DenseDiscretizedQLearner6D:
    def __init__(self, n_bins: int = 10, n_actions: int = 4, alpha: float = 0.08, gamma: float = 0.99) -> None:
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q = np.zeros((n_bins, n_bins, n_bins, n_bins, n_bins, n_bins, n_actions), dtype=np.float32)
        self.x_edges = np.linspace(0.0, 4.0, n_bins + 1)[1:-1]
        self.y_edges = np.linspace(0.0, 4.0, n_bins + 1)[1:-1]
        self.unit_edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]

    def encode(self, state: np.ndarray) -> tuple[int, ...]:
        x = int(np.digitize(state[0], self.x_edges, right=False))
        y = int(np.digitize(state[1], self.y_edges, right=False))
        theta = int(np.digitize((state[2] % (2.0 * math.pi)) / (2.0 * math.pi), self.unit_edges, right=False))
        speed = int(np.digitize(np.clip(state[3], 0.0, 1.0), self.unit_edges, right=False))
        load = int(np.digitize(np.clip(state[4], 0.0, 1.0), self.unit_edges, right=False))
        battery = int(np.digitize(np.clip(state[5], 0.0, 1.0), self.unit_edges, right=False))
        return (
            min(max(x, 0), self.n_bins - 1),
            min(max(y, 0), self.n_bins - 1),
            min(max(theta, 0), self.n_bins - 1),
            min(max(speed, 0), self.n_bins - 1),
            min(max(load, 0), self.n_bins - 1),
            min(max(battery, 0), self.n_bins - 1),
        )

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.q[self.encode(state)]

    def select_action(self, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, self.n_actions))
        values = self.q_values(state)
        best = np.flatnonzero(values == np.max(values))
        return int(rng.choice(best))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        q_values = self.q_values(state)
        q_sa = float(q_values[action])
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.q_values(next_state)))
        q_values[action] += self.alpha * (target - q_sa)

    def parameter_count(self) -> int:
        return int(self.q.size)


def train_bonus_discretized_q_learning(
    episodes: int,
    seed: int,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
) -> tuple[DenseDiscretizedQLearner6D, np.ndarray]:
    seed_everything(seed)
    env = BonusWarehouseEnv6D()
    agent = DenseDiscretizedQLearner6D()
    rng = np.random.default_rng(seed + 101)
    rewards = np.zeros(episodes, dtype=np.float32)
    epsilon = epsilon_start

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
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return agent, rewards


def train_bonus_dqn(
    episodes: int,
    seed: int,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.985,
    epsilon_min: float = 0.01,
    warmup: int = 500,
) -> tuple[DQNAgent, np.ndarray]:
    seed_everything(seed)
    env = BonusWarehouseEnv6D()
    agent = DQNAgent(state_dim=6)
    rewards = np.zeros(episodes, dtype=np.float32)
    epsilon = epsilon_start

    state = env.reset(seed)
    for _ in range(warmup):
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
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return agent, rewards


def plot_bonus_comparison(
    tabular_rewards: np.ndarray,
    dqn_rewards: np.ndarray,
    save_path: Path,
    window: int = 100,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rolling_mean(tabular_rewards, window), label="6D discretized Q-learning", alpha=0.9, lw=2)
    ax.plot(rolling_mean(dqn_rewards, window), label="6D DQN", alpha=0.9, lw=2)
    ax.set_title("6D Bonus Comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling average reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def best_rolling_average(rewards: np.ndarray, window: int = 100) -> float:
    curve = rolling_mean(rewards, window)
    if len(curve) == 0:
        return float(np.mean(rewards))
    return float(np.max(curve))


def optimal_reference(reward_sets: list[np.ndarray], window: int = 100) -> float:
    return max(best_rolling_average(rewards, window=window) for rewards in reward_sets)


def print_summary(name: str, rewards: np.ndarray, param_count: int, optimal: float) -> None:
    recent = float(np.mean(rewards[-200:]))
    threshold = 0.8 * optimal
    curve = rolling_mean(rewards, 100)
    episode = first_window_at_least(curve, threshold, window=100)
    print(f"{name}: final-200 average reward = {recent:.4f}")
    print(f"{name}: global optimal reference = {optimal:.4f}")
    print(f"{name}: 80% of optimal = {threshold:.4f}")
    print(f"{name}: first 100-episode window above threshold = {episode}")
    print(f"{name}: parameter count = {param_count}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rolling_mean(values: Iterable[float], window: int = 100) -> np.ndarray:
    values_arr = np.asarray(list(values), dtype=np.float32)
    if len(values_arr) < window:
        return values_arr
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values_arr, kernel, mode="valid")


def first_window_at_least(curve: np.ndarray, threshold: float, window: int = 100) -> int | None:
    if len(curve) < window:
        return None
    for idx in range(len(curve) - window + 1):
        if np.mean(curve[idx : idx + window]) >= threshold:
            return idx
    return None


def train_discretized_q_learning(
    episodes: int,
    seed: int,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
) -> tuple[DiscretizedQLearner, np.ndarray]:
    seed_everything(seed)
    env = ContinuousWarehouseEnv()
    agent = DiscretizedQLearner()
    rng = np.random.default_rng(seed)
    rewards = np.zeros(episodes, dtype=np.float32)
    epsilon = epsilon_start

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


def train_tile_coding_q_learning(
    episodes: int,
    seed: int,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
) -> tuple[TileCodedQLearner, np.ndarray]:
    seed_everything(seed)
    env = ContinuousWarehouseEnv()
    agent = TileCodedQLearner()
    rng = np.random.default_rng(seed + 17)
    rewards = np.zeros(episodes, dtype=np.float32)
    epsilon = epsilon_start

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


def train_dqn(
    episodes: int,
    seed: int,
    use_replay: bool = True,
    use_target: bool = True,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.985,
    epsilon_min: float = 0.01,
    warmup: int = 500,
) -> tuple[DQNAgent, np.ndarray]:
    seed_everything(seed)
    env = ContinuousWarehouseEnv()
    agent = DQNAgent(use_replay=use_replay, use_target=use_target)
    rewards = np.zeros(episodes, dtype=np.float32)
    epsilon = epsilon_start

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


def action_to_vector(action: int) -> tuple[float, float]:
    return float(ACTION_DELTAS[action][0]), float(ACTION_DELTAS[action][1])


def plot_learning_comparison(
    tabular_rewards: np.ndarray,
    tile_rewards: np.ndarray,
    dqn_rewards: np.ndarray,
    save_path: Path,
    window: int = 100,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rolling_mean(tabular_rewards, window), label="Discretized Q-learning", alpha=0.9, lw=2)
    ax.plot(rolling_mean(tile_rewards, window), label="Tile-coded linear Q-learning", alpha=0.9, lw=2)
    ax.plot(rolling_mean(dqn_rewards, window), label="DQN", alpha=0.9, lw=2)
    ax.set_title("Learning Curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling average reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_dqn_ablation(
    full_rewards: np.ndarray,
    no_replay_rewards: np.ndarray,
    no_target_rewards: np.ndarray,
    save_path: Path,
    window: int = 100,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rolling_mean(full_rewards, window), label="Full DQN", alpha=0.9, lw=2)
    ax.plot(rolling_mean(no_replay_rewards, window), label="No replay", alpha=0.9, lw=2)
    ax.plot(rolling_mean(no_target_rewards, window), label="No target net", alpha=0.9, lw=2)
    ax.set_title("DQN Ablation Study")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling average reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


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


def save_policy_comparison(
    tabular_agent: DiscretizedQLearner,
    tile_agent: TileCodedQLearner,
    dqn_agent: DQNAgent,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True)

    def tabular_policy(state: np.ndarray) -> int:
        return int(np.argmax(tabular_agent.q_values(state)))

    def tile_policy(state: np.ndarray) -> int:
        return int(np.argmax(tile_agent.q_values(state)))

    def dqn_policy(state: np.ndarray) -> int:
        with torch.no_grad():
            norm = dqn_agent.normalize_state(state)
            state_t = torch.tensor(norm, dtype=torch.float32, device=dqn_agent.device).unsqueeze(0)
            return int(torch.argmax(dqn_agent.q_net(state_t), dim=1).item())

    plot_policy_field(axes[0], tabular_policy, "Discretized Q-learning")
    plot_policy_field(axes[1], tile_policy, "Tile-coded linear Q-learning")
    plot_policy_field(axes[2], dqn_policy, "DQN")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def parameter_count_tabular(agent: DiscretizedQLearner) -> int:
    return int(agent.q.size)


def parameter_count_tile(agent: TileCodedQLearner) -> int:
    return int(agent.weights.size)


def parameter_count_dqn(agent: DQNAgent) -> int:
    return int(sum(param.numel() for param in agent.q_net.parameters()))


def best_rolling_average(rewards: np.ndarray, window: int = 100) -> float:
    curve = rolling_mean(rewards, window)
    if len(curve) == 0:
        return float(np.mean(rewards))
    return float(np.max(curve))


def optimal_reference(reward_sets: list[np.ndarray], window: int = 100) -> float:
    return max(best_rolling_average(rewards, window=window) for rewards in reward_sets)


def print_summary(name: str, rewards: np.ndarray, param_count: int, optimal: float) -> None:
    recent = float(np.mean(rewards[-200:]))
    threshold = 0.8 * optimal
    curve = rolling_mean(rewards, 100)
    episode = first_window_at_least(curve, threshold, window=100)
    print(f"{name}: final-200 average reward = {recent:.4f}")
    print(f"{name}: global optimal reference = {optimal:.4f}")
    print(f"{name}: 80% of optimal = {threshold:.4f}")
    print(f"{name}: first 100-episode window above threshold = {episode}")
    print(f"{name}: parameter count = {param_count}")


def run_main_experiment(episodes: int, seed: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training discretized Q-learning...")
    tabular_agent, tabular_rewards = train_discretized_q_learning(episodes=episodes, seed=seed)

    print("Training tile-coded linear Q-learning...")
    tile_agent, tile_rewards = train_tile_coding_q_learning(episodes=episodes, seed=seed)

    print("Training full DQN...")
    dqn_agent, dqn_rewards = train_dqn(episodes=episodes, seed=seed, use_replay=True, use_target=True)

    optimal_4d = optimal_reference([tabular_rewards, tile_rewards, dqn_rewards])

    plot_learning_comparison(
        tabular_rewards,
        tile_rewards,
        dqn_rewards,
        output_dir / "learning_comparison.png",
    )
    save_policy_comparison(
        tabular_agent,
        tile_agent,
        dqn_agent,
        output_dir / "policy_comparison.png",
    )

    print_summary("Discretized Q-learning", tabular_rewards, parameter_count_tabular(tabular_agent), optimal_4d)
    print_summary("Tile-coded Q-learning", tile_rewards, parameter_count_tile(tile_agent), optimal_4d)
    print_summary("DQN", dqn_rewards, parameter_count_dqn(dqn_agent), optimal_4d)

    print("Training DQN ablation variants...")
    _, dqn_no_replay = train_dqn(episodes=episodes, seed=seed + 11, use_replay=False, use_target=True)
    _, dqn_no_target = train_dqn(episodes=episodes, seed=seed + 23, use_replay=True, use_target=False)
    plot_dqn_ablation(
        dqn_rewards,
        dqn_no_replay,
        dqn_no_target,
        output_dir / "dqn_ablation.png",
    )

    print_summary("DQN no replay", dqn_no_replay, parameter_count_dqn(dqn_agent), optimal_4d)
    print_summary("DQN no target", dqn_no_target, parameter_count_dqn(dqn_agent), optimal_4d)


def run_bonus_experiment(episodes: int, seed: int, output_dir: Path) -> None:
    print("Training 6D discretized Q-learning baseline...")
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
    print("This makes the dimensionality cost explicit and keeps the comparison faithful to the assignment." )
    print_summary("6D discretized Q-learning", bonus_tabular_rewards, bonus_tabular_agent.parameter_count(), optimal_6d)
    print_summary("6D DQN", bonus_dqn_rewards, parameter_count_dqn(bonus_dqn_agent), optimal_6d)


def parse_args() -> argparse.Namespace:
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


def main() -> None:
    args = parse_args()
    run_main_experiment(args.episodes, args.seed, args.output_dir)
    run_bonus_experiment(args.episodes, args.seed, args.output_dir)


if __name__ == "__main__":
    main()