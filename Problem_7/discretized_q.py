from __future__ import annotations

import math
import numpy as np
import torch


class DiscretizedQLearner:
    def __init__(
        self,
        n_bins: int = 10,
        n_actions: int = 4,
        alpha: float = 0.15,
        gamma: float = 0.99,
        device: torch.device | None = None,
    ) -> None:
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = torch.zeros((n_bins, n_bins, n_actions), dtype=torch.float32, device=self.device)
        self.x_edges = np.linspace(0.0, 4.0, n_bins + 1)[1:-1]
        self.y_edges = np.linspace(0.0, 4.0, n_bins + 1)[1:-1]

    def encode(self, state: np.ndarray) -> tuple[int, int]:
        x = int(np.digitize(state[0], self.x_edges, right=False))
        y = int(np.digitize(state[1], self.y_edges, right=False))
        return min(max(x, 0), self.n_bins - 1), min(max(y, 0), self.n_bins - 1)

    def q_values(self, state: np.ndarray) -> torch.Tensor:
        x, y = self.encode(state)
        return self.q[x, y]

    def select_action(self, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, self.n_actions))
        values = self.q_values(state)
        best = torch.where(values == torch.max(values))[0]
        return int(best[rng.integers(0, len(best))].item())

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        x, y = self.encode(state)
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(torch.max(self.q_values(next_state)).item())
        self.q[x, y, action] += self.alpha * (target - float(self.q[x, y, action].item()))


class DenseDiscretizedQLearner6D:
    def __init__(self, n_bins: int = 10, n_actions: int = 4, alpha: float = 0.08, gamma: float = 0.99, device: torch.device | None = None) -> None:
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = torch.zeros((n_bins, n_bins, n_bins, n_bins, n_bins, n_bins, n_actions), dtype=torch.float32, device=self.device)
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

    def q_values(self, state: np.ndarray) -> torch.Tensor:
        return self.q[self.encode(state)]

    def select_action(self, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, self.n_actions))
        values = self.q_values(state)
        best = torch.where(values == torch.max(values))[0]
        return int(best[rng.integers(0, len(best))].item())

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        q_values = self.q_values(state)
        q_sa = float(q_values[action].item())
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(torch.max(self.q_values(next_state)).item())
        q_values[action] += self.alpha * (target - q_sa)

    def parameter_count(self) -> int:
        return int(self.q.numel())
