from __future__ import annotations

import numpy as np
import torch


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
        device: torch.device | None = None,
    ) -> None:
        self.encoder = TileCoder(n_tilings=n_tilings, tiles_per_dim=tiles_per_dim)
        self.n_actions = n_actions
        self.alpha = alpha / n_tilings
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = torch.zeros((n_actions, self.encoder.n_features), dtype=torch.float32, device=self.device)

    def q_values(self, state: np.ndarray) -> torch.Tensor:
        phi = self.encoder.features(state)
        phi_t = torch.tensor(phi, dtype=torch.float32, device=self.device)
        return self.weights @ phi_t

    def select_action(self, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, self.n_actions))
        values = self.q_values(state)
        best = torch.where(values == torch.max(values))[0]
        return int(best[rng.integers(0, len(best))].item())

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        phi = self.encoder.features(state)
        phi_t = torch.tensor(phi, dtype=torch.float32, device=self.device)
        q_sa = float((self.weights[action] @ phi_t).item())
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(torch.max(self.q_values(next_state)).item())
        self.weights[action] += self.alpha * (target - q_sa) * phi_t
