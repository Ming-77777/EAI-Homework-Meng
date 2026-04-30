from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
        self.buffer: Deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
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
        batch_size: int = 64,
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

    def _optimize_batch(self, states, actions, rewards, next_states, dones) -> float:
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
