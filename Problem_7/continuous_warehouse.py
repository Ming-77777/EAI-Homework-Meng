from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass

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
