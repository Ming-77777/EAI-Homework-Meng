from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


State = Tuple[int, int]


@dataclass(frozen=True)
class StepResult:
    next_state: State
    reward: float
    done: bool


class WarehouseEnvRL:
    """4x4 warehouse gridworld used for tabular RL experiments.

    Coordinates follow (x, y) with x in [0, 3], y in [0, 3], origin at bottom-left.
    """

    GRID_W = 4
    GRID_H = 4

    START: State = (0, 0)
    GOAL: State = (3, 3)
    HAZARD: State = (3, 2)
    TERMINALS = {GOAL, HAZARD}

    ACTIONS: List[str] = ["N", "E", "S", "W"]
    ACTION_DELTAS: List[State] = [
        (0, 1),   # N
        (1, 0),   # E
        (0, -1),  # S
        (-1, 0),  # W
    ]

    LIVING_REWARD = -0.04
    GOAL_REWARD = 1.0
    HAZARD_REWARD = -1.0

    P_INTENDED = 0.8
    P_SIDE = 0.1

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self.state: State = self.START

    @property
    def n_actions(self) -> int:
        return len(self.ACTIONS)

    def reset(self) -> State:
        self.state = self.START
        return self.state

    def all_non_terminal_states(self) -> List[State]:
        states: List[State] = []
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if (x, y) not in self.TERMINALS:
                    states.append((x, y))
        return states

    def step(self, action: int) -> StepResult:
        """Sample one transition from the environment dynamics.

        The reward follows r_{t+1} = R(s_{t+1}).
        """
        if self.state in self.TERMINALS:
            return StepResult(self.state, 0.0, True)

        move_action = self._sample_stochastic_action(action)
        dx, dy = self.ACTION_DELTAS[move_action]
        nx = int(np.clip(self.state[0] + dx, 0, self.GRID_W - 1))
        ny = int(np.clip(self.state[1] + dy, 0, self.GRID_H - 1))
        next_state: State = (nx, ny)

        if next_state == self.GOAL:
            reward, done = self.GOAL_REWARD, True
        elif next_state == self.HAZARD:
            reward, done = self.HAZARD_REWARD, True
        else:
            reward, done = self.LIVING_REWARD, False

        self.state = next_state
        return StepResult(next_state, reward, done)

    def transition_distribution(self, state: State, action: int) -> List[Tuple[float, State, float, bool]]:
        """Model view for evaluation only (value iteration and diagnostics)."""
        if state in self.TERMINALS:
            return [(1.0, state, 0.0, True)]

        outcomes: Dict[State, float] = {}
        for prob, move_action in self._action_outcomes(action):
            dx, dy = self.ACTION_DELTAS[move_action]
            nx = int(np.clip(state[0] + dx, 0, self.GRID_W - 1))
            ny = int(np.clip(state[1] + dy, 0, self.GRID_H - 1))
            ns: State = (nx, ny)
            outcomes[ns] = outcomes.get(ns, 0.0) + prob

        transitions: List[Tuple[float, State, float, bool]] = []
        for ns, prob in outcomes.items():
            if ns == self.GOAL:
                reward, done = self.GOAL_REWARD, True
            elif ns == self.HAZARD:
                reward, done = self.HAZARD_REWARD, True
            else:
                reward, done = self.LIVING_REWARD, False
            transitions.append((prob, ns, reward, done))
        return transitions

    def _sample_stochastic_action(self, action: int) -> int:
        r = self._rng.random()
        cumulative = 0.0
        for p, a in self._action_outcomes(action):
            cumulative += p
            if r <= cumulative:
                return a
        return action

    def _action_outcomes(self, action: int) -> List[Tuple[float, int]]:
        # N/S can drift to E/W, E/W can drift to N/S.
        if action in (0, 2):
            side_actions = [1, 3]
        else:
            side_actions = [0, 2]
        return [
            (self.P_INTENDED, action),
            (self.P_SIDE, side_actions[0]),
            (self.P_SIDE, side_actions[1]),
        ]
