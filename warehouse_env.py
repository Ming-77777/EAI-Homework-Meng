from dataclasses import dataclass
import random
from typing import Dict, List, Tuple, Union
Action = Union[int, str]
@dataclass
class WarehouseState:
    robot_pos: Tuple[int, int]
    has_item: bool
    battery: int
    steps: int
class WarehouseEnv:
    """
    Minimal, Gymnasium-style warehouse environment.
    - reset() -> observation
    - step(action) -> observation, reward, terminated, truncated, info
    """
    # Discrete action set for the agent.
    ACTIONS = ["N", "E", "S", "W", "WAIT", "PICK", "DROP"]
    MOVE_DELTAS = {
        "N": (-1, 0),
        "E": (0, 1),
        "S": (1, 0),
        "W": (0, -1),
    }
    def __init__(
        self,
        grid: List[str] | None = None,
        start_pos: Tuple[int, int] = (1, 1),
        max_steps: int = 200,
        battery: int = 200,
        view_radius: int = 2,
    ) -> None:
        # Legend: # = wall, . = empty, P = pickup, D = dropoff.
        self.grid = grid or [
            "############",
            "#..P....#..#",
            "#..##...#..#",
            "#......##..#",
            "#..#.......#",
            "#..#..D....#",
            "############",
        ] # Default warehouse layout
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.start_pos = start_pos
        self.max_steps = max_steps
        self.max_battery = battery
        self.view_radius = view_radius
        self.state = WarehouseState(
            robot_pos=self.start_pos,
            has_item=False,
            battery=self.max_battery,
            steps=0,
        )
    def reset(self, randomize: bool = False) -> Dict[str, object]:
        start_pos = self.start_pos
        if randomize:
            self._randomize_pickup_dropoff()
            start_pos = self._random_empty_cell()
        self.state = WarehouseState(
            robot_pos=start_pos,
            has_item=False,
            battery=self.max_battery,
            steps=0,
        )
        return self._observe()
    def step(self, action: Action) -> Tuple[Dict[str, object], float, bool, bool, Dict[str, object]]:
        act = self._normalize_action(action)
        # Small step penalty encourages shorter paths.
        reward = -0.1
        terminated = False
        truncated = False
        info: Dict[str, object] = {}
        if act in self.MOVE_DELTAS:
            reward += self._move(act)
        elif act == "WAIT":
            reward -= 0.05
        elif act == "PICK":
            reward += self._pick()
        elif act == "DROP":
            reward += self._drop()
            if reward >= 9.0:
                terminated = True
        else:
            reward -= 0.5
            info["invalid_action"] = True
        self.state.steps += 1
        self.state.battery -= 1
        # Truncate when the time or battery budget runs out.
        if self.state.steps >= self.max_steps or self.state.battery <= 0:
            truncated = True
        return self._observe(), reward, terminated, truncated, info
    def render_grid(self) -> List[List[str]]:
        """Return a 2D grid of characters for animation or visualization."""
        rows = [list(r) for r in self.grid]
        r, c = self.state.robot_pos
        rows[r][c] = "R" if not self.state.has_item else "r"
        return rows
    def render(self) -> str:
        # Render the full grid with the robot position overlaid.
        rows = self.render_grid()
        return "\n".join("".join(r) for r in rows)
    def render_with_legend(self) -> str:
        legend = [
            "Legend:",
            "# = wall",
            ". = empty",
            "P = pickup",
            "D = dropoff",
            "R = robot (empty)",
            "r = robot (loaded)",
        ]
        return f"{self.render()}\n\n" + "\n".join(legend)
    def _normalize_action(self, action: Action) -> str:
        if isinstance(action, int):
            if 0 <= action < len(self.ACTIONS):
                return self.ACTIONS[action]
            return "INVALID"
        return action.upper()
    def _move(self, act: str) -> float:
        dr, dc = self.MOVE_DELTAS[act]
        r, c = self.state.robot_pos
        nr, nc = r + dr, c + dc
        if self._is_wall(nr, nc):
            return -1.0
        self.state.robot_pos = (nr, nc)
        return 0.0
    def _pick(self) -> float:
        r, c = self.state.robot_pos
        # Successful pickup is only allowed on a pickup tile.
        if self.grid[r][c] == "P" and not self.state.has_item:
            self.state.has_item = True
            return 5.0
        return -0.5
    def _drop(self) -> float:
        r, c = self.state.robot_pos
        # Successful drop is only allowed on a dropoff tile.
        if self.grid[r][c] == "D" and self.state.has_item:
            self.state.has_item = False
            return 10.0
        return -0.5
    def _is_wall(self, r: int, c: int) -> bool:
        if r < 0 or c < 0 or r >= self.height or c >= self.width:
            return True
        return self.grid[r][c] == "#"
    def _observe(self) -> Dict[str, object]:
        # Local observation centered on the robot, using view_radius.
        r, c = self.state.robot_pos
        local = []
        for dr in range(-self.view_radius, self.view_radius + 1):
            row = []
            for dc in range(-self.view_radius, self.view_radius + 1):
                rr, cc = r + dr, c + dc
                if rr < 0 or cc < 0 or rr >= self.height or cc >= self.width:
                    row.append("#")
                elif (rr, cc) == self.state.robot_pos:
                    row.append("R" if not self.state.has_item else "r")
                else:
                    row.append(self.grid[rr][cc])
            local.append("".join(row))
        pickup_pos = self._find_tile("P")
        dropoff_pos = self._find_tile("D")
        return {
            "local_grid": local,
            "robot_pos": self.state.robot_pos,
            "has_item": self.state.has_item,
            "battery": self.state.battery,
            "steps": self.state.steps,
            "pickup_pos": pickup_pos,
            "dropoff_pos": dropoff_pos,
        }
    def _random_empty_cell(self) -> Tuple[int, int]:
        empties = []
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == ".":
                    empties.append((r, c))
        if not empties:
            return self.start_pos
        return random.choice(empties)
    def _randomize_pickup_dropoff(self) -> None:
        # Convert to mutable grid.
        rows = [list(r) for r in self.grid]
        positions = []
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch in {"P", "D"}:
                    rows[r][c] = "."
                if ch == ".":
                    positions.append((r, c))
        if len(positions) < 2:
            self.grid = ["".join(r) for r in rows]
            return
        pickup = random.choice(positions)
        positions.remove(pickup)
        dropoff = random.choice(positions)
        pr, pc = pickup
        dr, dc = dropoff
        rows[pr][pc] = "P"
        rows[dr][dc] = "D"
        self.grid = ["".join(r) for r in rows]
    def _find_tile(self, tile: str) -> Tuple[int, int] | None:
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == tile:
                    return (r, c)
        return None