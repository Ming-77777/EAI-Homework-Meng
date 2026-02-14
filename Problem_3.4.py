from collections import deque
from z3 import Bool, Bools, And, Or, Not, Solver, unsat
from hazardous_warehouse_env import HazardousWarehouseEnv, Action, Direction
from hazardous_warehouse_viz import configure_rn_example_layout


# ==========================================
# Task 1: Z3 Setup and Entailment
# ==========================================

def z3_entails(solver: Solver, expr) -> bool:
    solver.push()
    solver.add(Not(expr))
    res = solver.check()
    solver.pop()
    return res == unsat


def run_task1_demo() -> None:
    print("--- Task 1 Z3 Demo ---")
    P, Q = Bools("P Q")
    s = Solver()
    s.add(P == Q)
    s.add(P)
    print("Satisfiable:", s.check())
    model = s.model()
    print("Model:", model)
    print("Entails Q:", z3_entails(s, Q))


# ==========================================
# Task 2: Symbols and Physics
# ==========================================

def damaged_at(x: int, y: int):
    return Bool(f"D_{x}_{y}")


def forklift_at(x: int, y: int):
    return Bool(f"F_{x}_{y}")


def creaking_at(x: int, y: int):
    return Bool(f"C_{x}_{y}")


def rumbling_at(x: int, y: int):
    return Bool(f"R_{x}_{y}")


def safe_at(x: int, y: int):
    return Bool(f"S_{x}_{y}")

class WarehouseKB:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.solver = Solver()
        self.cells = [(x, y) for x in range(1, width + 1) for y in range(1, height + 1)]

        self.D = {(x, y): damaged_at(x, y) for x, y in self.cells}  # Damaged
        self.F = {(x, y): forklift_at(x, y) for x, y in self.cells}  # Forklift
        self.C = {(x, y): creaking_at(x, y) for x, y in self.cells}  # Creaking
        self.R = {(x, y): rumbling_at(x, y) for x, y in self.cells}  # Rumbling
        self.S = {(x, y): safe_at(x, y) for x, y in self.cells}  # Safe

        self._build_physics()

    def _get_adj(self, x: int, y: int) -> list[tuple[int, int]]:
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [
            (nx, ny)
            for nx, ny in candidates
            if 1 <= nx <= self.width and 1 <= ny <= self.height
        ]

    def _build_physics(self) -> None:
        for (x, y) in self.cells:
            adj = self._get_adj(x, y)
            self.solver.add(self.C[(x, y)] == Or([self.D[p] for p in adj]))
            self.solver.add(self.R[(x, y)] == Or([self.F[p] for p in adj]))
            self.solver.add(self.S[(x, y)] == And(Not(self.D[(x, y)]), Not(self.F[(x, y)])))

    def tell(self, pos: tuple[int, int], creaking: bool, rumbling: bool) -> None:
        self.solver.add(self.C[pos] == creaking)
        self.solver.add(self.R[pos] == rumbling)

    def mark_safe(self, pos: tuple[int, int]) -> None:
        self.solver.add(self.S[pos])

    def ask_safe(self, pos: tuple[int, int]) -> bool:
        return z3_entails(self.solver, self.S[pos])

    def ask_danger(self, pos: tuple[int, int]) -> bool:
        return z3_entails(self.solver, Or(self.D[pos], self.F[pos]))


def build_warehouse_kb(width: int, height: int) -> WarehouseKB:
    return WarehouseKB(width, height)


# ==========================================
# Task 3: Manual Reasoning
# ==========================================

def run_manual_reasoning() -> None:
    kb = build_warehouse_kb(4, 4)
    kb.mark_safe((1, 1))

    print("--- Task 3 Manual Reasoning Trace ---")
    kb.tell((1, 1), creaking=False, rumbling=False)
    print("At (1,1), no C/R.")
    print("Is (2,1) safe?", kb.ask_safe((2, 1)))
    print("Is (1,2) safe?", kb.ask_safe((1, 2)))

    kb.tell((2, 1), creaking=True, rumbling=False)
    print("At (2,1), Creaking only.")
    print("Is (3,1) safe?", kb.ask_safe((3, 1)))
    print("Is (2,2) safe?", kb.ask_safe((2, 2)))

    kb.tell((1, 2), creaking=False, rumbling=True)
    print("At (1,2), Rumbling only.")
    print("Is (3,1) safe?", kb.ask_safe((3, 1)))
    print("Is (2,2) safe?", kb.ask_safe((2, 2)))
    print("Is (1,3) safe?", kb.ask_safe((1, 3)))
    print("Forklift at (1,3)?", z3_entails(kb.solver, kb.F[(1, 3)]))
    print("Damaged at (3,1)?", z3_entails(kb.solver, kb.D[(3, 1)]))


# ==========================================
# Task 4: Agent Loop with Path Planning
# ==========================================

class WarehouseKBAgent:
    def __init__(self, env: HazardousWarehouseEnv):
        self.env = env
        self.kb = WarehouseKB(env.width, env.height)
        self.visited: set[tuple[int, int]] = set()
        self.known_safe: set[tuple[int, int]] = set()
        self.known_danger: set[tuple[int, int]] = set()
        self.pending_actions: deque[Action] = deque()

    def _update_kb(self, pos: tuple[int, int], percept) -> None:
        self.kb.tell(pos, percept.creaking, percept.rumbling)
        self.kb.mark_safe(pos)
        self.visited.add(pos)
        self.known_safe.add(pos)

        for cell in self.kb.cells:
            if cell in self.known_safe or cell in self.known_danger:
                continue
            if self.kb.ask_safe(cell):
                self.known_safe.add(cell)
            elif self.kb.ask_danger(cell):
                self.known_danger.add(cell)

    def _bfs_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        if goal not in self.known_safe and goal != start:
            return None
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            curr, path = queue.popleft()
            if curr == goal:
                return path
            for nxt in self.kb._get_adj(*curr):
                if nxt in seen or nxt not in self.known_safe:
                    continue
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
        return None

    def _bfs_to_unvisited_safe(self, start: tuple[int, int]) -> list[tuple[int, int]] | None:
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            curr, path = queue.popleft()
            if curr not in self.visited and curr in self.known_safe:
                return path
            for nxt in self.kb._get_adj(*curr):
                if nxt in seen or nxt not in self.known_safe:
                    continue
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
        return None

    def _direction_to(self, src: tuple[int, int], dst: tuple[int, int]) -> Direction:
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        if dx == 1 and dy == 0:
            return Direction.EAST
        if dx == -1 and dy == 0:
            return Direction.WEST
        if dx == 0 and dy == 1:
            return Direction.NORTH
        if dx == 0 and dy == -1:
            return Direction.SOUTH
        raise ValueError("Non-adjacent move")

    def _turn_actions(self, current: Direction, target: Direction) -> list[Action]:
        order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        curr_idx = order.index(current)
        target_idx = order.index(target)
        diff = (target_idx - curr_idx) % 4
        if diff == 0:
            return []
        if diff == 1:
            return [Action.TURN_RIGHT]
        if diff == 2:
            return [Action.TURN_RIGHT, Action.TURN_RIGHT]
        return [Action.TURN_LEFT]

    def _path_to_actions(self, start_dir: Direction, path: list[tuple[int, int]], start_pos: tuple[int, int]) -> list[Action]:
        actions: list[Action] = []
        current_dir = start_dir
        current_pos = start_pos
        for nxt in path:
            desired_dir = self._direction_to(current_pos, nxt)
            turn = self._turn_actions(current_dir, desired_dir)
            actions.extend(turn)
            if turn:
                current_dir = desired_dir
            actions.append(Action.FORWARD)
            current_pos = nxt
        return actions

    def run_episode(self, max_steps: int = 200) -> None:
        percept = self.env.reset()
        done = False

        while not done and self.env.steps < max_steps:
            pos = self.env.robot_position
            self._update_kb(pos, percept)

            if percept.beacon and not self.env.has_package:
                percept, _, done, _ = self.env.step(Action.GRAB)
                continue

            if self.env.has_package and pos == (1, 1):
                percept, _, done, _ = self.env.step(Action.EXIT)
                continue

            if not self.pending_actions:
                path = None
                if self.env.has_package:
                    path = self._bfs_path(pos, (1, 1))
                else:
                    path = self._bfs_to_unvisited_safe(pos)
                    if path is None:
                        path = self._bfs_path(pos, (1, 1))

                if path:
                    actions = self._path_to_actions(self.env.robot_direction, path, pos)
                    self.pending_actions = deque(actions)
                else:
                    if pos == (1, 1):
                        percept, _, done, _ = self.env.step(Action.EXIT)
                    else:
                        break

            if self.pending_actions:
                action = self.pending_actions.popleft()
                percept, _, done, _ = self.env.step(action)


# ==========================================
# Task 5: Testing on Example Layout
# ==========================================

def run_agent_on_example_layout() -> None:
    print("--- Task 5 Agent Run ---")
    env = HazardousWarehouseEnv()
    configure_rn_example_layout(env)
    agent = WarehouseKBAgent(env)
    agent.run_episode()
    print("Steps:", env.steps)
    print("Total reward:", env.total_reward)
    print("Success:", env.get_true_state()["terminated"], env.get_true_state()["success"])


if __name__ == "__main__":
    run_task1_demo()
    run_manual_reasoning()
    run_agent_on_example_layout()