from collections import deque
from z3 import *

from hazardous_warehouse_env import HazardousWarehouseEnv, Action, Direction
try:
    from hazardous_warehouse_viz import configure_rn_example_layout
except Exception:
    configure_rn_example_layout = None


class FOLWarehouseKB:
    """
    Environment-compatible FOL-style KB.

    We still use location terms and predicates such as Damaged(L), Forklift(L),
    Creaking(L), Rumbling(L), Safe(L), and Adjacent(L1,L2), but for efficient
    execution inside the provided environment we ground the 3 physics schemas
    over the finite 4x4 grid.
    """

    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height

        self.Location, consts = EnumSort(
            "Location", [f"L_{x}_{y}" for x in range(1, width + 1) for y in range(1, height + 1)]
        )
        self.locs = {}
        idx = 0
        for x in range(1, width + 1):
            for y in range(1, height + 1):
                self.locs[(x, y)] = consts[idx]
                idx += 1

        self.Adjacent = Function("Adjacent", self.Location, self.Location, BoolSort())
        self.Damaged = Function("Damaged", self.Location, BoolSort())
        self.Forklift = Function("Forklift", self.Location, BoolSort())
        self.Creaking = Function("Creaking", self.Location, BoolSort())
        self.Rumbling = Function("Rumbling", self.Location, BoolSort())
        self.Safe = Function("Safe", self.Location, BoolSort())

        self.solver = Solver()
        self._add_adjacency_axioms()
        self._ground_physics_rules()
        self.solver.add(Not(self.Damaged(self.loc(1, 1))))
        self.solver.add(Not(self.Forklift(self.loc(1, 1))))

    def loc(self, x, y):
        return self.locs[(x, y)]

    def in_bounds(self, x, y):
        return 1 <= x <= self.width and 1 <= y <= self.height

    def neighbors(self, x, y):
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [(nx, ny) for (nx, ny) in candidates if self.in_bounds(nx, ny)]

    def _add_adjacency_axioms(self):
        for x1 in range(1, self.width + 1):
            for y1 in range(1, self.height + 1):
                nbrs = set(self.neighbors(x1, y1))
                for x2 in range(1, self.width + 1):
                    for y2 in range(1, self.height + 1):
                        L1 = self.loc(x1, y1)
                        L2 = self.loc(x2, y2)
                        if (x2, y2) in nbrs:
                            self.solver.add(self.Adjacent(L1, L2))
                        else:
                            self.solver.add(Not(self.Adjacent(L1, L2)))

    def _ground_physics_rules(self):
        for x in range(1, self.width + 1):
            for y in range(1, self.height + 1):
                L = self.loc(x, y)
                nbr_terms = [self.loc(nx, ny) for (nx, ny) in self.neighbors(x, y)]

                self.solver.add(
                    self.Creaking(L) == Or([self.Damaged(n) for n in nbr_terms])
                )
                self.solver.add(
                    self.Rumbling(L) == Or([self.Forklift(n) for n in nbr_terms])
                )
                self.solver.add(
                    self.Safe(L) == And(Not(self.Damaged(L)), Not(self.Forklift(L)))
                )

    def tell_percepts(self, pos, percept):
        L = self.loc(*pos)
        self.solver.add(self.Safe(L))
        self.solver.add(self.Creaking(L) if percept.creaking else Not(self.Creaking(L)))
        self.solver.add(self.Rumbling(L) if percept.rumbling else Not(self.Rumbling(L)))
        if self.solver.check() != sat:
            raise RuntimeError("KB became inconsistent after percept update")

    def entails(self, formula):
        self.solver.push()
        self.solver.add(Not(formula))
        result = self.solver.check() == unsat
        self.solver.pop()
        return result

    def known_safe(self):
        return {pos for pos in self.locs if self.entails(self.Safe(self.loc(*pos)))}

    def known_dangerous(self):
        return {pos for pos in self.locs if self.entails(Not(self.Safe(self.loc(*pos))))}


def direction_for_step(src, dst):
    return {
        (0, 1): Direction.NORTH,
        (1, 0): Direction.EAST,
        (0, -1): Direction.SOUTH,
        (-1, 0): Direction.WEST,
    }[(dst[0] - src[0], dst[1] - src[1])]


def turn_actions(current, target):
    order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
    i = order.index(current)
    j = order.index(target)
    right = (j - i) % 4
    left = (i - j) % 4
    return [Action.TURN_RIGHT] * right if right <= left else [Action.TURN_LEFT] * left


def bfs_path(start, goals, traversable, width=4, height=4):
    if start in goals:
        return [start]
    q = deque([(start, [start])])
    seen = {start}
    while q:
        pos, path = q.popleft()
        x, y = pos
        for nxt in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if not (1 <= nxt[0] <= width and 1 <= nxt[1] <= height):
                continue
            if nxt in seen or nxt not in traversable:
                continue
            new_path = path + [nxt]
            if nxt in goals:
                return new_path
            seen.add(nxt)
            q.append((nxt, new_path))
    return None


def path_to_actions(path, facing):
    if path is None or len(path) < 2:
        return []
    actions = []
    current = facing
    for i in range(len(path) - 1):
        need = direction_for_step(path[i], path[i + 1])
        actions.extend(turn_actions(current, need))
        actions.append(Action.FORWARD)
        current = need
    return actions


class FOLWarehouseAgent:
    def __init__(self, width=4, height=4, verbose=True):
        self.kb = FOLWarehouseKB(width, height)
        self.width = width
        self.height = height
        self.verbose = verbose
        self.visited = set()
        self.plan = []

    def log(self, *args):
        if self.verbose:
            print(*args)

    def choose_action(self, env, percept):
        current = env.robot_position
        self.visited.add(current)
        self.kb.tell_percepts(current, percept)

        safe_cells = self.kb.known_safe()
        dangerous_cells = self.kb.known_dangerous()
        self.log(f"\n[Agent] pos={current}, facing={env.robot_direction.name}, percept={percept}")
        self.log(f"[Agent] safe={sorted(safe_cells)}")
        self.log(f"[Agent] dangerous={sorted(dangerous_cells)}")

        if self.plan:
            action = self.plan.pop(0)
            self.log(f"[Agent] execute planned action: {action.name}")
            return action

        if percept.beacon and not env.has_package:
            self.log("[Agent] beacon -> GRAB")
            return Action.GRAB

        if env.has_package:
            if current == (1, 1):
                self.log("[Agent] back at exit with package -> EXIT")
                return Action.EXIT
            path = bfs_path(current, {(1, 1)}, safe_cells, self.width, self.height)
            if path:
                self.plan = path_to_actions(path, env.robot_direction)
                action = self.plan.pop(0)
                self.log(f"[Agent] path home: {path}")
                return action
            return Action.TURN_LEFT

        safe_unvisited = safe_cells - self.visited
        if safe_unvisited:
            path = bfs_path(current, safe_unvisited, safe_cells, self.width, self.height)
            if path and len(path) > 1:
                self.plan = path_to_actions(path, env.robot_direction)
                action = self.plan.pop(0)
                self.log(f"[Agent] path to safe frontier: {path}")
                return action

        if current == (1, 1):
            self.log("[Agent] no more provably safe moves -> EXIT")
            return Action.EXIT

        path = bfs_path(current, {(1, 1)}, safe_cells, self.width, self.height)
        if path and len(path) > 1:
            self.plan = path_to_actions(path, env.robot_direction)
            action = self.plan.pop(0)
            self.log(f"[Agent] conservative return path: {path}")
            return action

        return Action.TURN_LEFT


def run_problem_4_1_example(verbose=True, reveal=True):
    env = HazardousWarehouseEnv(seed=0)
    if configure_rn_example_layout is not None:
        configure_rn_example_layout(env)

    agent = FOLWarehouseAgent(width=env.width, height=env.height, verbose=verbose)
    percept = env._last_percept

    if reveal:
        print("=== True state ===")
        print(env.render(reveal=True))
        print("\n=== Agent view ===")
        print(env.render(reveal=False))

    done = False
    while not done:
        action = agent.choose_action(env, percept)
        percept, reward, done, info = env.step(action)
        if verbose:
            print(f"[Env] action={action.name}, reward={reward}, done={done}, info={info}")
            print(env.render(reveal=False))

    print("\n=== Episode finished ===")
    print(f"alive={env.is_alive}, has_package={env.has_package}, steps={env.steps}, total_reward={env.total_reward}")
    print("\n=== Final true state ===")
    print(env.render(reveal=True))


if __name__ == "__main__":
    run_problem_4_1_example(verbose=True, reveal=True)