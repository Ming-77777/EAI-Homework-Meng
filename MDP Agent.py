import random
from collections import defaultdict

# -----------------------------
# MDP definition
# -----------------------------

ACTIONS = ["North", "South", "East", "West"]

DELTA = {
    "North": (0, 1),
    "South": (0, -1),
    "East":  (1, 0),
    "West":  (-1, 0),
}

LEFT_OF = {
    "North": "West",
    "South": "East",
    "East": "North",
    "West": "South",
}

RIGHT_OF = {
    "North": "East",
    "South": "West",
    "East": "South",
    "West": "North",
}

WIDTH, HEIGHT = 4, 3
WALLS = {(2, 2)}
TERMINALS = {(4, 3): 1.0, (4, 2): -1.0}
STEP_REWARD = -0.04


def all_states(extra_hazards=None):
    extra_hazards = extra_hazards or {}
    terminals = dict(TERMINALS)
    terminals.update(extra_hazards)

    states = []
    for x in range(1, WIDTH + 1):
        for y in range(1, HEIGHT + 1):
            s = (x, y)
            if s not in WALLS:
                states.append(s)
    return states, terminals


def is_valid(s, terminals):
    x, y = s
    if x < 1 or x > WIDTH or y < 1 or y > HEIGHT:
        return False
    if s in WALLS:
        return False
    return True


def move(state, action, terminals):
    if state in terminals:
        return state

    dx, dy = DELTA[action]
    nxt = (state[0] + dx, state[1] + dy)

    if not is_valid(nxt, terminals):
        return state
    return nxt


def rewards(state, terminals):
    if state in terminals:
        return terminals[state]
    return STEP_REWARD


def transitions(state, action, terminals):
    """
    Returns a list of (next_state, probability)
    """
    if state in terminals:
        return [(state, 1.0)]

    candidates = [
        (move(state, action, terminals), 0.8),
        (move(state, LEFT_OF[action], terminals), 0.1),
        (move(state, RIGHT_OF[action], terminals), 0.1),
    ]

    merged = defaultdict(float)
    for s_next, p in candidates:
        merged[s_next] += p

    return list(merged.items())


# -----------------------------
# Value Iteration
# -----------------------------

def value_iteration(states, terminals, gamma=1.0, theta=1e-8):
    V = {}
    for s in states:
        if s in terminals:
            V[s] = terminals[s]
        else:
            V[s] = 0.0

    while True:
        delta = 0.0
        newV = V.copy()

        for s in states:
            if s in terminals:
                continue

            action_values = []
            for a in ACTIONS:
                expected = 0.0
                for s_next, p in transitions(s, a, terminals):
                    expected += p * V[s_next]
                action_values.append(expected)

            newV[s] = rewards(s, terminals) + gamma * max(action_values)
            delta = max(delta, abs(newV[s] - V[s]))

        V = newV
        if delta < theta:
            break

    return V


def extract_policy(states, terminals, V, gamma=1.0):
    policy = {}

    for s in states:
        if s in terminals:
            policy[s] = "TERMINAL"
            continue

        best_action = None
        best_value = float("-inf")

        for a in ACTIONS:
            expected = 0.0
            for s_next, p in transitions(s, a, terminals):
                expected += p * V[s_next]

            q = rewards(s, terminals) + gamma * expected

            if q > best_value:
                best_value = q
                best_action = a

        policy[s] = best_action

    return policy


# -----------------------------
# Part 1: Environment simulator
# -----------------------------

def simulate_step(state, action, terminals):
    dist = transitions(state, action, terminals)
    next_states = [s for s, _ in dist]
    probs = [p for _, p in dist]
    return random.choices(next_states, weights=probs, k=1)[0]


def verify_simulator(trials=10000):
    states, terminals = all_states()
    counts = defaultdict(int)

    start = (3, 1)
    action = "North"

    for _ in range(trials):
        s2 = simulate_step(start, action, terminals)
        counts[s2] += 1

    print("Empirical frequencies from ((3,1), North):")
    for s, c in sorted(counts.items()):
        print(s, round(c / trials, 4))

# Expected roughly:
# (3,2) about 0.8
# (2,1) about 0.1
# (4,1) about 0.1


# -----------------------------
# Part 2: Episode runner
# -----------------------------

def run_episode(policy, terminals, start=(1, 1), max_steps=100):
    state = start
    trajectory = [state]
    total_reward = 0.0

    for _ in range(max_steps):
        if state in terminals:
            if terminals[state] == 1.0:
                outcome = "goal"
            else:
                outcome = "hazard"
            return trajectory, total_reward, outcome

        action = policy[state]
        next_state = simulate_step(state, action, terminals)

        total_reward += rewards(next_state, terminals)
        state = next_state
        trajectory.append(state)

    return trajectory, total_reward, "timeout"


def evaluate_policy(policy, terminals, episodes=1000, start=(1, 1), max_steps=100):
    outcomes = defaultdict(int)
    total_returns = []

    for _ in range(episodes):
        traj, ret, outcome = run_episode(policy, terminals, start=start, max_steps=max_steps)
        outcomes[outcome] += 1
        total_returns.append(ret)

    goal_rate = outcomes["goal"] / episodes
    hazard_rate = outcomes["hazard"] / episodes
    timeout_rate = outcomes["timeout"] / episodes
    avg_return = sum(total_returns) / episodes

    return {
        "goal_rate": goal_rate,
        "hazard_rate": hazard_rate,
        "timeout_rate": timeout_rate,
        "avg_return": avg_return
    }


# -----------------------------
# Part 3: Greedy baseline
# -----------------------------

GOAL = (4, 3)

def greedy_action(state):
    x, y = state
    gx, gy = GOAL

    if x < gx:
        return "East"
    if y < gy:
        return "North"
    if x > gx:
        return "West"
    if y > gy:
        return "South"
    return "North"


def greedy_policy(states, terminals):
    policy = {}
    for s in states:
        if s in terminals:
            policy[s] = "TERMINAL"
        else:
            policy[s] = greedy_action(s)
    return policy


# -----------------------------
# Part 4: Discount factor experiment
# -----------------------------

def gamma_experiment(gammas=(0.1, 0.5, 0.9, 0.99), episodes=1000):
    states, terminals = all_states()
    results = {}

    for gamma in gammas:
        V = value_iteration(states, terminals, gamma=gamma)
        pi = extract_policy(states, terminals, V, gamma=gamma)
        stats = evaluate_policy(pi, terminals, episodes=episodes)
        results[gamma] = {
            "policy": pi,
            "stats": stats
        }
    return results


# -----------------------------
# Part 5: Harder warehouse
# -----------------------------

def harder_warehouse_experiment(episodes=1000, gamma=1.0):
    extra_hazards = {(2, 3): -1.0}
    states, terminals = all_states(extra_hazards=extra_hazards)

    V = value_iteration(states, terminals, gamma=gamma)
    pi = extract_policy(states, terminals, V, gamma=gamma)
    stats = evaluate_policy(pi, terminals, episodes=episodes)

    return pi, stats


# -----------------------------
# Example main
# -----------------------------

if __name__ == "__main__":
    random.seed(42)

    # Part 1
    verify_simulator()

    # Part 2
    states, terminals = all_states()
    V_star = value_iteration(states, terminals, gamma=1.0)
    pi_star = extract_policy(states, terminals, V_star, gamma=1.0)

    optimal_stats = evaluate_policy(pi_star, terminals, episodes=1000)
    print("\nOptimal policy stats:", optimal_stats)

    # Part 3
    pi_greedy = greedy_policy(states, terminals)
    greedy_stats = evaluate_policy(pi_greedy, terminals, episodes=1000)
    print("Greedy policy stats:", greedy_stats)

    # Part 4
    gamma_results = gamma_experiment()
    print("\nGamma experiment:")
    for gamma, res in gamma_results.items():
        print(gamma, res["stats"])

    # Part 5
    hard_pi, hard_stats = harder_warehouse_experiment()
    print("\nHarder warehouse stats:", hard_stats)