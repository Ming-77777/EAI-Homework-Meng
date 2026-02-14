"""
Hazardous Warehouse Visualization

Visualization and animation utilities for the Hazardous Warehouse environment.
Provides grid rendering, percept display, and episode replay animations.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hazardous_warehouse_env import HazardousWarehouseEnv, Direction


# -----------------------------------------------------------------------------
# Color Schemes
# -----------------------------------------------------------------------------

# Color palette for grid elements
COLORS = {
    # Terrain
    "empty": (0.95, 0.95, 0.95),
    "wall": (0.2, 0.2, 0.2),
    "unknown": (0.7, 0.7, 0.7),
    "exit": (0.3, 0.8, 0.3),

    # Hazards
    "damaged": (1.0, 0.5, 0.0),      # Orange - damaged floor
    "forklift": (1.0, 0.0, 0.0),     # Red - active forklift
    "forklift_dead": (0.5, 0.5, 0.5), # Gray - disabled forklift

    # Package
    "package": (1.0, 0.85, 0.0),     # Gold

    # Robot states
    "robot": (0.2, 0.5, 0.9),        # Blue - robot without package
    "robot_loaded": (0.6, 0.2, 0.8), # Purple - robot with package
    "robot_dead": (0.1, 0.1, 0.1),   # Black - destroyed robot

    # Percept indicators
    "creaking": (1.0, 0.5, 0.0),     # Orange - adjacent to damaged floor
    "rumbling": (1.0, 0.0, 0.0),     # Red - adjacent to forklift
    "safe": (0.7, 0.95, 0.7),        # Light green
    "uncertain": (1.0, 0.8, 0.5),    # Light orange
}


# -----------------------------------------------------------------------------
# Grid Rendering
# -----------------------------------------------------------------------------

def state_to_grid(
    env: "HazardousWarehouseEnv",
    reveal: bool = False,
    show_percepts: bool = True,
    known_safe: set[tuple[int, int]] | None = None,
    known_dangerous: set[tuple[int, int]] | None = None,
) -> list[list[tuple[float, float, float]]]:
    """
    Convert environment state to RGB grid for visualization.

    Parameters:
        env: The HazardousWarehouseEnv instance
        reveal: If True, show all hazards; if False, show only known info
        show_percepts: If True, tint squares based on percepts
        known_safe: Set of positions the agent has deduced are safe
        known_dangerous: Set of positions the agent has deduced are dangerous
    """
    true_state = env.get_true_state()
    width, height = true_state["width"], true_state["height"]
    damaged = set(tuple(p) for p in true_state["damaged"])
    forklift = tuple(true_state["forklift"]) if true_state["forklift"] else None
    forklift_alive = true_state["forklift_alive"]
    package = tuple(true_state["package"]) if true_state["package"] else None
    robot = true_state["robot"]
    robot_pos = (robot["x"], robot["y"])

    known_safe = known_safe or set()
    known_dangerous = known_dangerous or set()

    # Build grid (note: y is inverted for display, row 0 = top = max y)
    grid = []
    for row in range(height, 0, -1):  # Top to bottom
        grid_row = []
        for col in range(1, width + 1):  # Left to right
            pos = (col, row)

            # Determine cell color
            if pos == robot_pos:
                if not robot["alive"]:
                    color = COLORS["robot_dead"]
                elif robot["has_package"]:
                    color = COLORS["robot_loaded"]
                else:
                    color = COLORS["robot"]
            elif reveal:
                if pos in damaged:
                    color = COLORS["damaged"]
                elif pos == forklift:
                    color = COLORS["forklift"] if forklift_alive else COLORS["forklift_dead"]
                elif pos == package and not robot["has_package"]:
                    color = COLORS["package"]
                elif pos == (1, 1):
                    color = COLORS["exit"]
                else:
                    color = COLORS["empty"]
            else:
                # Agent's view - only show what's known
                if pos in known_dangerous:
                    color = COLORS["damaged"]  # Known dangerous
                elif pos in known_safe:
                    if pos == (1, 1):
                        color = COLORS["exit"]
                    else:
                        color = COLORS["safe"]
                else:
                    color = COLORS["unknown"]

            grid_row.append(color)
        grid.append(grid_row)

    return grid


def render_percept_overlay(
    grid: list[list[tuple[float, float, float]]],
    env: "HazardousWarehouseEnv",
    alpha: float = 0.3,
) -> list[list[tuple[float, float, float]]]:
    """
    Add percept indicators as color overlay on grid.
    Squares adjacent to robot are tinted based on current percepts.
    """
    true_state = env.get_true_state()
    robot = true_state["robot"]
    robot_pos = (robot["x"], robot["y"])
    width, height = true_state["width"], true_state["height"]

    # Get current percept from environment
    percept = env._last_percept

    # Find adjacent squares
    x, y = robot_pos
    adjacent = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    adjacent = [(ax, ay) for ax, ay in adjacent if 1 <= ax <= width and 1 <= ay <= height]

    # Create copy of grid
    new_grid = [row.copy() for row in grid]

    for ax, ay in adjacent:
        # Convert to grid coordinates (row 0 = top = max y)
        grid_row = height - ay
        grid_col = ax - 1

        if 0 <= grid_row < len(new_grid) and 0 <= grid_col < len(new_grid[0]):
            base_color = new_grid[grid_row][grid_col]

            # Blend with percept colors
            if percept.creaking and percept.rumbling:
                overlay = tuple((c1 + c2) / 2 for c1, c2 in
                               zip(COLORS["creaking"], COLORS["rumbling"]))
            elif percept.creaking:
                overlay = COLORS["creaking"]
            elif percept.rumbling:
                overlay = COLORS["rumbling"]
            else:
                continue

            # Alpha blend
            blended = tuple(
                base_color[i] * (1 - alpha) + overlay[i] * alpha
                for i in range(3)
            )
            new_grid[grid_row][grid_col] = blended

    return new_grid


# -----------------------------------------------------------------------------
# Static Visualization
# -----------------------------------------------------------------------------

def plot_state(
    env: "HazardousWarehouseEnv",
    ax=None,
    reveal: bool = False,
    show_percepts: bool = True,
    known_safe: set[tuple[int, int]] | None = None,
    known_dangerous: set[tuple[int, int]] | None = None,
    title: str | None = None,
):
    """
    Plot the current environment state.

    Parameters:
        env: The HazardousWarehouseEnv instance
        ax: Matplotlib axes (created if None)
        reveal: Show hidden hazards
        show_percepts: Show percept overlays
        known_safe: Squares agent knows are safe
        known_dangerous: Squares agent knows are dangerous
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrow, Circle
    except ImportError:
        print("matplotlib not available")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Get grid colors
    grid = state_to_grid(env, reveal=reveal,
                         known_safe=known_safe, known_dangerous=known_dangerous)
    if show_percepts:
        grid = render_percept_overlay(grid, env)

    # Display grid
    ax.imshow(grid, interpolation="nearest", aspect="equal")

    # Get state info
    true_state = env.get_true_state()
    width, height = true_state["width"], true_state["height"]
    robot = true_state["robot"]

    # Draw robot direction arrow
    if robot["alive"]:
        # Convert robot position to grid coordinates
        rx, ry = robot["x"], robot["y"]
        grid_x = rx - 0.5  # Center of cell (0-indexed from left)
        grid_y = height - ry - 0.5  # Center of cell (0-indexed from top)

        # Direction vectors
        dir_map = {
            "NORTH": (0, -0.3),
            "EAST": (0.3, 0),
            "SOUTH": (0, 0.3),
            "WEST": (-0.3, 0),
        }
        dx, dy = dir_map.get(robot["direction"], (0, 0))

        ax.annotate(
            "",
            xy=(grid_x + dx, grid_y + dy),
            xytext=(grid_x - dx * 0.5, grid_y - dy * 0.5),
            arrowprops=dict(arrowstyle="->", color="white", lw=2),
        )

    # Draw grid lines
    for i in range(width + 1):
        ax.axvline(i - 0.5, color="gray", linewidth=0.5)
    for i in range(height + 1):
        ax.axhline(i - 0.5, color="gray", linewidth=0.5)

    # Axis labels
    ax.set_xticks(range(width))
    ax.set_xticklabels(range(1, width + 1))
    ax.set_yticks(range(height))
    ax.set_yticklabels(range(height, 0, -1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if title:
        ax.set_title(title)
    else:
        percept = env._last_percept
        percept_str = []
        if percept.creaking:
            percept_str.append("Creaking")
        if percept.rumbling:
            percept_str.append("Rumbling")
        if percept.beacon:
            percept_str.append("Beacon")
        if percept.bump:
            percept_str.append("Bump")
        if percept.beep:
            percept_str.append("Beep")
        if not percept_str:
            percept_str.append("None")

        status = f"Step {env.steps} | Percepts: {', '.join(percept_str)}"
        if robot["has_package"]:
            status += " | Has Package"
        ax.set_title(status)

    return ax


def plot_legend(ax=None):
    """Plot a legend for the visualization colors."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib not available")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))

    legend_items = [
        ("Robot (empty)", COLORS["robot"]),
        ("Robot (with package)", COLORS["robot_loaded"]),
        ("Robot (destroyed)", COLORS["robot_dead"]),
        ("Exit (1,1)", COLORS["exit"]),
        ("Empty/Safe", COLORS["safe"]),
        ("Unknown", COLORS["unknown"]),
        ("Damaged floor", COLORS["damaged"]),
        ("Forklift (active)", COLORS["forklift"]),
        ("Forklift (disabled)", COLORS["forklift_dead"]),
        ("Package", COLORS["package"]),
        ("Creaking zone", COLORS["creaking"]),
        ("Rumbling zone", COLORS["rumbling"]),
    ]

    handles = [
        Patch(facecolor=color, edgecolor="gray", label=label)
        for label, color in legend_items
    ]

    ax.legend(handles=handles, loc="center", frameon=False, fontsize=10)
    ax.set_axis_off()
    return ax


# -----------------------------------------------------------------------------
# Figure Generation (Static Exports)
# -----------------------------------------------------------------------------

def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Use a clean style
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 150
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["axes.labelsize"] = 11

    return plt


def get_adjacent(pos, width, height):
    """Return list of adjacent positions (cardinal directions only)."""
    x, y = pos
    candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    return [(ax, ay) for ax, ay in candidates if 1 <= ax <= width and 1 <= ay <= height]


def create_grid_figure(plt, env, output_path):
    """
    Create the main grid layout figure for 'The Grid' section.

    Shows:
    - 4x4 grid with coordinates
    - Robot at (1,1) facing east
    - Hidden hazards revealed with icons
    - Creaking and rumbling sensor zones
    - Legend explaining symbols
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    true_state = env.get_true_state()
    width, height = true_state["width"], true_state["height"]
    damaged = set(tuple(p) for p in true_state["damaged"])
    forklift = tuple(true_state["forklift"])
    package = tuple(true_state["package"])
    robot = true_state["robot"]

    # Compute sensor zones
    creaking_zones = set()  # Squares adjacent to damaged floor
    for d in damaged:
        for adj in get_adjacent(d, width, height):
            if adj not in damaged:
                creaking_zones.add(adj)

    rumbling_zones = set()  # Squares adjacent to forklift
    for adj in get_adjacent(forklift, width, height):
        rumbling_zones.add(adj)

    # Draw grid cells
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            pos = (x, y)
            plot_x = x - 1
            plot_y = y - 1

            # Determine color
            if pos == (1, 1):
                color = COLORS["exit"]
            elif pos in damaged:
                color = COLORS["damaged"]
            elif pos == forklift:
                color = COLORS["forklift"]
            elif pos == package:
                color = COLORS["package"]
            else:
                color = COLORS["empty"]

            # Draw cell
            rect = plt.Rectangle(
                (plot_x, plot_y),
                1,
                1,
                facecolor=color,
                edgecolor="gray",
                linewidth=1.5,
            )
            ax.add_patch(rect)

            # Draw sensor zone indicators (full diagonal hatching)
            if pos in creaking_zones and pos not in damaged and pos != forklift:
                # Creaking indicator: diagonal lines from bottom-left to top-right
                for i in range(5):
                    offset = -0.6 + i * 0.4
                    ax.plot(
                        [plot_x + max(0, offset), plot_x + min(1, offset + 1)],
                        [plot_y + max(0, -offset), plot_y + min(1, 1 - offset)],
                        color=COLORS["creaking"],
                        linewidth=6,
                        alpha=0.7,
                    )

            if pos in rumbling_zones and pos not in damaged and pos != forklift:
                # Rumbling indicator: diagonal lines from bottom-right to top-left
                for i in range(5):
                    offset = -0.6 + i * 0.4
                    ax.plot(
                        [plot_x + 1 - max(0, offset), plot_x + 1 - min(1, offset + 1)],
                        [plot_y + max(0, -offset), plot_y + min(1, 1 - offset)],
                        color=COLORS["rumbling"],
                        linewidth=6,
                        alpha=0.7,
                    )

            # Add icons for special cells
            if pos in damaged:
                # Draw construction/hazard symbol (X pattern)
                cx, cy = plot_x + 0.5, plot_y + 0.5
                ax.plot(
                    [cx - 0.25, cx + 0.25],
                    [cy - 0.25, cy + 0.25],
                    color="white",
                    linewidth=4,
                    solid_capstyle="round",
                )
                ax.plot(
                    [cx - 0.25, cx + 0.25],
                    [cy + 0.25, cy - 0.25],
                    color="white",
                    linewidth=4,
                    solid_capstyle="round",
                )
                ax.plot(
                    [cx - 0.25, cx + 0.25],
                    [cy - 0.25, cy + 0.25],
                    color="black",
                    linewidth=2,
                    solid_capstyle="round",
                )
                ax.plot(
                    [cx - 0.25, cx + 0.25],
                    [cy + 0.25, cy - 0.25],
                    color="black",
                    linewidth=2,
                    solid_capstyle="round",
                )
            elif pos == forklift:
                # Draw forklift symbol
                cx, cy = plot_x + 0.5, plot_y + 0.5
                # Forklift body (cab)
                cab = plt.Rectangle(
                    (cx - 0.15, cy - 0.1),
                    0.25,
                    0.3,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=2,
                )
                ax.add_patch(cab)
                # Mast (vertical bar)
                ax.plot([cx + 0.15, cx + 0.15], [cy - 0.15, cy + 0.25], color="black", linewidth=3)
                # Forks (two horizontal prongs)
                ax.plot([cx + 0.15, cx + 0.35], [cy - 0.05, cy - 0.05], color="black", linewidth=3)
                ax.plot([cx + 0.15, cx + 0.35], [cy - 0.15, cy - 0.15], color="black", linewidth=3)
                # Wheel
                wheel = plt.Circle((cx - 0.05, cy - 0.18), 0.08, facecolor="black", edgecolor="black")
                ax.add_patch(wheel)
            elif pos == package:
                # Draw package/box symbol
                cx, cy = plot_x + 0.5, plot_y + 0.5
                # Box outline
                box = plt.Rectangle(
                    (cx - 0.25, cy - 0.2),
                    0.5,
                    0.4,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=2,
                )
                ax.add_patch(box)
                # Ribbon/tape horizontal
                ax.plot([cx - 0.25, cx + 0.25], [cy, cy], color="#8B4513", linewidth=3)
                # Ribbon/tape vertical
                ax.plot([cx, cx], [cy - 0.2, cy + 0.2], color="#8B4513", linewidth=3)

    # Draw robot
    robot_x, robot_y = robot["x"], robot["y"]
    plot_rx = robot_x - 1 + 0.5
    plot_ry = robot_y - 1 + 0.5

    # Robot body (circle)
    robot_circle = plt.Circle(
        (plot_rx, plot_ry), 0.35, facecolor=COLORS["robot"], edgecolor="white", linewidth=2
    )
    ax.add_patch(robot_circle)

    # Robot direction arrow
    dir_deltas = {
        "NORTH": (0, 0.25),
        "EAST": (0.25, 0),
        "SOUTH": (0, -0.25),
        "WEST": (-0.25, 0),
    }
    dx, dy = dir_deltas[robot["direction"]]
    ax.annotate(
        "",
        xy=(plot_rx + dx * 1.5, plot_ry + dy * 1.5),
        xytext=(plot_rx, plot_ry),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2),
    )

    # Add "R" label on robot
    ax.text(plot_rx, plot_ry, "R", ha="center", va="center", fontsize=12, fontweight="bold", color="white")

    # Set axis properties
    ax.set_xlim(-0.1, width + 0.1)
    ax.set_ylim(-0.1, height + 0.1)
    ax.set_aspect("equal")

    # Axis labels with coordinates
    ax.set_xticks([i + 0.5 for i in range(width)])
    ax.set_xticklabels([str(i + 1) for i in range(width)])
    ax.set_yticks([i + 0.5 for i in range(height)])
    ax.set_yticklabels([str(i + 1) for i in range(height)])
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)

    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor=COLORS["robot"], edgecolor="white", label="Robot (R)"),
        Patch(facecolor=COLORS["exit"], edgecolor="gray", label="Exit (1,1)"),
        Patch(facecolor=COLORS["damaged"], edgecolor="gray", label="Damaged floor (X)"),
        Patch(facecolor=COLORS["forklift"], edgecolor="gray", label="Forklift (forklift)"),
        Patch(facecolor=COLORS["package"], edgecolor="gray", label="Package"),
        Patch(facecolor=COLORS["empty"], edgecolor="gray", label="Empty"),
        Line2D([0], [0], color=COLORS["creaking"], linewidth=2, label="Creaking zone"),
        Line2D([0], [0], color=COLORS["rumbling"], linewidth=2, label="Rumbling zone"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        fancybox=True,
    )

    ax.set_title("Hazardous Warehouse: True State (Hidden from Agent)", fontsize=13)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def create_reasoning_step_figure(
    plt,
    env,
    title,
    subtitle,
    known_safe,
    known_dangerous,
    known_forklift=None,
    known_creaking=None,
    known_rumbling=None,
    uncertain=None,
    annotations=None,
    output_path=None,
):
    """Create a figure showing one step of the reasoning process."""
    fig, (ax_agent, ax_true) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"wspace": 0.15})

    true_state = env.get_true_state()
    width, height = true_state["width"], true_state["height"]
    damaged = set(tuple(p) for p in true_state["damaged"])
    forklift = tuple(true_state["forklift"])
    package = tuple(true_state["package"])
    robot = true_state["robot"]

    uncertain = uncertain or set()
    known_forklift = known_forklift or set()
    known_creaking = known_creaking or set()
    known_rumbling = known_rumbling or set()
    annotations = annotations or []

    # Compute sensor zones
    creaking_zones = set()
    for d in damaged:
        for adj in get_adjacent(d, width, height):
            if adj not in damaged:
                creaking_zones.add(adj)

    rumbling_zones = set()
    for adj in get_adjacent(forklift, width, height):
        rumbling_zones.add(adj)

    # Draw both views
    for ax, view_type in [(ax_agent, "agent"), (ax_true, "true")]:
        for y in range(1, height + 1):
            for x in range(1, width + 1):
                pos = (x, y)
                plot_x = x - 1
                plot_y = y - 1

                # Determine color based on view type
                if view_type == "true":
                    # True state view
                    if pos == (1, 1):
                        color = COLORS["exit"]
                    elif pos in damaged:
                        color = COLORS["damaged"]
                    elif pos == forklift:
                        color = COLORS["forklift"]
                    elif pos == package:
                        color = COLORS["package"]
                    else:
                        color = COLORS["empty"]
                else:
                    # Agent's view
                    if pos in known_forklift:
                        color = COLORS["forklift"]
                    elif pos in known_dangerous:
                        color = COLORS["damaged"]
                    elif pos in uncertain:
                        color = COLORS["uncertain"]
                    elif pos in known_safe:
                        if pos == (1, 1):
                            color = COLORS["exit"]
                        else:
                            color = COLORS["safe"]
                    else:
                        color = COLORS["unknown"]

                rect = plt.Rectangle((plot_x, plot_y), 1, 1, facecolor=color, edgecolor="gray", linewidth=1.5)
                ax.add_patch(rect)

                if view_type == "agent":
                    if pos in known_creaking:
                        # Creaking indicator: diagonal lines from bottom-left to top-right
                        for i in range(5):
                            offset = -0.6 + i * 0.4
                            ax.plot(
                                [plot_x + max(0, offset), plot_x + min(1, offset + 1)],
                                [plot_y + max(0, -offset), plot_y + min(1, 1 - offset)],
                                color=COLORS["creaking"],
                                linewidth=6,
                                alpha=0.7,
                            )

                    if pos in known_rumbling:
                        # Rumbling indicator: diagonal lines from bottom-right to top-left
                        for i in range(5):
                            offset = -0.6 + i * 0.4
                            ax.plot(
                                [plot_x + 1 - max(0, offset), plot_x + 1 - min(1, offset + 1)],
                                [plot_y + max(0, -offset), plot_y + min(1, 1 - offset)],
                                color=COLORS["rumbling"],
                                linewidth=6,
                                alpha=0.7,
                            )

                # Add sensor zone indicators for true state view
                if view_type == "true":
                    if pos in creaking_zones and pos not in damaged and pos != forklift:
                        # Creaking indicator: diagonal lines from bottom-left to top-right
                        for i in range(5):
                            offset = -0.6 + i * 0.4
                            ax.plot(
                                [plot_x + max(0, offset), plot_x + min(1, offset + 1)],
                                [plot_y + max(0, -offset), plot_y + min(1, 1 - offset)],
                                color=COLORS["creaking"],
                                linewidth=6,
                                alpha=0.7,
                            )

                    if pos in rumbling_zones and pos not in damaged and pos != forklift:
                        # Rumbling indicator: diagonal lines from bottom-right to top-left
                        for i in range(5):
                            offset = -0.6 + i * 0.4
                            ax.plot(
                                [plot_x + 1 - max(0, offset), plot_x + 1 - min(1, offset + 1)],
                                [plot_y + max(0, -offset), plot_y + min(1, 1 - offset)],
                                color=COLORS["rumbling"],
                                linewidth=6,
                                alpha=0.7,
                            )

                # Add icons for true state view
                if view_type == "true" and pos != (robot["x"], robot["y"]):
                    cx, cy = plot_x + 0.5, plot_y + 0.5
                    if pos in damaged:
                        # Draw X pattern for damaged floor
                        ax.plot(
                            [cx - 0.2, cx + 0.2],
                            [cy - 0.2, cy + 0.2],
                            color="white",
                            linewidth=3,
                            solid_capstyle="round",
                        )
                        ax.plot(
                            [cx - 0.2, cx + 0.2],
                            [cy + 0.2, cy - 0.2],
                            color="white",
                            linewidth=3,
                            solid_capstyle="round",
                        )
                        ax.plot(
                            [cx - 0.2, cx + 0.2],
                            [cy - 0.2, cy + 0.2],
                            color="black",
                            linewidth=1.5,
                            solid_capstyle="round",
                        )
                        ax.plot(
                            [cx - 0.2, cx + 0.2],
                            [cy + 0.2, cy - 0.2],
                            color="black",
                            linewidth=1.5,
                            solid_capstyle="round",
                        )
                    elif pos == forklift:
                        # Draw forklift symbol
                        cab = plt.Rectangle(
                            (cx - 0.15, cy - 0.1),
                            0.25,
                            0.3,
                            facecolor="white",
                            edgecolor="black",
                            linewidth=2,
                        )
                        ax.add_patch(cab)
                        # Mast (vertical bar)
                        ax.plot([cx + 0.15, cx + 0.15], [cy - 0.15, cy + 0.25], color="black", linewidth=3)
                        # Forks (two horizontal prongs)
                        ax.plot([cx + 0.15, cx + 0.35], [cy - 0.05, cy - 0.05], color="black", linewidth=3)
                        ax.plot([cx + 0.15, cx + 0.35], [cy - 0.15, cy - 0.15], color="black", linewidth=3)
                        # Wheel
                        wheel = plt.Circle((cx - 0.05, cy - 0.18), 0.08, facecolor="black", edgecolor="black")
                        ax.add_patch(wheel)
                    elif pos == package:
                        # Draw package/box symbol
                        box = plt.Rectangle(
                            (cx - 0.2, cy - 0.15),
                            0.4,
                            0.3,
                            facecolor="white",
                            edgecolor="black",
                            linewidth=1.5,
                        )
                        ax.add_patch(box)
                        # Ribbon/tape
                        ax.plot([cx - 0.2, cx + 0.2], [cy, cy], color="#8B4513", linewidth=2)
                        ax.plot([cx, cx], [cy - 0.15, cy + 0.15], color="#8B4513", linewidth=2)

        # Draw robot
        robot_x, robot_y = robot["x"], robot["y"]
        plot_rx = robot_x - 1 + 0.5
        plot_ry = robot_y - 1 + 0.5

        robot_circle = plt.Circle((plot_rx, plot_ry), 0.3, facecolor=COLORS["robot"], edgecolor="white", linewidth=2)
        ax.add_patch(robot_circle)

        # Direction arrow
        dir_deltas = {
            "NORTH": (0, 0.2),
            "EAST": (0.2, 0),
            "SOUTH": (0, -0.2),
            "WEST": (-0.2, 0),
        }
        dx, dy = dir_deltas[robot["direction"]]
        ax.annotate(
            "",
            xy=(plot_rx + dx * 1.5, plot_ry + dy * 1.5),
            xytext=(plot_rx, plot_ry),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5),
        )

        # Add annotations for agent view
        if view_type == "agent":
            for (ax_pos, text) in annotations:
                plot_ax = ax_pos[0] - 1 + 0.5
                plot_ay = ax_pos[1] - 1 + 0.5
                ax.text(plot_ax, plot_ay, text, ha="center", va="center", fontsize=9, color="black", style="italic")

        # Axis setup
        ax.set_xlim(-0.1, width + 0.1)
        ax.set_ylim(-0.1, height + 0.1)
        ax.set_aspect("equal")
        ax.set_xticks([i + 0.5 for i in range(width)])
        ax.set_xticklabels([str(i + 1) for i in range(width)])
        ax.set_yticks([i + 0.5 for i in range(height)])
        ax.set_yticklabels([str(i + 1) for i in range(height)])
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax_agent.set_title("Agent's Knowledge", fontsize=12)
    ax_true.set_title("True State (Hidden)", fontsize=12)

    fig.suptitle(f"{title}\n{subtitle}", fontsize=13, y=1.02)

    # Add legend below
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor=COLORS["robot"], edgecolor="white", label="Robot"),
        Patch(facecolor=COLORS["safe"], edgecolor="gray", label="Known safe"),
        Patch(facecolor=COLORS["uncertain"], edgecolor="gray", label="Uncertain"),
        Patch(facecolor=COLORS["unknown"], edgecolor="gray", label="Unknown"),
        Patch(facecolor=COLORS["damaged"], edgecolor="gray", label="Damaged (X)"),
        Patch(facecolor=COLORS["forklift"], edgecolor="gray", label="Forklift (forklift)"),
        Patch(facecolor=COLORS["package"], edgecolor="gray", label="Package"),
        Line2D([0], [0], color=COLORS["creaking"], linewidth=2, label="Creaking"),
        Line2D([0], [0], color=COLORS["rumbling"], linewidth=2, label="Rumbling"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
        fancybox=True,
        fontsize=9,
    )

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {output_path}")


def configure_rn_example_layout(env):
    """Configure the Russell and Norvig-inspired layout for figures."""
    env.reset(seed=0)
    env._damaged = {(3, 1), (3, 3)}
    env._forklift = (1, 3)
    env._package = (2, 3)
    env._forklift_alive = True
    env._robot = env._robot.__class__(1, 1, env._robot.direction)
    env._steps = 0
    env._total_reward = 0.0
    env._terminated = False
    env._success = False
    env._history = []
    env._last_percept = env._get_percept(bump=False, beep=False)
    env._record_state()


# -----------------------------------------------------------------------------
# Animation
# -----------------------------------------------------------------------------

def replay_episode(
    history: list[dict],
    env: "HazardousWarehouseEnv",
    interval_ms: int = 500,
    reveal: bool = True,
):
    """
    Animate an episode from recorded history.

    Parameters:
        history: List of state dictionaries from env.history
        env: The environment (for true state info)
        interval_ms: Milliseconds between frames
        reveal: Show hidden hazards during replay
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError:
        print("matplotlib not available")
        return

    if not history:
        print("No history to replay")
        return

    fig, (ax_grid, ax_info) = plt.subplots(1, 2, figsize=(12, 6),
                                            gridspec_kw={"width_ratios": [1.2, 1]})

    # Get true state for dimensions
    true_state = env.get_true_state()
    width, height = true_state["width"], true_state["height"]

    # Initial frame
    def get_frame_grid(frame_idx: int) -> list[list[tuple[float, float, float]]]:
        """Generate grid for a specific frame."""
        state = history[frame_idx]

        # Build grid showing true hazards
        damaged = set(tuple(p) for p in true_state["damaged"])
        forklift = tuple(true_state["forklift"]) if true_state["forklift"] else None
        package = tuple(true_state["package"]) if true_state["package"] else None

        grid = []
        for row in range(height, 0, -1):
            grid_row = []
            for col in range(1, width + 1):
                pos = (col, row)
                robot_pos = (state["robot_x"], state["robot_y"])

                if pos == robot_pos:
                    if not state["alive"]:
                        color = COLORS["robot_dead"]
                    elif state["has_package"]:
                        color = COLORS["robot_loaded"]
                    else:
                        color = COLORS["robot"]
                elif reveal:
                    if pos in damaged:
                        color = COLORS["damaged"]
                    elif pos == forklift:
                        color = COLORS["forklift"] if state["forklift_alive"] else COLORS["forklift_dead"]
                    elif pos == package and not state["has_package"]:
                        color = COLORS["package"]
                    elif pos == (1, 1):
                        color = COLORS["exit"]
                    else:
                        color = COLORS["empty"]
                else:
                    color = COLORS["unknown"]

                grid_row.append(color)
            grid.append(grid_row)
        return grid

    im = ax_grid.imshow(get_frame_grid(0), interpolation="nearest", aspect="equal")

    # Grid lines
    for i in range(width + 1):
        ax_grid.axvline(i - 0.5, color="gray", linewidth=0.5)
    for i in range(height + 1):
        ax_grid.axhline(i - 0.5, color="gray", linewidth=0.5)

    ax_grid.set_xticks(range(width))
    ax_grid.set_xticklabels(range(1, width + 1))
    ax_grid.set_yticks(range(height))
    ax_grid.set_yticklabels(range(height, 0, -1))

    title = ax_grid.set_title("Step 0")

    # Info panel
    ax_info.set_axis_off()
    info_text = ax_info.text(0.1, 0.9, "", transform=ax_info.transAxes,
                             fontsize=11, verticalalignment="top",
                             family="monospace")

    def update(frame_idx: int):
        state = history[frame_idx]
        im.set_data(get_frame_grid(frame_idx))

        # Update title
        action_str = state["action"] or "START"
        title.set_text(f"Step {state['step']} | Action: {action_str}")

        # Update info text
        percept = state["percept"]
        percept_parts = []
        if percept["creaking"]:
            percept_parts.append("Creaking")
        if percept["rumbling"]:
            percept_parts.append("Rumbling")
        if percept["beacon"]:
            percept_parts.append("Beacon")
        if percept["bump"]:
            percept_parts.append("Bump")
        if percept["beep"]:
            percept_parts.append("Beep")

        info_lines = [
            f"Position: ({state['robot_x']}, {state['robot_y']})",
            f"Facing: {state['direction']}",
            f"",
            f"Percepts:",
            f"  {', '.join(percept_parts) if percept_parts else 'None'}",
            f"",
            f"Has package: {state['has_package']}",
            f"Has shutdown: {state['has_shutdown']}",
            f"Alive: {state['alive']}",
            f"Forklift alive: {state['forklift_alive']}",
            f"",
            f"Total reward: {state['total_reward']:.1f}",
        ]
        info_text.set_text("\n".join(info_lines))

        return [im, title, info_text]

    anim = animation.FuncAnimation(
        fig, update, frames=len(history), interval=interval_ms, blit=False
    )

    # Keyboard controls
    paused = {"value": False}
    current = {"index": 0}

    def on_key(event):
        if event.key == " ":
            if paused["value"]:
                anim.event_source.start()
            else:
                anim.event_source.stop()
            paused["value"] = not paused["value"]
        elif event.key in ("left", "right"):
            if not paused["value"]:
                anim.event_source.stop()
                paused["value"] = True
            delta = -1 if event.key == "left" else 1
            current["index"] = max(0, min(len(history) - 1, current["index"] + delta))
            update(current["index"])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    plt.show()
    return anim


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def save_frames(
    history: list[dict],
    env: "HazardousWarehouseEnv",
    output_dir: str,
    reveal: bool = True,
    dpi: int = 100,
):
    """Save each frame of an episode as a PNG file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    true_state = env.get_true_state()
    width, height = true_state["width"], true_state["height"]
    damaged = set(tuple(p) for p in true_state["damaged"])
    forklift = tuple(true_state["forklift"]) if true_state["forklift"] else None
    package = tuple(true_state["package"]) if true_state["package"] else None

    for i, state in enumerate(history):
        fig, ax = plt.subplots(figsize=(6, 6))

        # Build grid
        grid = []
        for row in range(height, 0, -1):
            grid_row = []
            for col in range(1, width + 1):
                pos = (col, row)
                robot_pos = (state["robot_x"], state["robot_y"])

                if pos == robot_pos:
                    if not state["alive"]:
                        color = COLORS["robot_dead"]
                    elif state["has_package"]:
                        color = COLORS["robot_loaded"]
                    else:
                        color = COLORS["robot"]
                elif reveal:
                    if pos in damaged:
                        color = COLORS["damaged"]
                    elif pos == forklift:
                        color = COLORS["forklift"] if state["forklift_alive"] else COLORS["forklift_dead"]
                    elif pos == package and not state["has_package"]:
                        color = COLORS["package"]
                    elif pos == (1, 1):
                        color = COLORS["exit"]
                    else:
                        color = COLORS["empty"]
                else:
                    color = COLORS["unknown"]

                grid_row.append(color)
            grid.append(grid_row)

        ax.imshow(grid, interpolation="nearest", aspect="equal")

        for j in range(width + 1):
            ax.axvline(j - 0.5, color="gray", linewidth=0.5)
        for j in range(height + 1):
            ax.axhline(j - 0.5, color="gray", linewidth=0.5)

        ax.set_xticks(range(width))
        ax.set_xticklabels(range(1, width + 1))
        ax.set_yticks(range(height))
        ax.set_yticklabels(range(height, 0, -1))

        action_str = state["action"] or "START"
        ax.set_title(f"Step {state['step']} | {action_str}")

        fig.savefig(Path(output_dir) / f"frame_{i:04d}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from hazardous_warehouse_env import HazardousWarehouseEnv, Action

    # Create environment
    env = HazardousWarehouseEnv(seed=42)

    # Run a simple episode
    actions = [
        Action.FORWARD, Action.FORWARD, Action.TURN_LEFT,
        Action.FORWARD, Action.TURN_RIGHT, Action.FORWARD,
    ]

    for action in actions:
        percept, reward, done, info = env.step(action)
        if done:
            break

    # Plot final state
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Agent's view (unknown squares)
        plot_state(env, ax=axes[0], reveal=False, title="Agent's View")

        # True state
        plot_state(env, ax=axes[1], reveal=True, title="True State (Revealed)")

        # Legend
        plot_legend(ax=axes[2])

        plt.tight_layout()
        plt.show()

        # Replay animation
        print("\nReplaying episode (press SPACE to pause, arrows to step)...")
        replay_episode(env.history, env, interval_ms=400)

    except ImportError:
        print("matplotlib not available - skipping visualization demo")
