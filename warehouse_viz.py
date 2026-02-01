from pathlib import Path
def _grid_to_rgb(grid: list[list[str]]) -> list[list[tuple[float, float, float]]]:
    colors = {
        "#": (0.1, 0.1, 0.1),
        ".": (0.95, 0.95, 0.95),
        "P": (0.2, 0.6, 1.0),
        "D": (0.2, 0.8, 0.3),
        "R": (0.9, 0.2, 0.2),
        "r": (0.95, 0.6, 0.2),
    }
    return [[colors.get(ch, (0.8, 0.8, 0.8)) for ch in row] for row in grid]
def _legend_handles():
    from matplotlib.patches import Patch
    colors = {
        "wall": (0.1, 0.1, 0.1),
        "empty": (0.95, 0.95, 0.95),
        "pickup": (0.2, 0.6, 1.0),
        "dropoff": (0.2, 0.8, 0.3),
        "robot (empty)": (0.9, 0.2, 0.2),
        "robot (loaded)": (0.95, 0.6, 0.2),
    }
    return [
        Patch(facecolor=colors["wall"], edgecolor="none", label="Wall"),
        Patch(facecolor=colors["empty"], edgecolor="none", label="Empty"),
        Patch(facecolor=colors["pickup"], edgecolor="none", label="Pickup"),
        Patch(facecolor=colors["dropoff"], edgecolor="none", label="Dropoff"),
        Patch(facecolor=colors["robot (empty)"], edgecolor="none", label="Robot (empty)"),
        Patch(facecolor=colors["robot (loaded)"], edgecolor="none", label="Robot (loaded)"),
    ]
def save_frames_to_svg(
    frames: list[list[list[str]]], output_dir: str, dpi: int = 120
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping SVG export.")
        return
    if not frames:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.imshow(_grid_to_rgb(frame), interpolation="nearest")
        fig.savefig(Path(output_dir) / f"frame_{i:04d}.svg", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.legend(handles=_legend_handles(), loc="center", frameon=False, fontsize=9)
    fig.savefig(Path(output_dir) / "legend.svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
def replay_animation(
    frames: list[list[list[str]]],
    metrics: dict | None = None,
    interval_ms: int = 150,
    speed: float = 1.0,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError:
        print("matplotlib not available; skipping animation replay.")
        return
    if not frames:
        return
    fig = plt.figure(figsize=(9.5, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3])
    gs_left = gs[0, 0].subgridspec(2, 1, height_ratios=[0.12, 1.0], hspace=0.0)
    gs_battery = gs_left[0, 0].subgridspec(1, 3, width_ratios=[0.35, 0.15, 0.5])
    ax_battery = fig.add_subplot(gs_battery[0, 1])
    ax_grid = fig.add_subplot(gs_left[1, 0])
    gs_metrics = gs[0, 1].subgridspec(2, 1, hspace=0.35)
    ax_top = fig.add_subplot(gs_metrics[0, 0])
    ax_bottom = fig.add_subplot(gs_metrics[1, 0])
    ax_grid.set_axis_off()
    im = ax_grid.imshow(_grid_to_rgb(frames[0]), interpolation="nearest")
    step_text = ax_grid.text(
        0.02,
        0.98,
        "Step 0",
        transform=ax_grid.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
    )
    ax_grid.legend(
        handles=_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(0.0, -0.05),
        frameon=False,
        fontsize=8,
        ncol=2,
    )
    lines = {}
    battery_bar = None
    if metrics:
        x = list(range(len(frames)))
        ax_top.set_xlim(0, 1)
        ax_top.set_xlabel("Step")
        ax_top.set_ylabel("Reward")
        ax_top.grid(True, alpha=0.3)
        if "rewards" in metrics:
            (line_reward,) = ax_top.plot(x[:1], metrics["rewards"][:1], label="Reward")
            lines["rewards"] = line_reward
        if "battery" in metrics:
            ax_battery.set_title("Battery", fontsize=8)
            ax_battery.set_xlim(0, 1)
            ax_battery.set_xticks([])
            max_battery = max(metrics["battery"]) if metrics["battery"] else 1
            ax_battery.set_ylim(0, max_battery)
            ax_battery.set_yticks([0, max_battery])
            ax_battery.grid(True, axis="y", alpha=0.3)
            battery_bar = ax_battery.bar([0.5], [metrics["battery"][0]], width=0.6)[0]
        ax_bottom.set_xlim(0, 1)
        ax_bottom.set_xlabel("Step")
        ax_bottom.set_ylabel("Distance")
        ax_bottom.grid(True, alpha=0.3)
        if "dist_pickup" in metrics:
            (line_pickup,) = ax_bottom.plot(
                x[:1], metrics["dist_pickup"][:1], label="Dist to Pickup"
            )
            lines["dist_pickup"] = line_pickup
        if "dist_dropoff" in metrics:
            (line_dropoff,) = ax_bottom.plot(
                x[:1], metrics["dist_dropoff"][:1], label="Dist to Dropoff"
            )
            lines["dist_dropoff"] = line_dropoff
        ax_top.legend(loc="upper left", fontsize=8)
        ax_bottom.legend(loc="upper right", fontsize=8)
    current = {"index": 0}
    def update(i: int):
        im.set_data(_grid_to_rgb(frames[i]))
        step_text.set_text(f"Step {i}")
        current["index"] = i
        if metrics:
            for key, line in lines.items():
                line.set_data(list(range(i + 1)), metrics[key][: i + 1])
            ax_top.set_xlim(0, max(1, i))
            ax_bottom.set_xlim(0, max(1, i))
            if "rewards" in lines:
                y = metrics["rewards"][: i + 1]
                ax_top.set_ylim(min(y) - 1, max(y) + 1)
            if "dist_pickup" in lines or "dist_dropoff" in lines:
                y1 = metrics.get("dist_pickup", [])[: i + 1]
                y2 = metrics.get("dist_dropoff", [])[: i + 1]
                ys = [v for v in y1 + y2 if y1 or y2]
                if ys:
                    ax_bottom.set_ylim(min(ys) - 1, max(ys) + 1)
            if battery_bar is not None:
                battery_bar.set_height(metrics["battery"][i])
        return [im, step_text] + list(lines.values()) + ([battery_bar] if battery_bar else [])
    effective_interval = max(10, int(interval_ms / max(speed, 0.1)))
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=effective_interval, blit=False
    )
    paused = {"value": False}
    def on_key(event):
        if event.key == " ":
            if paused["value"]:
                anim.event_source.start()
            else:
                anim.event_source.stop()
            paused["value"] = not paused["value"]
        elif event.key in {"left", "right"}:
            if not paused["value"]:
                anim.event_source.stop()
                paused["value"] = True
            delta = -1 if event.key == "left" else 1
            current["index"] = max(0, min(len(frames) - 1, current["index"] + delta))
            update(current["index"])
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return anim