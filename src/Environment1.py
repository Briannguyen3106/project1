from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .Qlearning import QLearning
from .Sarsa import Sarsa
from .common import TrainConfig, greedy_actions, plot_learning_curves, plot_training_summary, run_many, save_experiment


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures" / "environment1_cliff_walking"


def make_env():
    return gym.make("CliffWalking-v1")


def cliff_event(reward: float, _info: dict) -> int:
    return int(reward <= -100.0)


def success(total_reward: float, terminated: bool) -> bool:
    return bool(terminated)


def _agent_factory(cls):
    return lambda states, actions, alpha, gamma, seed: cls(states, actions, alpha, gamma, seed)


def plot_policy_paths(q_tables: dict[str, np.ndarray], output_path: Path) -> None:
    rows, cols = 4, 12
    start, goal = 36, 47
    cliff = set(range(37, 47))
    action_delta = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    fig, axes = plt.subplots(1, len(q_tables), figsize=(12, 3.4), squeeze=False)
    for axis, (name, q_table) in zip(axes[0], q_tables.items()):
        axis.set_xlim(-0.5, cols - 0.5)
        axis.set_ylim(rows - 0.5, -0.5)
        axis.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        axis.grid(which="minor", color="0.75")
        axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for state in cliff:
            row, col = divmod(state, cols)
            axis.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color="#d95f5f"))
        path = [start]
        state = start
        seen = {start}
        for _ in range(100):
            if state == goal:
                break
            action = int(greedy_actions(q_table[state])[0])
            row, col = divmod(state, cols)
            dr, dc = action_delta[action]
            next_row = min(max(row + dr, 0), rows - 1)
            next_col = min(max(col + dc, 0), cols - 1)
            next_state = next_row * cols + next_col
            if next_state in cliff:
                next_state = start
            path.append(next_state)
            if next_state in seen and next_state != goal:
                break
            seen.add(next_state)
            state = next_state
        coords = np.array([divmod(item, cols) for item in path])
        axis.plot(coords[:, 1], coords[:, 0], marker="o", color="#1f5a85", linewidth=2)
        axis.text(0, 3, "S", ha="center", va="center", color="white", weight="bold")
        axis.text(11, 3, "G", ha="center", va="center", color="black", weight="bold")
        axis.set_title(name)
    fig.suptitle("Greedy policy paths (representative seed)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(seeds: list[int] | None = None, episodes: int = 500):
    seeds = list(range(30)) if seeds is None else seeds
    config = TrainConfig(
        episodes=episodes,
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay_fraction=1.0,
        max_steps=500,
        eval_interval=max(10, episodes // 25),
        eval_episodes=30,
    )
    all_episodes, all_evaluations = [], []
    q_tables_by_algorithm = {}
    for name, agent in (("SARSA", Sarsa), ("Q-Learning", QLearning)):
        episode_df, evaluation_df, q_tables = run_many(
            name, _agent_factory(agent), make_env, config, seeds, success, cliff_event
        )
        all_episodes.append(episode_df)
        all_evaluations.append(evaluation_df)
        q_tables_by_algorithm[name] = q_tables
    episodes_df = pd.concat(all_episodes, ignore_index=True)
    evaluations_df = pd.concat(all_evaluations, ignore_index=True)
    save_experiment(OUTPUT_DIR, "cliff_walking", config, episodes_df, evaluations_df)
    plot_training_summary(episodes_df, OUTPUT_DIR / "training_summary.png", "Cliff Walking")
    plot_learning_curves(evaluations_df, OUTPUT_DIR / "evaluation_curves.png", "Cliff Walking")
    representative = seeds[0]
    plot_policy_paths(
        {name: tables[representative] for name, tables in q_tables_by_algorithm.items()},
        OUTPUT_DIR / "policy_paths.png",
    )
    return episodes_df, evaluations_df, q_tables_by_algorithm


if __name__ == "__main__":
    run_experiment()
