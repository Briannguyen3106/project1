from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ExpectedSarsa import ExpectedSarsa
from .Sarsa import Sarsa
from .common import TrainConfig, plot_learning_curves, plot_training_summary, run_many, save_experiment


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures" / "environment2_frozen_lake"


def make_env(slippery: bool):
    return gym.make("FrozenLake-v1", is_slippery=slippery)


def success(total_reward: float, terminated: bool) -> bool:
    return bool(terminated and total_reward > 0.0)


def _agent_factory(cls):
    return lambda states, actions, alpha, gamma, seed: cls(states, actions, alpha, gamma, seed)


def plot_target_variance(episode_df: pd.DataFrame, output_path: Path) -> None:
    data = episode_df.copy()
    data["phase"] = pd.cut(data["episode"], bins=10, labels=False, include_lowest=True)
    summary = data.groupby(["setting", "algorithm", "phase"], as_index=False).agg(
        target_variance=("td_target_variance", "mean"),
        error_variance=("td_error_variance", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for (setting, algorithm), group in summary.groupby(["setting", "algorithm"]):
        label = f"{algorithm} ({setting})"
        axes[0].plot(group["phase"] + 1, group["target_variance"], marker="o", label=label)
        axes[1].plot(group["phase"] + 1, group["error_variance"], marker="o", label=label)
    axes[0].set(title="Mean TD-target variance", xlabel="Training phase", ylabel="Variance")
    axes[1].set(title="Mean TD-error variance", xlabel="Training phase", ylabel="Variance")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Frozen Lake update variance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_q_heatmaps(q_tables: dict, seed: int, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for axis, ((setting, algorithm), tables) in zip(axes.flat, q_tables.items()):
        values = np.max(tables[seed], axis=1).reshape(4, 4)
        image = axis.imshow(values, cmap="viridis")
        for row in range(4):
            for col in range(4):
                axis.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", color="white")
        axis.set_title(f"{algorithm} - {setting}")
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle(f"Q-value heatmaps (seed {seed})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(seeds: list[int] | None = None, episodes: int = 5000):
    seeds = list(range(30)) if seeds is None else seeds
    config = TrainConfig(
        episodes=episodes,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_fraction=0.8,
        max_steps=200,
        eval_interval=max(50, episodes // 25),
        eval_episodes=100,
    )
    all_episodes, all_evaluations = [], []
    q_tables = {}
    for slippery, setting in ((False, "deterministic"), (True, "slippery")):
        env_factory = lambda slippery=slippery: make_env(slippery)
        for name, agent in (("SARSA", Sarsa), ("Expected SARSA", ExpectedSarsa)):
            episode_df, evaluation_df, tables = run_many(
                name, _agent_factory(agent), env_factory, config, seeds, success
            )
            episode_df["setting"] = setting
            evaluation_df["setting"] = setting
            all_episodes.append(episode_df)
            all_evaluations.append(evaluation_df)
            q_tables[(setting, name)] = tables
    episodes_df = pd.concat(all_episodes, ignore_index=True)
    evaluations_df = pd.concat(all_evaluations, ignore_index=True)
    save_experiment(OUTPUT_DIR, "frozen_lake", config, episodes_df, evaluations_df)
    for setting in ("deterministic", "slippery"):
        plot_training_summary(
            episodes_df[episodes_df["setting"] == setting],
            OUTPUT_DIR / f"training_{setting}.png",
            f"Frozen Lake - {setting}",
        )
        plot_learning_curves(
            evaluations_df[evaluations_df["setting"] == setting],
            OUTPUT_DIR / f"evaluation_{setting}.png",
            f"Frozen Lake - {setting}",
        )
    plot_target_variance(episodes_df, OUTPUT_DIR / "td_variance.png")
    plot_q_heatmaps(q_tables, seeds[0], OUTPUT_DIR / "q_heatmaps.png")
    return episodes_df, evaluations_df, q_tables


if __name__ == "__main__":
    run_experiment()
