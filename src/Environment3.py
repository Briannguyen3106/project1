from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ExpectedSarsa import ExpectedSarsa
from .Qlearning import QLearning
from .common import TrainConfig, plot_learning_curves, plot_training_summary, run_many, save_experiment


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures" / "environment3_taxi"


def make_env():
    return gym.make("Taxi-v3")


def illegal_action(reward: float, _info: dict) -> int:
    return int(reward <= -10.0)


def success(total_reward: float, terminated: bool) -> bool:
    return bool(terminated)


def _agent_factory(cls):
    return lambda states, actions, alpha, gamma, seed: cls(states, actions, alpha, gamma, seed)


def plot_first_success(episode_df: pd.DataFrame, output_path: Path) -> None:
    values = (
        episode_df.groupby(["algorithm", "seed"], as_index=False)["first_success_step"]
        .max()
    )
    values = values[values["first_success_step"] >= 0]
    algorithms = list(values["algorithm"].unique())
    series = [values.loc[values["algorithm"] == name, "first_success_step"].to_numpy() for name in algorithms]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot(series, tick_labels=algorithms, showmeans=True)
    axis.set(title="Environment steps to first success", ylabel="Steps")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(seeds: list[int] | None = None, episodes: int = 10000):
    seeds = list(range(30)) if seeds is None else seeds
    config = TrainConfig(
        episodes=episodes,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_fraction=0.8,
        max_steps=500,
        eval_interval=max(100, episodes // 25),
        eval_episodes=50,
    )
    all_episodes, all_evaluations = [], []
    q_tables_by_algorithm = {}
    for name, agent in (("Q-Learning", QLearning), ("Expected SARSA", ExpectedSarsa)):
        episode_df, evaluation_df, q_tables = run_many(
            name, _agent_factory(agent), make_env, config, seeds, success, illegal_action
        )
        all_episodes.append(episode_df)
        all_evaluations.append(evaluation_df)
        q_tables_by_algorithm[name] = q_tables
    episodes_df = pd.concat(all_episodes, ignore_index=True)
    evaluations_df = pd.concat(all_evaluations, ignore_index=True)
    save_experiment(OUTPUT_DIR, "taxi", config, episodes_df, evaluations_df)
    plot_training_summary(episodes_df, OUTPUT_DIR / "training_summary.png", "Taxi-v3")
    plot_learning_curves(evaluations_df, OUTPUT_DIR / "evaluation_curves.png", "Taxi-v3")
    plot_first_success(episodes_df, OUTPUT_DIR / "first_success.png")
    return episodes_df, evaluations_df, q_tables_by_algorithm


if __name__ == "__main__":
    run_experiment()
