from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ExpectedSarsa import ExpectedSarsa
from .Qlearning import QLearning
from .common import greedy_actions


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures" / "environment3_taxi"


@dataclass(frozen=True)
class TaxiConfig:
    step_budget: int = 500_000
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_fraction: float = 0.8
    max_episode_steps: int = 200
    eval_interval_steps: int = 20_000
    eval_episodes: int = 200
    return_threshold: float = 5.0
    threshold_patience: int = 3

    def epsilon(self, cumulative_steps: int) -> float:
        decay_steps = max(1, int(self.step_budget * self.epsilon_decay_fraction))
        progress = min(cumulative_steps / decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)


def make_env():
    return gym.make("Taxi-v3", max_episode_steps=200)


def _agent_factory(cls):
    return lambda states, actions, alpha, gamma, seed: cls(states, actions, alpha, gamma, seed)


def evaluate_greedy(
    q_table: np.ndarray,
    episodes: int,
    seed: int,
    max_steps: int,
) -> dict[str, float]:
    """Evaluate on a fixed seed panel shared by algorithms and checkpoints."""
    env = make_env()
    tie_rng = np.random.default_rng(seed + 10_000_000)
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    illegal_actions = 0
    try:
        for index in range(episodes):
            state, _ = env.reset(seed=seed + index)
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated) and steps < max_steps:
                best = greedy_actions(q_table[int(state)])
                action = int(tie_rng.choice(best))
                state, reward, terminated, truncated, _ = env.step(action)
                reward = float(reward)
                total_reward += reward
                illegal_actions += int(reward <= -10.0)
                steps += 1
            returns.append(total_reward)
            lengths.append(steps)
            successes += int(terminated)
    finally:
        env.close()

    return {
        "eval_return": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns, ddof=1)),
        "eval_success_rate": successes / episodes,
        "eval_episode_length": float(np.mean(lengths)),
        "eval_illegal_actions": illegal_actions / episodes,
    }


def train_seed(
    algorithm: str,
    agent_factory: Callable,
    config: TaxiConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    env = make_env()
    state_count = int(env.observation_space.n)
    action_count = int(env.action_space.n)
    agent = agent_factory(state_count, action_count, config.alpha, config.gamma, seed)
    episode_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    cumulative_steps = 0
    episode = 0
    first_success_step: int | None = None
    next_evaluation_step = 0

    def record_evaluation() -> None:
        metrics = evaluate_greedy(
            agent.q_table,
            config.eval_episodes,
            seed=seed * 1_000_000,
            max_steps=config.max_episode_steps,
        )
        evaluation_rows.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "episode": episode,
                "cumulative_steps": cumulative_steps,
                **metrics,
            }
        )

    try:
        record_evaluation()
        next_evaluation_step = config.eval_interval_steps
        while cumulative_steps < config.step_budget:
            episode += 1
            state, _ = env.reset(seed=seed * 100_000 + episode)
            state = int(state)
            epsilon = config.epsilon(cumulative_steps)
            action = agent.select_action(state, epsilon)
            total_reward = 0.0
            illegal_actions = 0
            steps = 0
            terminated = truncated = False

            while (
                not (terminated or truncated)
                and steps < config.max_episode_steps
                and cumulative_steps < config.step_budget
            ):
                epsilon = config.epsilon(cumulative_steps)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = int(next_state)
                reward = float(reward)
                environment_done = bool(terminated or truncated)
                next_action = None if environment_done else agent.select_action(next_state, epsilon)
                agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    environment_done,
                    next_action,
                    epsilon,
                )
                total_reward += reward
                illegal_actions += int(reward <= -10.0)
                steps += 1
                cumulative_steps += 1
                state = next_state
                if next_action is not None:
                    action = next_action

                while cumulative_steps >= next_evaluation_step:
                    record_evaluation()
                    next_evaluation_step += config.eval_interval_steps

            success = bool(terminated)
            if success and first_success_step is None:
                first_success_step = cumulative_steps
            episode_rows.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "episode": episode,
                    "cumulative_steps": cumulative_steps,
                    "epsilon": config.epsilon(cumulative_steps),
                    "return": total_reward,
                    "episode_length": steps,
                    "success": int(success),
                    "illegal_actions": illegal_actions,
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "budget_cutoff": int(cumulative_steps >= config.step_budget and not environment_done),
                    "first_success_step": first_success_step if first_success_step is not None else -1,
                }
            )

        if evaluation_rows[-1]["cumulative_steps"] != cumulative_steps:
            record_evaluation()
    finally:
        env.close()

    return pd.DataFrame(episode_rows), pd.DataFrame(evaluation_rows), agent.q_table.copy()


def add_seed_metrics(evaluation_df: pd.DataFrame, config: TaxiConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for (algorithm, seed), group in evaluation_df.groupby(["algorithm", "seed"]):
        group = group.sort_values("cumulative_steps").reset_index(drop=True)
        x = group["cumulative_steps"].to_numpy(dtype=float)
        y = group["eval_return"].to_numpy(dtype=float)
        normalized_auc = float(np.trapezoid(y, x) / config.step_budget)
        reached = y >= config.return_threshold
        threshold_step = np.nan
        for index in range(0, len(group) - config.threshold_patience + 1):
            if reached[index:index + config.threshold_patience].all():
                threshold_step = float(group.loc[index, "cumulative_steps"])
                break
        rows.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "steps_to_threshold": threshold_step,
                "reached_threshold": int(not np.isnan(threshold_step)),
                "normalized_return_auc": normalized_auc,
            }
        )
    return pd.DataFrame(rows)


def _step_bin_summary(data: pd.DataFrame, columns: list[str], bins: int = 25) -> pd.DataFrame:
    result = data.copy()
    maximum = int(result["cumulative_steps"].max())
    result["step_bin"] = pd.cut(
        result["cumulative_steps"],
        np.linspace(0, maximum, bins + 1),
        labels=False,
        include_lowest=True,
    )
    aggregations = {column: (column, "mean") for column in columns}
    return result.groupby(["algorithm", "step_bin"], as_index=False).agg(
        steps=("cumulative_steps", "mean"),
        **aggregations,
    )


def plot_training(episode_df: pd.DataFrame, output_path: Path) -> None:
    summary = _step_bin_summary(episode_df, ["return", "success", "illegal_actions"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    columns = [
        ("return", "Online return", "Return"),
        ("success", "Online success rate", "Success rate"),
        ("illegal_actions", "Illegal actions per episode", "Illegal actions"),
    ]
    for axis, (column, title, ylabel) in zip(axes, columns):
        for algorithm, group in summary.groupby("algorithm"):
            axis.plot(group["steps"], group[column], marker="o", label=algorithm)
        axis.set(title=title, xlabel="Environment steps", ylabel=ylabel)
        axis.grid(alpha=0.25)
    axes[-1].legend()
    fig.suptitle("Taxi-v3 training behavior")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_evaluation(evaluation_df: pd.DataFrame, output_path: Path) -> None:
    summary = evaluation_df.groupby(["algorithm", "cumulative_steps"], as_index=False).agg(
        return_mean=("eval_return", "mean"),
        return_std=("eval_return", "std"),
        success_mean=("eval_success_rate", "mean"),
        success_std=("eval_success_rate", "std"),
        illegal_mean=("eval_illegal_actions", "mean"),
        count=("seed", "nunique"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    specifications = [
        ("return_mean", "return_std", "Greedy evaluation return", "Return"),
        ("success_mean", "success_std", "Greedy success rate", "Success rate"),
        ("illegal_mean", None, "Greedy illegal actions", "Illegal actions/episode"),
    ]
    for axis, (mean_column, std_column, title, ylabel) in zip(axes, specifications):
        for algorithm, group in summary.groupby("algorithm"):
            x = group["cumulative_steps"].to_numpy()
            mean = group[mean_column].to_numpy()
            axis.plot(x, mean, marker="o", label=algorithm)
            if std_column is not None:
                ci = 1.96 * group[std_column].fillna(0).to_numpy() / np.sqrt(group["count"])
                axis.fill_between(x, mean - ci, mean + ci, alpha=0.18)
        axis.set(title=title, xlabel="Environment steps", ylabel=ylabel)
        axis.grid(alpha=0.25)
    axes[-1].legend()
    fig.suptitle("Taxi-v3 greedy evaluation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sample_efficiency(seed_metrics: pd.DataFrame, output_path: Path) -> None:
    algorithms = list(seed_metrics["algorithm"].unique())
    threshold_series = [
        seed_metrics.loc[
            (seed_metrics["algorithm"] == algorithm) & seed_metrics["steps_to_threshold"].notna(),
            "steps_to_threshold",
        ].to_numpy()
        for algorithm in algorithms
    ]
    auc_series = [
        seed_metrics.loc[seed_metrics["algorithm"] == algorithm, "normalized_return_auc"].to_numpy()
        for algorithm in algorithms
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].boxplot(threshold_series, tick_labels=algorithms, showmeans=True)
    axes[0].set(title="Steps to sustained return threshold", ylabel="Environment steps")
    axes[1].boxplot(auc_series, tick_labels=algorithms, showmeans=True)
    axes[1].set(title="Normalized evaluation-return AUC", ylabel="AUC")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    seeds: list[int] | None = None,
    step_budget: int = 500_000,
):
    seeds = list(range(30)) if seeds is None else seeds
    config = TaxiConfig(
        step_budget=step_budget,
        eval_interval_steps=max(5_000, step_budget // 25),
    )
    all_episodes, all_evaluations = [], []
    q_tables_by_algorithm = {}
    for name, agent in (("Q-Learning", QLearning), ("Expected SARSA", ExpectedSarsa)):
        episode_parts, evaluation_parts, q_tables = [], [], {}
        for seed in seeds:
            episode_df, evaluation_df, q_table = train_seed(
                name,
                _agent_factory(agent),
                config,
                seed,
            )
            episode_parts.append(episode_df)
            evaluation_parts.append(evaluation_df)
            q_tables[seed] = q_table
        all_episodes.append(pd.concat(episode_parts, ignore_index=True))
        all_evaluations.append(pd.concat(evaluation_parts, ignore_index=True))
        q_tables_by_algorithm[name] = q_tables

    episodes_df = pd.concat(all_episodes, ignore_index=True)
    evaluations_df = pd.concat(all_evaluations, ignore_index=True)
    seed_metrics = add_seed_metrics(evaluations_df, config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    episodes_df.to_csv(OUTPUT_DIR / "taxi_episodes.csv", index=False)
    evaluations_df.to_csv(OUTPUT_DIR / "taxi_evaluations.csv", index=False)
    seed_metrics.to_csv(OUTPUT_DIR / "taxi_seed_metrics.csv", index=False)
    pd.Series(asdict(config)).to_json(OUTPUT_DIR / "taxi_config.json", indent=2)
    plot_training(episodes_df, OUTPUT_DIR / "training_summary.png")
    plot_evaluation(evaluations_df, OUTPUT_DIR / "evaluation_curves.png")
    plot_sample_efficiency(seed_metrics, OUTPUT_DIR / "sample_efficiency.png")
    return episodes_df, evaluations_df, seed_metrics, q_tables_by_algorithm


if __name__ == "__main__":
    run_experiment()
