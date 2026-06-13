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

from .Qlearning import QLearning
from .Sarsa import Sarsa
from .common import greedy_actions


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures" / "environment4_cliff_sensitivity"
ALPHA_VALUES = (0.01, 0.1, 0.3, 0.5, 0.8)
SCHEDULES = ("fixed", "decay")


@dataclass(frozen=True)
class SensitivityConfig:
    step_budget: int = 30_000
    gamma: float = 0.9
    max_episode_steps: int = 500
    eval_interval_steps: int = 1_200
    epsilon_fixed: float = 0.1
    epsilon_decay_start: float = 0.1
    epsilon_decay_end: float = 0.01
    epsilon_decay_fraction: float = 0.8
    return_threshold: float = -20.0
    threshold_patience: int = 3

    def epsilon(self, cumulative_steps: int, schedule: str) -> float:
        if schedule == "fixed":
            return self.epsilon_fixed
        if schedule != "decay":
            raise ValueError(f"Unknown epsilon schedule: {schedule}")
        decay_steps = max(1, int(self.step_budget * self.epsilon_decay_fraction))
        progress = min(cumulative_steps / decay_steps, 1.0)
        return self.epsilon_decay_start + progress * (
            self.epsilon_decay_end - self.epsilon_decay_start
        )


def make_env():
    return gym.make("CliffWalking-v1", max_episode_steps=500)


def _agent_factory(cls):
    return lambda states, actions, alpha, gamma, seed: cls(states, actions, alpha, gamma, seed)


def exact_greedy_evaluation(
    env: gym.Env,
    q_table: np.ndarray,
    max_steps: int,
) -> dict[str, float]:
    """Evaluate the tie-randomized greedy policy exactly from the environment model."""
    model = env.unwrapped.P
    state_count, action_count = q_table.shape
    policy = np.zeros((state_count, action_count), dtype=float)
    for state in range(state_count):
        best = greedy_actions(q_table[state])
        policy[state, best] = 1.0 / len(best)

    continuation = np.zeros((state_count, state_count), dtype=float)
    expected_reward = np.zeros(state_count, dtype=float)
    success = np.zeros(state_count, dtype=float)
    expected_cliff_falls = np.zeros(state_count, dtype=float)
    for state in range(state_count):
        for action in range(action_count):
            action_probability = policy[state, action]
            if action_probability == 0.0:
                continue
            for probability, next_state, reward, terminated in model[state][action]:
                mass = action_probability * probability
                expected_reward[state] += mass * float(reward)
                expected_cliff_falls[state] += mass * int(float(reward) <= -100.0)
                if terminated:
                    success[state] += mass
                else:
                    continuation[state, int(next_state)] += mass

    distribution = np.zeros(state_count, dtype=float)
    distribution[36] = 1.0
    total_return = 0.0
    success_probability = 0.0
    expected_length = 0.0
    cliff_falls = 0.0
    for _ in range(max_steps):
        alive = float(distribution.sum())
        if alive <= 1e-15:
            break
        expected_length += alive
        total_return += float(np.dot(distribution, expected_reward))
        cliff_falls += float(np.dot(distribution, expected_cliff_falls))
        success_probability += float(np.dot(distribution, success))
        distribution = distribution @ continuation

    return {
        "eval_return": total_return,
        "eval_success_rate": success_probability,
        "eval_episode_length": expected_length,
        "eval_cliff_falls": cliff_falls,
        "eval_unfinished_probability": float(distribution.sum()),
    }


def train_seed(
    experiment: str,
    condition: str,
    algorithm: str,
    agent_factory: Callable,
    alpha: float,
    epsilon_schedule: str,
    config: SensitivityConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = make_env()
    state_count = int(env.observation_space.n)
    action_count = int(env.action_space.n)
    agent = agent_factory(state_count, action_count, alpha, config.gamma, seed)
    episode_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    cumulative_steps = 0
    episode = 0
    next_evaluation_step = 0

    def record_evaluation() -> None:
        metrics = exact_greedy_evaluation(env, agent.q_table, config.max_episode_steps)
        evaluation_rows.append(
            {
                "experiment": experiment,
                "condition": condition,
                "algorithm": algorithm,
                "seed": seed,
                "episode": episode,
                "cumulative_steps": cumulative_steps,
                "alpha": alpha,
                "epsilon_schedule": epsilon_schedule,
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
            epsilon = config.epsilon(cumulative_steps, epsilon_schedule)
            action = agent.select_action(state, epsilon)
            total_reward = 0.0
            cliff_falls = 0
            steps = 0
            terminated = truncated = False

            while (
                not (terminated or truncated)
                and steps < config.max_episode_steps
                and cumulative_steps < config.step_budget
            ):
                epsilon = config.epsilon(cumulative_steps, epsilon_schedule)
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
                cliff_falls += int(reward <= -100.0)
                steps += 1
                cumulative_steps += 1
                state = next_state
                if next_action is not None:
                    action = next_action

                while cumulative_steps >= next_evaluation_step:
                    record_evaluation()
                    next_evaluation_step += config.eval_interval_steps

            episode_rows.append(
                {
                    "experiment": experiment,
                    "condition": condition,
                    "algorithm": algorithm,
                    "seed": seed,
                    "episode": episode,
                    "cumulative_steps": cumulative_steps,
                    "alpha": alpha,
                    "epsilon_schedule": epsilon_schedule,
                    "epsilon": config.epsilon(cumulative_steps, epsilon_schedule),
                    "return": total_reward,
                    "episode_length": steps,
                    "success": int(terminated),
                    "cliff_falls": cliff_falls,
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "budget_cutoff": int(cumulative_steps >= config.step_budget and not environment_done),
                }
            )

        if evaluation_rows[-1]["cumulative_steps"] != cumulative_steps:
            record_evaluation()
    finally:
        env.close()

    return pd.DataFrame(episode_rows), pd.DataFrame(evaluation_rows)


def add_seed_metrics(evaluation_df: pd.DataFrame, config: SensitivityConfig) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["experiment", "condition", "algorithm", "seed"]
    for key, group in evaluation_df.groupby(keys):
        group = group.sort_values("cumulative_steps").reset_index(drop=True)
        x = group["cumulative_steps"].to_numpy(dtype=float)
        y = group["eval_return"].to_numpy(dtype=float)
        reached = y >= config.return_threshold
        threshold_step = np.nan
        for index in range(len(group) - config.threshold_patience + 1):
            if reached[index:index + config.threshold_patience].all():
                threshold_step = float(group.loc[index, "cumulative_steps"])
                break
        rows.append(
            {
                **dict(zip(keys, key)),
                "steps_to_threshold": threshold_step,
                "reached_threshold": int(not np.isnan(threshold_step)),
                "normalized_return_auc": float(np.trapezoid(y, x) / config.step_budget),
            }
        )
    return pd.DataFrame(rows)


def _step_summary(data: pd.DataFrame, columns: list[str], bins: int = 25) -> pd.DataFrame:
    result = data.copy()
    maximum = int(result["cumulative_steps"].max())
    result["step_bin"] = pd.cut(
        result["cumulative_steps"],
        np.linspace(0, maximum, bins + 1),
        labels=False,
        include_lowest=True,
    )
    aggregations = {column: (column, "mean") for column in columns}
    return result.groupby(
        ["experiment", "condition", "algorithm", "step_bin"], as_index=False
    ).agg(steps=("cumulative_steps", "mean"), **aggregations)


def plot_alpha_sweep(episode_df: pd.DataFrame, evaluation_df: pd.DataFrame, output_path: Path) -> None:
    training = _step_summary(
        episode_df[episode_df["experiment"] == "alpha"], ["return", "cliff_falls"]
    )
    evaluation = evaluation_df[evaluation_df["experiment"] == "alpha"].groupby(
        ["condition", "algorithm", "cumulative_steps"], as_index=False
    ).eval_return.mean()
    algorithms = list(episode_df["algorithm"].unique())
    fig, axes = plt.subplots(len(algorithms), 3, figsize=(16, 8), squeeze=False)
    for row, algorithm in enumerate(algorithms):
        for condition, group in training[training["algorithm"] == algorithm].groupby("condition"):
            axes[row, 0].plot(group["steps"], group["return"], label=condition)
            axes[row, 1].plot(group["steps"], group["cliff_falls"], label=condition)
        for condition, group in evaluation[evaluation["algorithm"] == algorithm].groupby("condition"):
            axes[row, 2].plot(group["cumulative_steps"], group["eval_return"], label=condition)
        axes[row, 0].set_ylabel(algorithm)
        for column, (title, ylabel) in enumerate((
            ("Online return", "Return"),
            ("Cliff falls per episode", "Falls"),
            ("Exact greedy return", "Return"),
        )):
            axes[row, column].set(title=title, xlabel="Environment steps", ylabel=ylabel)
            axes[row, column].grid(alpha=0.25)
    axes[0, -1].legend(title="Alpha", fontsize=8)
    fig.suptitle("Cliff Walking learning-rate sensitivity (epsilon = 0.1 fixed)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_schedule_sweep(episode_df: pd.DataFrame, evaluation_df: pd.DataFrame, output_path: Path) -> None:
    training = _step_summary(
        episode_df[episode_df["experiment"] == "epsilon_schedule"],
        ["return", "cliff_falls"],
    )
    evaluation = evaluation_df[evaluation_df["experiment"] == "epsilon_schedule"].groupby(
        ["condition", "algorithm", "cumulative_steps"], as_index=False
    ).eval_return.mean()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for (algorithm, condition), group in training.groupby(["algorithm", "condition"]):
        label = f"{algorithm} - {condition}"
        axes[0].plot(group["steps"], group["return"], label=label)
        axes[1].plot(group["steps"], group["cliff_falls"], label=label)
    for (algorithm, condition), group in evaluation.groupby(["algorithm", "condition"]):
        axes[2].plot(
            group["cumulative_steps"], group["eval_return"],
            label=f"{algorithm} - {condition}",
        )
    for axis, (title, ylabel) in zip(axes, (
        ("Online return", "Return"),
        ("Cliff falls per episode", "Falls"),
        ("Exact greedy return", "Return"),
    )):
        axis.set(title=title, xlabel="Environment steps", ylabel=ylabel)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Cliff Walking exploration-schedule sensitivity (alpha = 0.1)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_summary(evaluation_df: pd.DataFrame, output_path: Path) -> None:
    final = evaluation_df.sort_values("cumulative_steps").groupby(
        ["experiment", "condition", "algorithm", "seed"], as_index=False
    ).tail(1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, experiment in zip(axes, ("alpha", "epsilon_schedule")):
        data = final[final["experiment"] == experiment]
        labels, series = [], []
        for (algorithm, condition), group in data.groupby(["algorithm", "condition"]):
            labels.append(f"{algorithm}\n{condition}")
            series.append(group["eval_return"].to_numpy())
        axis.boxplot(series, tick_labels=labels, showmeans=True)
        axis.set(title=f"Final greedy return: {experiment}", ylabel="Return")
        axis.tick_params(axis="x", labelrotation=45)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(seeds: list[int] | None = None, step_budget: int = 30_000):
    seeds = list(range(30)) if seeds is None else seeds
    config = SensitivityConfig(
        step_budget=step_budget,
        eval_interval_steps=max(500, step_budget // 25),
    )
    episode_parts: list[pd.DataFrame] = []
    evaluation_parts: list[pd.DataFrame] = []
    algorithms = (("SARSA", Sarsa), ("Q-Learning", QLearning))

    configurations: list[tuple[str, str, float, str]] = []
    configurations.extend(("alpha", f"alpha={alpha:g}", alpha, "fixed") for alpha in ALPHA_VALUES)
    configurations.extend(
        ("epsilon_schedule", schedule, 0.1, schedule) for schedule in SCHEDULES
    )

    for experiment, condition, alpha, schedule in configurations:
        for algorithm, agent in algorithms:
            for seed in seeds:
                episodes, evaluations = train_seed(
                    experiment,
                    condition,
                    algorithm,
                    _agent_factory(agent),
                    alpha,
                    schedule,
                    config,
                    seed,
                )
                episode_parts.append(episodes)
                evaluation_parts.append(evaluations)

    episodes_df = pd.concat(episode_parts, ignore_index=True)
    evaluations_df = pd.concat(evaluation_parts, ignore_index=True)
    seed_metrics = add_seed_metrics(evaluations_df, config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    episodes_df.to_csv(OUTPUT_DIR / "cliff_sensitivity_episodes.csv", index=False)
    evaluations_df.to_csv(OUTPUT_DIR / "cliff_sensitivity_evaluations.csv", index=False)
    seed_metrics.to_csv(OUTPUT_DIR / "cliff_sensitivity_seed_metrics.csv", index=False)
    pd.Series(asdict(config)).to_json(OUTPUT_DIR / "cliff_sensitivity_config.json", indent=2)
    plot_alpha_sweep(episodes_df, evaluations_df, OUTPUT_DIR / "alpha_sensitivity.png")
    plot_schedule_sweep(episodes_df, evaluations_df, OUTPUT_DIR / "epsilon_sensitivity.png")
    plot_final_summary(evaluations_df, OUTPUT_DIR / "final_summary.png")
    return episodes_df, evaluations_df, seed_metrics


if __name__ == "__main__":
    run_experiment()
