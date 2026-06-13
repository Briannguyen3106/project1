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
from .Sarsa import Sarsa
from .common import epsilon_greedy_probabilities, greedy_actions


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures" / "environment2_frozen_lake"


@dataclass(frozen=True)
class FrozenLakeConfig:
    step_budget: int = 100_000
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_fraction: float = 0.8
    max_episode_steps: int = 200
    eval_interval_steps: int = 4_000
    diagnostic_interval_steps: int = 10

    def epsilon(self, cumulative_steps: int) -> float:
        decay_steps = max(1, int(self.step_budget * self.epsilon_decay_fraction))
        progress = min(cumulative_steps / decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)


def make_env(slippery: bool):
    return gym.make(
        "FrozenLake-v1",
        is_slippery=slippery,
        max_episode_steps=200,
    )


def _agent_factory(cls):
    return lambda states, actions, alpha, gamma, seed: cls(states, actions, alpha, gamma, seed)


def _greedy_probabilities(q_values: np.ndarray) -> np.ndarray:
    probabilities = np.zeros(q_values.shape[0], dtype=float)
    best = greedy_actions(q_values)
    probabilities[best] = 1.0 / len(best)
    return probabilities


def exact_greedy_evaluation(
    env: gym.Env,
    q_table: np.ndarray,
    max_steps: int,
) -> dict[str, float]:
    """Evaluate the tie-randomized greedy policy exactly from Frozen Lake's model."""
    transition_model = env.unwrapped.P
    state_count, action_count = q_table.shape
    policy = np.vstack([_greedy_probabilities(q_table[state]) for state in range(state_count)])
    continuation = np.zeros((state_count, state_count), dtype=float)
    success = np.zeros(state_count, dtype=float)
    for state in range(state_count):
        for action in range(action_count):
            action_probability = policy[state, action]
            if action_probability == 0.0:
                continue
            for probability, next_state, reward, terminated in transition_model[state][action]:
                mass = action_probability * probability
                if terminated:
                    if reward > 0.0:
                        success[state] += mass
                else:
                    continuation[state, int(next_state)] += mass

    distribution = np.zeros(state_count, dtype=float)
    distribution[0] = 1.0
    success_probability = 0.0
    expected_length = 0.0

    for _ in range(max_steps):
        alive_probability = float(distribution.sum())
        if alive_probability <= 1e-15:
            break
        expected_length += alive_probability
        success_probability += float(np.dot(distribution, success))
        distribution = distribution @ continuation

    return {
        "eval_return": success_probability,
        "eval_success_rate": success_probability,
        "eval_episode_length": expected_length,
        "eval_unfinished_probability": float(distribution.sum()),
    }


def _variance_components(
    transition_model: dict,
    q_table: np.ndarray,
    state: int,
    action: int,
    epsilon: float,
    gamma: float,
) -> tuple[float, float]:
    """Return transition variance and expected next-action sampling variance.

    The decomposition conditions on the current (state, action) and current Q-table.
    Expected SARSA incurs only the transition component. SARSA additionally incurs
    the action-sampling component.
    """
    conditional_means: list[float] = []
    probabilities: list[float] = []
    action_variances: list[float] = []

    for probability, next_state, reward, terminated in transition_model[state][action]:
        if terminated:
            mean_target = float(reward)
            action_variance = 0.0
        else:
            action_probabilities = epsilon_greedy_probabilities(q_table[int(next_state)], epsilon)
            next_values = q_table[int(next_state)]
            expected_value = float(np.dot(action_probabilities, next_values))
            value_variance = float(
                np.dot(action_probabilities, np.square(next_values - expected_value))
            )
            mean_target = float(reward) + gamma * expected_value
            action_variance = gamma**2 * value_variance
        probabilities.append(float(probability))
        conditional_means.append(mean_target)
        action_variances.append(action_variance)

    weights = np.asarray(probabilities, dtype=float)
    means = np.asarray(conditional_means, dtype=float)
    overall_mean = float(np.dot(weights, means))
    transition_variance = float(np.dot(weights, np.square(means - overall_mean)))
    expected_action_variance = float(np.dot(weights, np.asarray(action_variances)))
    return transition_variance, expected_action_variance


def train_seed(
    algorithm: str,
    agent_factory: Callable,
    slippery: bool,
    config: FrozenLakeConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    env = make_env(slippery)
    transition_model = env.unwrapped.P
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
        metrics = exact_greedy_evaluation(env, agent.q_table, config.max_episode_steps)
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
            targets: list[float] = []
            errors: list[float] = []
            transition_variances: list[float] = []
            action_variances: list[float] = []
            steps = 0
            terminated = truncated = False

            while (
                not (terminated or truncated)
                and steps < config.max_episode_steps
                and cumulative_steps < config.step_budget
            ):
                epsilon = config.epsilon(cumulative_steps)
                diagnostic_step = cumulative_steps % config.diagnostic_interval_steps == 0
                if diagnostic_step:
                    transition_variance, action_variance = _variance_components(
                        transition_model,
                        agent.q_table,
                        state,
                        action,
                        epsilon,
                        config.gamma,
                    )
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = int(next_state)
                reward = float(reward)
                environment_done = bool(terminated or truncated)
                next_action = None if environment_done else agent.select_action(next_state, epsilon)
                target, error = agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    environment_done,
                    next_action,
                    epsilon,
                )
                targets.append(target)
                errors.append(error)
                if diagnostic_step:
                    transition_variances.append(transition_variance)
                    action_variances.append(action_variance if algorithm == "SARSA" else 0.0)
                total_reward += reward
                steps += 1
                cumulative_steps += 1
                state = next_state
                if next_action is not None:
                    action = next_action

                while cumulative_steps >= next_evaluation_step:
                    record_evaluation()
                    next_evaluation_step += config.eval_interval_steps

            success = bool(terminated and total_reward > 0.0)
            if success and first_success_step is None:
                first_success_step = cumulative_steps
            transition_mean = (
                float(np.mean(transition_variances)) if transition_variances else np.nan
            )
            action_mean = float(np.mean(action_variances)) if action_variances else np.nan
            diagnostic_count = len(transition_variances)
            transition_sum = float(np.sum(transition_variances))
            action_sum = float(np.sum(action_variances))
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
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "budget_cutoff": int(cumulative_steps >= config.step_budget and not environment_done),
                    "td_target_mean": float(np.mean(targets)),
                    "td_error_mean": float(np.mean(errors)),
                    "transition_variance": transition_mean,
                    "action_sampling_variance": action_mean,
                    "model_target_variance": transition_mean + action_mean,
                    "transition_variance_sum": transition_sum,
                    "action_sampling_variance_sum": action_sum,
                    "model_target_variance_sum": transition_sum + action_sum,
                    "diagnostic_count": diagnostic_count,
                    "first_success_step": first_success_step if first_success_step is not None else -1,
                }
            )

        if evaluation_rows[-1]["cumulative_steps"] != cumulative_steps:
            record_evaluation()
    finally:
        env.close()

    return pd.DataFrame(episode_rows), pd.DataFrame(evaluation_rows), agent.q_table.copy()


def _step_bin_summary(data: pd.DataFrame, value_columns: list[str], bins: int = 25) -> pd.DataFrame:
    result = data.copy()
    maximum = int(result["cumulative_steps"].max())
    edges = np.linspace(0, maximum, bins + 1)
    result["step_bin"] = pd.cut(
        result["cumulative_steps"], edges, labels=False, include_lowest=True
    )
    aggregations = {column: (column, "mean") for column in value_columns}
    summary = result.groupby(["setting", "algorithm", "step_bin"], as_index=False).agg(
        steps=("cumulative_steps", "mean"),
        **aggregations,
    )
    return summary


def plot_training(episode_df: pd.DataFrame, setting: str, output_path: Path) -> None:
    data = episode_df[episode_df["setting"] == setting]
    summary = _step_bin_summary(data, ["success"])
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for algorithm, group in summary.groupby("algorithm"):
        axis.plot(group["steps"], group["success"], marker="o", label=algorithm)
    axis.set(
        title=f"Frozen Lake - {setting}: online success rate",
        xlabel="Environment steps",
        ylabel="Success rate",
        ylim=(-0.03, 1.03),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_evaluation(evaluation_df: pd.DataFrame, setting: str, output_path: Path) -> None:
    data = evaluation_df[evaluation_df["setting"] == setting]
    summary = data.groupby(["algorithm", "cumulative_steps"], as_index=False).agg(
        mean_success=("eval_success_rate", "mean"),
        std_success=("eval_success_rate", "std"),
        count=("seed", "nunique"),
    )
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for algorithm, group in summary.groupby("algorithm"):
        x = group["cumulative_steps"].to_numpy()
        mean = group["mean_success"].to_numpy()
        ci = 1.96 * group["std_success"].fillna(0).to_numpy() / np.sqrt(group["count"])
        axis.plot(x, mean, marker="o", label=algorithm)
        axis.fill_between(x, mean - ci, mean + ci, alpha=0.18)
    axis.set(
        title=f"Frozen Lake - {setting}: exact greedy success probability",
        xlabel="Environment steps",
        ylabel="Success probability",
        ylim=(-0.03, 1.03),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_variance_components(episode_df: pd.DataFrame, output_path: Path) -> None:
    data = episode_df.copy()
    maximum = int(data["cumulative_steps"].max())
    data["step_bin"] = pd.cut(
        data["cumulative_steps"],
        np.linspace(0, maximum, 26),
        labels=False,
        include_lowest=True,
    )
    summary = data.groupby(["setting", "algorithm", "step_bin"], as_index=False).agg(
        steps=("cumulative_steps", "mean"),
        diagnostic_count=("diagnostic_count", "sum"),
        action_sampling_variance_sum=("action_sampling_variance_sum", "sum"),
        transition_variance_sum=("transition_variance_sum", "sum"),
        model_target_variance_sum=("model_target_variance_sum", "sum"),
    )
    for name in ("action_sampling_variance", "transition_variance", "model_target_variance"):
        summary[name] = summary[f"{name}_sum"] / summary["diagnostic_count"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    columns = [
        ("action_sampling_variance", "Action-sampling variance"),
        ("transition_variance", "Transition/reward variance"),
        ("model_target_variance", "Total conditional target variance"),
    ]
    for axis, (column, title) in zip(axes, columns):
        for (setting, algorithm), group in summary.groupby(["setting", "algorithm"]):
            axis.plot(group["steps"], group[column], label=f"{algorithm} ({setting})")
        axis.set(title=title, xlabel="Environment steps", ylabel="Variance")
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Frozen Lake: model-based conditional variance decomposition")
    fig.tight_layout()
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
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    seeds: list[int] | None = None,
    step_budget: int = 100_000,
):
    seeds = list(range(30)) if seeds is None else seeds
    config = FrozenLakeConfig(
        step_budget=step_budget,
        eval_interval_steps=max(1_000, step_budget // 25),
    )
    all_episodes, all_evaluations = [], []
    q_tables = {}
    for slippery, setting in ((False, "deterministic"), (True, "slippery")):
        for name, agent in (("SARSA", Sarsa), ("Expected SARSA", ExpectedSarsa)):
            episode_parts, evaluation_parts, tables = [], [], {}
            for seed in seeds:
                episode_df, evaluation_df, q_table = train_seed(
                    name,
                    _agent_factory(agent),
                    slippery,
                    config,
                    seed,
                )
                episode_parts.append(episode_df)
                evaluation_parts.append(evaluation_df)
                tables[seed] = q_table
            episode_df = pd.concat(episode_parts, ignore_index=True)
            evaluation_df = pd.concat(evaluation_parts, ignore_index=True)
            episode_df["setting"] = setting
            evaluation_df["setting"] = setting
            all_episodes.append(episode_df)
            all_evaluations.append(evaluation_df)
            q_tables[(setting, name)] = tables

    episodes_df = pd.concat(all_episodes, ignore_index=True)
    evaluations_df = pd.concat(all_evaluations, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    episodes_df.to_csv(OUTPUT_DIR / "frozen_lake_episodes.csv", index=False)
    evaluations_df.to_csv(OUTPUT_DIR / "frozen_lake_evaluations.csv", index=False)
    pd.Series(asdict(config)).to_json(OUTPUT_DIR / "frozen_lake_config.json", indent=2)
    for setting in ("deterministic", "slippery"):
        plot_training(episodes_df, setting, OUTPUT_DIR / f"training_{setting}.png")
        plot_evaluation(evaluations_df, setting, OUTPUT_DIR / f"evaluation_{setting}.png")
    plot_variance_components(episodes_df, OUTPUT_DIR / "variance_decomposition.png")
    plot_q_heatmaps(q_tables, seeds[0], OUTPUT_DIR / "q_heatmaps.png")
    return episodes_df, evaluations_df, q_tables


if __name__ == "__main__":
    run_experiment()
