from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Agent(Protocol):
    q_table: np.ndarray

    def select_action(self, state: int, epsilon: float) -> int: ...

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        next_action: int | None = None,
        epsilon: float = 0.0,
    ) -> tuple[float, float]: ...


@dataclass(frozen=True)
class TrainConfig:
    episodes: int
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_fraction: float = 0.8
    max_steps: int = 500
    eval_interval: int = 100
    eval_episodes: int = 30

    def epsilon(self, episode: int) -> float:
        decay_episodes = max(1, int(self.episodes * self.epsilon_decay_fraction))
        progress = min(episode / decay_episodes, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)


def greedy_actions(q_values: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isclose(q_values, np.max(q_values)))


def epsilon_greedy_probabilities(q_values: np.ndarray, epsilon: float) -> np.ndarray:
    action_count = q_values.shape[0]
    probabilities = np.full(action_count, epsilon / action_count, dtype=float)
    best = greedy_actions(q_values)
    probabilities[best] += (1.0 - epsilon) / len(best)
    return probabilities


def evaluate(
    env_factory: Callable[[], gym.Env],
    q_table: np.ndarray,
    episodes: int,
    seed: int,
    max_steps: int,
    success_test: Callable[[float, bool], bool],
) -> dict[str, float]:
    env = env_factory()
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    try:
        for index in range(episodes):
            state, _ = env.reset(seed=seed + index)
            total_reward = 0.0
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated) and steps < max_steps:
                best = greedy_actions(q_table[int(state)])
                action = int(rng.choice(best))
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1
            returns.append(total_reward)
            lengths.append(steps)
            successes += int(success_test(total_reward, terminated))
    finally:
        env.close()
    return {
        "eval_return": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "eval_success_rate": successes / episodes,
        "eval_episode_length": float(np.mean(lengths)),
    }


def train_seed(
    algorithm: str,
    agent_factory: Callable[[int, int, float, float, int], Agent],
    env_factory: Callable[[], gym.Env],
    config: TrainConfig,
    seed: int,
    success_test: Callable[[float, bool], bool],
    event_counter: Callable[[float, dict], int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    env = env_factory()
    state_count = int(env.observation_space.n)
    action_count = int(env.action_space.n)
    agent = agent_factory(state_count, action_count, config.alpha, config.gamma, seed)
    episode_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    cumulative_steps = 0
    first_success_step: int | None = None

    try:
        for episode in range(config.episodes):
            epsilon = config.epsilon(episode)
            state, _ = env.reset(seed=seed * 100_000 + episode)
            state = int(state)
            action = agent.select_action(state, epsilon)
            total_reward = 0.0
            td_targets: list[float] = []
            td_errors: list[float] = []
            events = 0
            terminated = truncated = False
            steps = 0

            while not (terminated or truncated) and steps < config.max_steps:
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = int(next_state)
                reward = float(reward)
                next_action = None if terminated else agent.select_action(next_state, epsilon)
                target, error = agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated,
                    next_action,
                    epsilon,
                )
                td_targets.append(target)
                td_errors.append(error)
                total_reward += reward
                steps += 1
                cumulative_steps += 1
                if event_counter is not None:
                    events += event_counter(reward, info)
                state = next_state
                if next_action is not None:
                    action = next_action

            success = success_test(total_reward, terminated)
            if success and first_success_step is None:
                first_success_step = cumulative_steps
            episode_rows.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "episode": episode + 1,
                    "cumulative_steps": cumulative_steps,
                    "epsilon": epsilon,
                    "return": total_reward,
                    "episode_length": steps,
                    "success": int(success),
                    "events": events,
                    "td_target_mean": float(np.mean(td_targets)),
                    "td_target_variance": float(np.var(td_targets, ddof=1)) if len(td_targets) > 1 else 0.0,
                    "td_error_mean": float(np.mean(td_errors)),
                    "td_error_variance": float(np.var(td_errors, ddof=1)) if len(td_errors) > 1 else 0.0,
                    "first_success_step": first_success_step if first_success_step is not None else -1,
                }
            )

            checkpoint = episode == 0 or (episode + 1) % config.eval_interval == 0
            if checkpoint or episode + 1 == config.episodes:
                metrics = evaluate(
                    env_factory,
                    agent.q_table,
                    config.eval_episodes,
                    seed * 1_000_000 + episode,
                    config.max_steps,
                    success_test,
                )
                evaluation_rows.append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "episode": episode + 1,
                        "cumulative_steps": cumulative_steps,
                        **metrics,
                    }
                )
    finally:
        env.close()

    return pd.DataFrame(episode_rows), pd.DataFrame(evaluation_rows), agent.q_table.copy()


def run_many(
    algorithm: str,
    agent_factory: Callable[[int, int, float, float, int], Agent],
    env_factory: Callable[[], gym.Env],
    config: TrainConfig,
    seeds: list[int],
    success_test: Callable[[float, bool], bool],
    event_counter: Callable[[float, dict], int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    episodes: list[pd.DataFrame] = []
    evaluations: list[pd.DataFrame] = []
    q_tables: dict[int, np.ndarray] = {}
    for seed in seeds:
        episode_df, evaluation_df, q_table = train_seed(
            algorithm,
            agent_factory,
            env_factory,
            config,
            seed,
            success_test,
            event_counter,
        )
        episodes.append(episode_df)
        evaluations.append(evaluation_df)
        q_tables[seed] = q_table
    return pd.concat(episodes, ignore_index=True), pd.concat(evaluations, ignore_index=True), q_tables


def save_experiment(
    output_dir: Path,
    name: str,
    config: TrainConfig,
    episode_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_df.to_csv(output_dir / f"{name}_episodes.csv", index=False)
    evaluation_df.to_csv(output_dir / f"{name}_evaluations.csv", index=False)
    pd.Series(asdict(config)).to_json(output_dir / f"{name}_config.json", indent=2)


def plot_learning_curves(evaluation_df: pd.DataFrame, output_path: Path, title: str) -> None:
    summary = (
        evaluation_df.groupby(["algorithm", "episode"], as_index=False)
        .agg(
            mean_return=("eval_return", "mean"),
            std_return=("eval_return", "std"),
            mean_success=("eval_success_rate", "mean"),
            std_success=("eval_success_rate", "std"),
        )
    )
    counts = evaluation_df.groupby(["algorithm", "episode"]).size().rename("n").reset_index()
    summary = summary.merge(counts, on=["algorithm", "episode"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for algorithm, group in summary.groupby("algorithm"):
        x = group["episode"].to_numpy()
        return_mean = group["mean_return"].to_numpy()
        return_ci = 1.96 * group["std_return"].fillna(0).to_numpy() / np.sqrt(group["n"].to_numpy())
        success_mean = group["mean_success"].to_numpy()
        success_ci = 1.96 * group["std_success"].fillna(0).to_numpy() / np.sqrt(group["n"].to_numpy())
        axes[0].plot(x, return_mean, label=algorithm)
        axes[0].fill_between(x, return_mean - return_ci, return_mean + return_ci, alpha=0.18)
        axes[1].plot(x, success_mean, label=algorithm)
        axes[1].fill_between(x, np.maximum(0, success_mean - success_ci), np.minimum(1, success_mean + success_ci), alpha=0.18)
    axes[0].set(title="Greedy evaluation return", xlabel="Episode", ylabel="Mean return")
    axes[1].set(title="Greedy evaluation success rate", xlabel="Episode", ylabel="Success rate", ylim=(-0.02, 1.02))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_summary(episode_df: pd.DataFrame, output_path: Path, title: str, window: int = 50) -> None:
    ordered = episode_df.sort_values(["algorithm", "seed", "episode"]).copy()
    ordered["moving_return"] = ordered.groupby(["algorithm", "seed"])["return"].transform(
        lambda values: values.rolling(window, min_periods=1).mean()
    )
    summary = ordered.groupby(["algorithm", "episode"], as_index=False).agg(
        mean_return=("moving_return", "mean"),
        std_return=("moving_return", "std"),
        mean_events=("events", "mean"),
        n=("seed", "nunique"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for algorithm, group in summary.groupby("algorithm"):
        x = group["episode"].to_numpy()
        mean = group["mean_return"].to_numpy()
        ci = 1.96 * group["std_return"].fillna(0).to_numpy() / np.sqrt(group["n"].to_numpy())
        axes[0].plot(x, mean, label=algorithm)
        axes[0].fill_between(x, mean - ci, mean + ci, alpha=0.18)
        axes[1].plot(x, group["mean_events"], label=algorithm)
    axes[0].set(title=f"Training return ({window}-episode moving average)", xlabel="Episode", ylabel="Return")
    axes[1].set(title="Mean safety/error events per episode", xlabel="Episode", ylabel="Events")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
