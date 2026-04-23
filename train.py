"""
Training and evaluation entry-point for the Bus Terminal Allocation DDQN agent.

Usage:
    python train.py                       # Train with defaults
    python train.py --episodes 500        # Override episode count
    python train.py --eval                 # Evaluate a saved model
    python train.py --compare              # Compare trained vs random policy
"""

from src.agent import DDQNAgent
from src.environment import BusTransitEnvironment
import torch
import matplotlib.pyplot as plt
import argparse
import os
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI


# ======================================================================
# Defaults
# ======================================================================

DEFAULT_LINE_CONFIG = {
    "line_1": [1, 2, 3, 3, 1, 3, 4, 4, 3, 4, 2, 1],
    "line_2": [6, 7, 4, 3, 8, 9, 7, 6, 8, 9, 8, 3, 4, 4, 9, 9],
}

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")


def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across NumPy, Python, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ======================================================================
# Training loop
# ======================================================================

def train(
    env: BusTransitEnvironment,
    agent: DDQNAgent,
    episodes: int = 200,
    batch_size: int = 64,
    verbose: bool = True,
) -> tuple[list[float], list[float]]:
    """
    Train the DDQN agent.

    Returns:
        (episode_rewards, running_avg_rewards)
    """
    episode_rewards: list[float] = []
    avg_rewards: list[float] = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done, truncated = False, False

        while not done and not truncated:
            action = agent.select_action(state)
            next_state, reward, done, truncated = env.step(action)
            agent.buffer.store(state, action, reward,
                               next_state, 1.0 - float(done))
            agent.learn(batch_size)
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        avg = np.mean(episode_rewards).item()
        avg_rewards.append(avg)

        if verbose:
            print(
                f"Episode {ep:>4d}/{episodes}  |  "
                f"Reward: {total_reward:>10.1f}  |  "
                f"Avg: {avg:>10.1f}  |  "
                f"ε: {agent.epsilon:.4f}"
            )

    return episode_rewards, avg_rewards


# ======================================================================
# Random baseline
# ======================================================================

def evaluate_random(
    env: BusTransitEnvironment,
    episodes: int = 200,
    verbose: bool = True,
) -> tuple[list[float], list[float]]:
    """
    Run a random policy baseline for comparison.

    Returns:
        (episode_rewards, running_avg_rewards)
    """
    episode_rewards: list[float] = []
    avg_rewards: list[float] = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done, truncated = False, False

        while not done and not truncated:
            action = np.random.randint(env.num_actions)
            next_state, reward, done, truncated = env.step(action)
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        avg = np.mean(episode_rewards).item()
        avg_rewards.append(avg)

        if verbose:
            print(
                f"[Random] Episode {ep:>4d}/{episodes}  |  "
                f"Reward: {total_reward:>10.1f}  |  "
                f"Avg: {avg:>10.1f}"
            )

    return episode_rewards, avg_rewards


# ======================================================================
# Plotting
# ======================================================================

def _style_plot() -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "figure.figsize": (12, 5),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_results(
    rewards: list[float],
    avg_rewards: list[float],
    label: str = "DDQN",
    save_dir: Path = RESULTS_DIR,
) -> None:
    """Save training curve plots."""
    save_dir.mkdir(parents=True, exist_ok=True)
    _style_plot()

    # --- Episode rewards ---
    fig, ax = plt.subplots()
    ax.plot(rewards, alpha=0.4, label="Episode reward")
    ax.plot(avg_rewards, linewidth=2, label="Running average")
    ax.set_title(f"{label} — Episode Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_yscale("symlog")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / f"{label}_rewards.png", dpi=200)
    plt.close(fig)
    print(f"Saved → {save_dir / f'{label}_rewards.png'}")

    # --- Running average ---
    fig, ax = plt.subplots()
    ax.plot(avg_rewards, linewidth=2, color="tab:orange")
    ax.set_title(f"{label} — Running Average Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    fig.tight_layout()
    fig.savefig(save_dir / f"{label}_avg_rewards.png", dpi=200)
    plt.close(fig)
    print(f"Saved → {save_dir / f'{label}_avg_rewards.png'}")


def plot_comparison(
    ddqn_avg: list[float],
    random_avg: list[float],
    save_dir: Path = RESULTS_DIR,
) -> None:
    """Overlay DDQN vs random baseline on a single chart."""
    save_dir.mkdir(parents=True, exist_ok=True)
    _style_plot()

    fig, ax = plt.subplots()
    ax.plot(ddqn_avg, linewidth=2, label="DDQN Agent")
    ax.plot(random_avg, linewidth=2, linestyle="--", label="Random Policy")
    ax.set_title("DDQN vs Random Policy — Running Average Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "comparison.png", dpi=200)
    plt.close(fig)
    print(f"Saved → {save_dir / 'comparison.png'}")


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train / evaluate a Double DQN agent for bus terminal allocation."
    )
    parser.add_argument("--episodes", type=int,
                        default=200, help="Training episodes")
    parser.add_argument("--buses", type=int, default=15, help="Fleet size")
    parser.add_argument("--batch-size", type=int,
                        default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float,
                        default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--eval", action="store_true", help="Run evaluation only (load saved model)"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare DDQN vs random baseline"
    )
    parser.add_argument("--model-path", type=str,
                        default=None, help="Path to saved model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    env = BusTransitEnvironment(
        line_config=DEFAULT_LINE_CONFIG,
        num_buses=args.buses,
    )

    agent = DDQNAgent(
        state_dim=env.observation_dim,
        action_dim=env.num_actions,
        lr=args.lr,
        gamma=args.gamma,
    )

    if args.eval:
        model_path = args.model_path or str(MODELS_DIR / "ddqn_bus.pt")
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
        rewards, avg = train(env, agent, episodes=args.episodes, batch_size=0)
        plot_results(rewards, avg, label="DDQN_eval")
        return

    # ----- Train DDQN -----
    print("=" * 60)
    print("  Training Double DQN Agent")
    print("=" * 60)
    ddqn_rewards, ddqn_avg = train(
        env, agent, episodes=args.episodes, batch_size=args.batch_size
    )
    plot_results(ddqn_rewards, ddqn_avg, label="DDQN")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "ddqn_bus.pt"
    agent.save(str(save_path))
    print(f"Model saved → {save_path}")

    # ----- Random baseline comparison -----
    if args.compare:
        set_seed(args.seed)
        env_rand = BusTransitEnvironment(
            line_config=DEFAULT_LINE_CONFIG,
            num_buses=args.buses,
        )
        print("\n" + "=" * 60)
        print("  Running Random Policy Baseline")
        print("=" * 60)
        rand_rewards, rand_avg = evaluate_random(
            env_rand, episodes=args.episodes
        )
        plot_results(rand_rewards, rand_avg, label="Random")
        plot_comparison(ddqn_avg, rand_avg)


if __name__ == "__main__":
    main()
