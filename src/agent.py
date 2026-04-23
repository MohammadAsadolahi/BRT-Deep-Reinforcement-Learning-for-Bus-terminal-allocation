"""
Double Deep Q-Network (DDQN) — neural network and agent.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.replay_buffer import ReplayBuffer


# ======================================================================
# Network
# ======================================================================

class QNetwork(nn.Module):
    """
    Fully-connected Q-value network.

    Architecture: state → 256 → 256 → |A| (action values)

    Args:
        state_dim: Dimensionality of the observation vector.
        action_dim: Number of discrete actions.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ======================================================================
# Agent
# ======================================================================

class DDQNAgent:
    """
    Double DQN agent with epsilon-greedy exploration.

    Key DDQN idea: the *online* network selects the best next action,
    but the *target* network evaluates that action's Q-value — decoupling
    selection from evaluation to reduce overestimation bias.

    Args:
        state_dim: Observation vector size.
        action_dim: Number of discrete actions.
        lr: Learning rate for Adam optimiser.
        gamma: Discount factor.
        epsilon: Initial exploration rate.
        epsilon_decay: Multiplicative decay per learning step.
        epsilon_min: Floor for exploration rate.
        buffer_capacity: Replay buffer size.
        target_update_freq: Copy online → target every N learning steps.
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9999,
        epsilon_min: float = 0.01,
        buffer_capacity: int = 1_000_000,
        target_update_freq: int = 100,
        device: str | None = None,
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dim = action_dim
        self.target_update_freq = target_update_freq

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Networks
        self.online_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        self.optimiser = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity, state_dim)

        self._learn_steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: Current observation.
            eval_mode: If True, always pick the greedy action (no exploration).
        """
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, batch_size: int = 64) -> float | None:
        """
        Sample a mini-batch and perform one gradient step (Double DQN).

        Returns the loss value, or None if the buffer is too small.
        """
        if self.buffer.size < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            batch_size
        )

        states_t = torch.tensor(
            states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(
            rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q(s, a) from the online network
        q_pred = self.online_net(states_t).gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_actions = self.online_net(next_states_t).argmax(dim=1)
            q_next = self.target_net(next_states_t).gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
            q_target = rewards_t + self.gamma * dones_t * q_next

        loss = F.mse_loss(q_pred, q_target)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # Bookkeeping
        self._learn_steps += 1
        self.epsilon = max(
            self.epsilon * self.epsilon_decay, self.epsilon_min
        )

        # Periodic hard target update
        if self._learn_steps % self.target_update_freq == 0:
            self.target_net = deepcopy(self.online_net)
            self.target_net.eval()

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save online network weights to disk."""
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load online network weights and sync target network."""
        self.online_net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.target_net = deepcopy(self.online_net)
        self.target_net.eval()
