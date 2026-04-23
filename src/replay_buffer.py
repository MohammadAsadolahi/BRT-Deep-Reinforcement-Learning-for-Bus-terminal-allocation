"""
Replay Buffer for experience replay in off-policy RL algorithms.
"""

import numpy as np
from typing import Tuple


class ReplayBuffer:
    """
    Fixed-size circular replay buffer storing (s, a, r, s', done) transitions.

    Uses pre-allocated NumPy arrays for cache-friendly, zero-copy sampling —
    significantly faster than Python-list-based buffers at scale.

    Args:
        capacity: Maximum number of transitions stored.
        state_dim: Dimensionality of the observation vector.
    """

    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._cursor = 0
        self.size = 0

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        """Store a single transition, overwriting oldest data if full."""
        idx = self._cursor
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self._cursor = (self._cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch of transitions."""
        effective = min(self.size, batch_size)
        indices = np.random.choice(self.size, size=effective, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
