"""
Bus Transit Environment for Reinforcement Learning.

Simulates a bus rapid transit (BRT) system where an RL agent dispatches
buses across multiple transit lines to minimize passenger wait times
and maximize throughput.
"""

import numpy as np
from typing import Dict, List, Tuple


class BusTransitEnvironment:
    """
    A custom OpenAI-Gym-style environment modeling a bus rapid transit network.

    The environment simulates passenger arrivals at stations across multiple
    transit lines. At each timestep the agent decides which line to dispatch
    the next available bus to (or to hold). Buses traverse their assigned line,
    picking up passengers at each station (up to capacity), then return to the
    depot.

    Observation Space:
        A 1-D vector of shape (num_stations + num_buses,) containing:
        - Current passenger counts at every station across all lines.
        - Current position index of each bus (-1 = idle at depot).

    Action Space:
        Discrete(num_lines + 1):
        - action ∈ [0, num_lines-1]: dispatch an idle bus to the given line.
        - action == num_lines: hold (do nothing).

    Reward:
        At each step the reward penalises total waiting passengers and rewards
        passengers picked up by active buses (weighted ×3).

    Args:
        line_config: Mapping of line names to lists of mean passenger arrival
            rates per station. Example:
            ``{"line_1": [1, 2, 3], "line_2": [6, 7, 4]}``
        num_buses: Total fleet size available for dispatch.
        bus_capacity: Maximum passengers a single bus picks up per station.
        max_steps: Steps after which the episode is truncated.
    """

    def __init__(
        self,
        line_config: Dict[str, List[int]],
        num_buses: int = 20,
        bus_capacity: int = 20,
        max_steps: int = 1000,
    ) -> None:
        self.line_config = line_config
        self.num_buses = num_buses
        self.bus_capacity = bus_capacity
        self.max_steps = max_steps

        self.num_actions = len(line_config) + 1  # one per line + hold
        self._arrival_means = np.concatenate(
            [np.array(v) for v in line_config.values()]
        )
        self.num_stations = len(self._arrival_means)

        # Pre-compute (start, end) index ranges for each line
        self._line_ranges: List[Tuple[int, int]] = []
        offset = 0
        for stations in line_config.values():
            length = len(stations)
            self._line_ranges.append((offset, offset + length))
            offset += length

        # State arrays initialised in reset()
        self._passengers = np.zeros(self.num_stations, dtype=np.float64)
        self._bus_positions = np.full(self.num_buses, -1, dtype=np.float64)
        self._step_count = 0

    # ------------------------------------------------------------------
    # Gym-compatible interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        self._step_count = 0
        self._bus_positions = np.full(self.num_buses, -1, dtype=np.float64)
        self._passengers = self._sample_arrivals()
        return self._observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool]:
        """
        Execute one timestep.

        Returns:
            observation, reward, done, truncated
        """
        # 1. Dispatch a bus if action selects a valid line
        if action < len(self._line_ranges):
            idle = np.where(self._bus_positions == -1)[0]
            if idle.size > 0:
                self._bus_positions[idle[0]] = self._line_ranges[action][0]

        # 2. Move all active buses and compute reward
        reward = self._advance_buses()

        # 3. New passengers arrive
        self._passengers += self._sample_arrivals()

        self._step_count += 1
        truncated = self._step_count > self.max_steps
        done = False

        return self._observation(), reward, done, truncated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        return np.concatenate([self._passengers, self._bus_positions])

    def _advance_buses(self) -> float:
        """Move each active bus one station forward and compute reward."""
        reward = -float(self._passengers.sum())

        for i in range(self.num_buses):
            pos = int(self._bus_positions[i])
            if pos == -1:
                continue
            for start, end in self._line_ranges:
                if start <= pos < end:
                    picked = min(self._passengers[pos], self.bus_capacity)
                    reward += 3.0 * picked
                    self._passengers[pos] = max(
                        self._passengers[pos] - self.bus_capacity, 0
                    )
                    self._bus_positions[i] += 1
                    if self._bus_positions[i] == end:
                        self._bus_positions[i] = -1
                    break

        return reward

    def _sample_arrivals(self) -> np.ndarray:
        """Sample new passenger arrivals from |N(0,1)| + mean."""
        noise = np.abs(np.random.randn())
        arrivals = (noise + self._arrival_means).astype(int)
        return arrivals.astype(np.float64)

    @property
    def observation_dim(self) -> int:
        return self.num_stations + self.num_buses
