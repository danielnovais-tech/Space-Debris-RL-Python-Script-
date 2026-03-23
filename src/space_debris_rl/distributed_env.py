from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DistributedServiceEnv(gym.Env):
    """Multi-node service-cluster simulation environment.

    Observations: flattened per-node metrics [cpu, error_rate, mem] for each node.
    Actions: MultiDiscrete([4]*num_nodes) where per-node action is:
      0=no-op, 1=restart, 2=scale_up, 3=clear_cache

    This is intentionally simple and deterministic-ish to serve as a testbed for
    safety wrappers, hierarchical control, and federated coordination.
    """

    metadata = {"render_modes": []}

    def __init__(self, *, num_nodes: int = 4, max_steps: int = 200, seed: int = 0):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.max_steps = int(max_steps)
        self._rng = np.random.default_rng(int(seed))

        self.step_count = 0
        self.state = np.zeros((self.num_nodes, 3), dtype=np.float32)
        self.healthy_state = np.array(
            [[30.0, 1.0, 40.0] for _ in range(self.num_nodes)], dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete([4] * self.num_nodes)
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(self.num_nodes * 3,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self.step_count = 0
        self.state = self.healthy_state.copy()
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return self.state.flatten().astype(np.float32, copy=False)

    def step(self, actions):
        self.step_count += 1
        acts = np.asarray(actions, dtype=np.int64).reshape((self.num_nodes,))

        for i, act in enumerate(acts.tolist()):
            if act == 1:  # restart
                self.state[i] = self.healthy_state[i]
            elif act == 2:  # scale_up
                self.state[i, 0] = max(5.0, float(self.state[i, 0] - 10.0))
            elif act == 3:  # clear_cache
                self.state[i, 2] = max(10.0, float(self.state[i, 2] - 20.0))

        noise = self._rng.normal(0.0, 2.0, size=self.state.shape).astype(np.float32)
        self.state = np.clip(self.state + noise, 0.0, 100.0)

        reward = -float(np.mean(self.state[:, 1]))
        reward -= 0.01 * float(np.mean(self.state[:, 0]))
        reward -= 0.01 * float(np.mean(self.state[:, 2]))

        terminated = False
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, {}
