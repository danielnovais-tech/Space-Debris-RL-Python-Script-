from __future__ import annotations

import os
import sys
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SpaceDebrisAvoidanceEnv(gym.Env):
    """2-D spacecraft collision-avoidance environment."""

    metadata = {"render_modes": ["human"], "render.modes": ["human"]}

    def __init__(
        self,
        dt: float = 0.1,
        max_steps: int = 200,
        num_debris: int = 2,
        thrust: float = 0.5,
        debris_speed_range: tuple[float, float] = (0.2, 0.8),
        collision_threshold: float = 1.0,
        goal_threshold: float = 1.0,
        spacecraft_start_pos: np.ndarray | None = None,
        spacecraft_start_vel: np.ndarray | None = None,
        goal_pos: np.ndarray | None = None,
        boundary: float = 20.0,
    ):
        super().__init__()

        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.num_debris = int(num_debris)
        self.thrust = float(thrust)
        self.debris_speed_range = tuple(debris_speed_range)
        self.collision_threshold = float(collision_threshold)
        self.goal_threshold = float(goal_threshold)
        self.boundary = float(boundary)

        self.spacecraft_start_pos = (
            np.array([0.0, 0.0], dtype=np.float32)
            if spacecraft_start_pos is None
            else np.array(spacecraft_start_pos, dtype=np.float32)
        )
        self.spacecraft_start_vel = (
            np.array([0.0, 0.0], dtype=np.float32)
            if spacecraft_start_vel is None
            else np.array(spacecraft_start_vel, dtype=np.float32)
        )
        self.goal_pos = (
            np.array([10.0, 10.0], dtype=np.float32)
            if goal_pos is None
            else np.array(goal_pos, dtype=np.float32)
        )

        obs_dim = 4 + 2 + 4 * self.num_debris
        self.observation_space = spaces.Box(
            low=-self.boundary,
            high=self.boundary,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

        self.step_count: int = 0
        self.spacecraft_pos: np.ndarray = self.spacecraft_start_pos.copy()
        self.spacecraft_vel: np.ndarray = self.spacecraft_start_vel.copy()
        self.debris_pos: np.ndarray = np.zeros((self.num_debris, 2), dtype=np.float32)
        self.debris_vel: np.ndarray = np.zeros((self.num_debris, 2), dtype=np.float32)
        self._traj: list[np.ndarray] = []

        self._configure_matplotlib_backend_for_headless()

    def _configure_matplotlib_backend_for_headless(self) -> None:
        if os.environ.get("DISPLAY") is None and sys.platform != "win32":
            os.environ.setdefault("MPLBACKEND", "Agg")

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.spacecraft_pos = self.spacecraft_start_pos.copy()
        self.spacecraft_vel = self.spacecraft_start_vel.copy()
        self._traj = [self.spacecraft_pos.copy()]

        debris_pos_list: list[np.ndarray] = []
        debris_vel_list: list[np.ndarray] = []
        for _ in range(self.num_debris):
            while True:
                pos = np.random.uniform(-self.boundary / 2, self.boundary / 2, size=2)
                if np.linalg.norm(pos - self.spacecraft_pos) > 5.0:
                    break
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            speed = np.random.uniform(*self.debris_speed_range)
            vel = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * speed
            debris_pos_list.append(pos.astype(np.float32))
            debris_vel_list.append(vel)

        self.debris_pos = np.array(debris_pos_list, dtype=np.float32)
        self.debris_vel = np.array(debris_vel_list, dtype=np.float32)

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.spacecraft_pos,
                self.spacecraft_vel,
                self.goal_pos,
                self.debris_pos.flatten(),
                self.debris_vel.flatten(),
            ]
        ).astype(np.float32)

    def step(self, action: int):
        self.step_count += 1

        dv = self.thrust * self.dt
        if action == 1:
            self.spacecraft_vel[0] += dv
        elif action == 2:
            self.spacecraft_vel[0] -= dv
        elif action == 3:
            self.spacecraft_vel[1] += dv
        elif action == 4:
            self.spacecraft_vel[1] -= dv

        self.spacecraft_pos = self.spacecraft_pos + self.spacecraft_vel * self.dt
        self.debris_pos = self.debris_pos + self.debris_vel * self.dt
        self._traj.append(self.spacecraft_pos.copy())

        collision = bool(
            any(
                np.linalg.norm(self.spacecraft_pos - dpos) < self.collision_threshold
                for dpos in self.debris_pos
            )
        )
        goal_reached = bool(
            np.linalg.norm(self.spacecraft_pos - self.goal_pos) < self.goal_threshold
        )
        out_of_bounds = bool(np.any(np.abs(self.spacecraft_pos) > self.boundary))

        if collision:
            reward = -1.0
            terminated = True
        elif goal_reached:
            reward = 1.0
            terminated = True
        elif out_of_bounds:
            reward = -0.5
            terminated = True
        else:
            reward = -0.01
            terminated = False

        truncated = (not terminated) and (self.step_count >= self.max_steps)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def render(self, mode: str = "human") -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        plt.clf()
        ax = plt.gca()

        if len(self._traj) > 1:
            traj = np.array(self._traj)
            ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=0.8, alpha=0.5, label="Path")

        ax.plot(
            self.spacecraft_pos[0], self.spacecraft_pos[1], "bo", markersize=8, label="Spacecraft"
        )

        for i, pos in enumerate(self.debris_pos):
            ax.plot(pos[0], pos[1], "ro", markersize=6, label="Debris" if i == 0 else "")

        ax.plot(self.goal_pos[0], self.goal_pos[1], "g*", markersize=12, label="Goal")

        circle_c = Circle(
            (float(self.goal_pos[0]), float(self.goal_pos[1])),
            self.goal_threshold,
            color="green",
            fill=False,
            linestyle="--",
            linewidth=0.8,
        )
        ax.add_patch(circle_c)

        ax.set_xlim(-self.boundary, self.boundary)
        ax.set_ylim(-self.boundary, self.boundary)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_title("Space Debris Avoidance — RL Agent")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.02)

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close("all")
