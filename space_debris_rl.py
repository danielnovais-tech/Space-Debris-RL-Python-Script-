"""
RL-based automated space debris collision avoidance.

This script implements a 2D reinforcement-learning simulation where a
spacecraft must navigate from the origin to a goal position while avoiding
moving debris pieces. A PPO agent (Stable-Baselines3) is trained inside a
custom OpenAI-Gym environment and then evaluated with trajectory rendering.

Requirements
------------
    pip install gymnasium numpy matplotlib stable-baselines3 torch

Usage
-----
    python space_debris_rl.py

Customisation knobs (see SpaceDebrisAvoidanceEnv.__init__)
----------------------------------------------------------
    num_debris      – number of moving debris objects (default 2)
    thrust          – impulse magnitude per action    (default 0.5)
    max_steps       – maximum steps per episode       (default 200)
    goal_pos        – target position                 (default [10, 10])
    total_timesteps – PPO training budget             (default 100 000)

Notes
-----
* Render calls use Matplotlib; on headless systems set MPLBACKEND=Agg or
  remove the render / plt.show() calls.
* Real-world applications would require 3-D orbital dynamics, SGP4
  propagation, sensor-noise models, and safety-constrained policies.
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Use a non-interactive backend when no display is available
# ---------------------------------------------------------------------------
import os
if os.environ.get("DISPLAY") is None and sys.platform != "win32":
    matplotlib.use("Agg")

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_VERSION = "gymnasium"
except ImportError:
    import gym
    from gym import spaces
    _GYM_VERSION = "gym"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# ---------------------------------------------------------------------------
# Custom Gym / Gymnasium Environment
# ---------------------------------------------------------------------------
class SpaceDebrisAvoidanceEnv(gym.Env):
    """2-D spacecraft collision-avoidance environment.

    State
    -----
    [ x,  y,  vx,  vy,                         # spacecraft kinematics
      gx, gy,                                   # fixed goal position
      d1x, d1y, d1vx, d1vy,                    # debris 1 kinematics
      d2x, d2y, d2vx, d2vy, ... ]              # debris N kinematics

    Actions (Discrete 5)
    --------------------
    0 – no thrust
    1 – thrust +x
    2 – thrust −x
    3 – thrust +y
    4 – thrust −y

    Reward
    ------
    +1.0  goal reached
    −1.0  collision with any debris
    −0.5  spacecraft exits the bounded region
    −0.01 every time step (encourages efficiency)
    """

    metadata = {"render_modes": ["human"], "render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Simulation parameters
        self.dt = 0.1                         # integration time step (s)
        self.max_steps = 200                  # maximum steps per episode
        self.num_debris = 2                   # number of debris objects
        self.thrust = 0.5                     # thrust impulse magnitude
        self.debris_speed_range = (0.2, 0.8) # debris speed [min, max]
        self.collision_threshold = 1.0        # collision detection radius
        self.goal_threshold = 1.0             # goal arrival radius
        self.boundary = 20.0                  # ±boundary for positions

        # Fixed start and goal
        self.spacecraft_start_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.spacecraft_start_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.goal_pos = np.array([10.0, 10.0], dtype=np.float32)

        # Observation space: spacecraft (4) + goal (2) + debris (4 × N)
        obs_dim = 4 + 2 + 4 * self.num_debris
        self.observation_space = spaces.Box(
            low=-self.boundary,
            high=self.boundary,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action space: 5 discrete thrust options
        self.action_space = spaces.Discrete(5)

        # Internal state (initialised in reset)
        self.step_count: int = 0
        self.spacecraft_pos: np.ndarray = self.spacecraft_start_pos.copy()
        self.spacecraft_vel: np.ndarray = self.spacecraft_start_vel.copy()
        self.debris_pos: np.ndarray = np.zeros((self.num_debris, 2), dtype=np.float32)
        self.debris_vel: np.ndarray = np.zeros((self.num_debris, 2), dtype=np.float32)

        # Trajectory history for rendering
        self._traj: list = []

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Reset environment to a randomised initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.spacecraft_pos = self.spacecraft_start_pos.copy()
        self.spacecraft_vel = self.spacecraft_start_vel.copy()
        self._traj = [self.spacecraft_pos.copy()]

        # Random debris: position outside 5-unit exclusion zone, random velocity
        debris_pos_list = []
        debris_vel_list = []
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
        if _GYM_VERSION == "gymnasium":
            return obs, {}
        return obs  # gym (legacy) returns obs only

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Flatten all state components into a single observation vector."""
        return np.concatenate([
            self.spacecraft_pos,
            self.spacecraft_vel,
            self.goal_pos,
            self.debris_pos.flatten(),
            self.debris_vel.flatten(),
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    def step(self, action):
        """Advance the simulation by one time step.

        Parameters
        ----------
        action : int
            Discrete action index (0–4).

        Returns
        -------
        obs, reward, terminated, truncated (gymnasium) or done (gym), info
        """
        self.step_count += 1

        # Apply thrust impulse
        dv = self.thrust * self.dt
        if action == 1:
            self.spacecraft_vel[0] += dv
        elif action == 2:
            self.spacecraft_vel[0] -= dv
        elif action == 3:
            self.spacecraft_vel[1] += dv
        elif action == 4:
            self.spacecraft_vel[1] -= dv
        # action 0: coast (no thrust)

        # Euler integration
        self.spacecraft_pos = self.spacecraft_pos + self.spacecraft_vel * self.dt
        self.debris_pos = self.debris_pos + self.debris_vel * self.dt
        self._traj.append(self.spacecraft_pos.copy())

        # Check termination conditions
        collision = bool(any(
            np.linalg.norm(self.spacecraft_pos - dpos) < self.collision_threshold
            for dpos in self.debris_pos
        ))
        goal_reached = bool(
            np.linalg.norm(self.spacecraft_pos - self.goal_pos) < self.goal_threshold
        )
        out_of_bounds = bool(np.any(np.abs(self.spacecraft_pos) > self.boundary))

        # Reward shaping
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
        done = terminated or truncated  # legacy gym compatibility

        obs = self._get_obs()
        if _GYM_VERSION == "gymnasium":
            return obs, reward, terminated, truncated, {}
        return obs, reward, done, {}  # legacy gym

    # ------------------------------------------------------------------
    def render(self, mode="human"):
        """Render the current simulation state using Matplotlib."""
        plt.clf()
        ax = plt.gca()

        # Trajectory
        if len(self._traj) > 1:
            traj = np.array(self._traj)
            ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=0.8, alpha=0.5, label="Path")

        # Spacecraft
        ax.plot(
            self.spacecraft_pos[0], self.spacecraft_pos[1],
            "bo", markersize=8, label="Spacecraft",
        )

        # Debris
        for i, pos in enumerate(self.debris_pos):
            ax.plot(
                pos[0], pos[1],
                "ro", markersize=6,
                label="Debris" if i == 0 else "",
            )

        # Goal
        ax.plot(
            self.goal_pos[0], self.goal_pos[1],
            "g*", markersize=12, label="Goal",
        )

        # Collision / goal threshold circles
        circle_c = plt.Circle(
            self.goal_pos, self.goal_threshold,
            color="green", fill=False, linestyle="--", linewidth=0.8,
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

    # ------------------------------------------------------------------
    def close(self):
        """Clean up Matplotlib resources."""
        plt.close("all")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(total_timesteps: int = 100_000, seed: int = 0) -> PPO:
    """Train a PPO agent on the space debris avoidance environment.

    Parameters
    ----------
    total_timesteps : int
        Number of environment steps to train for.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    PPO
        Trained Stable-Baselines3 PPO model.
    """
    # Stable-Baselines3 requires the old gym API; wrap accordingly
    def make_env():
        env = SpaceDebrisAvoidanceEnv()
        return env

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,   # small entropy bonus to encourage exploration
    )
    model.learn(total_timesteps=total_timesteps)
    model.save("space_debris_ppo")
    print("Model saved to space_debris_ppo.zip")
    return model


# ---------------------------------------------------------------------------
# Evaluation & Visualisation
# ---------------------------------------------------------------------------
def evaluate(model: PPO, num_episodes: int = 5, render: bool = True) -> None:
    """Run evaluation episodes with the trained agent.

    Parameters
    ----------
    model : PPO
        Trained Stable-Baselines3 model.
    num_episodes : int
        Number of evaluation episodes to run.
    render : bool
        Whether to render each step with Matplotlib.
    """
    env = SpaceDebrisAvoidanceEnv()

    print(f"\nEvaluating agent over {num_episodes} episode(s)...")
    for ep in range(num_episodes):
        if _GYM_VERSION == "gymnasium":
            obs, _ = env.reset(seed=ep)
        else:
            obs = env.reset()

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if _GYM_VERSION == "gymnasium":
                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
            else:
                obs, reward, done, _ = env.step(int(action))

            total_reward += float(reward)
            steps += 1

            if render:
                env.render()

        print(f"  Episode {ep + 1:2d}: steps={steps:3d}, total_reward={total_reward:+.2f}")

    env.close()
    if render and matplotlib.get_backend() != "Agg":
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Space Debris Collision Avoidance — RL Training & Evaluation")
    print("=" * 60)

    trained_model = train(total_timesteps=100_000)

    evaluate(trained_model, num_episodes=5, render=True)
