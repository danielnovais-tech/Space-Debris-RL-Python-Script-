from __future__ import annotations

from pathlib import Path

from ._deps import require
from .env import SpaceDebrisAvoidanceEnv


def train(
    total_timesteps: int = 100_000, seed: int = 0, model_path: str | Path = "space_debris_ppo"
):
    """Train PPO on the environment and save the model."""
    require("stable_baselines3", extra="rl", pip_name="stable-baselines3")

    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

    def make_env():
        return SpaceDebrisAvoidanceEnv()

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
        ent_coef=0.0,
    )
    model.learn(total_timesteps=int(total_timesteps))

    model_path = Path(model_path)
    model.save(str(model_path))
    return model


def load_model(model_path: str | Path):
    require("stable_baselines3", extra="rl", pip_name="stable-baselines3")

    from stable_baselines3 import PPO  # type: ignore

    return PPO.load(str(model_path))


def evaluate(model, num_episodes: int = 5, render: bool = True) -> None:
    """Evaluate a trained agent and optionally render trajectories."""
    env = SpaceDebrisAvoidanceEnv()

    if render:
        require("matplotlib", extra="rl")
        import matplotlib

    for episode in range(int(num_episodes)):
        obs, _info = env.reset()

        done = False
        total_reward = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _info = env.step(int(action))
            done = terminated or truncated

            total_reward += float(reward)
            if render:
                env.render()

        print(f"Episode {episode + 1} total reward: {total_reward:.2f}")

    env.close()
    if render and matplotlib.get_backend() != "Agg":
        import matplotlib.pyplot as plt

        plt.show()
