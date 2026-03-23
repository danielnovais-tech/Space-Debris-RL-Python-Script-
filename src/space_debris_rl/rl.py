from __future__ import annotations

from pathlib import Path
from typing import Any

from ._deps import require
from .corruption import CorruptionConfig
from .env import SpaceDebrisAvoidanceEnv
from .model_integrity import ModelIntegrityGuard
from .policy import RuleBasedFallbackPolicy
from .robust_env import RobustEnv, RobustEnvConfig


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


def evaluate_robust(
    model: Any,
    *,
    num_episodes: int = 5,
    render: bool = True,
    obs_bitflip_p: float = 0.0,
    model_path_for_hash: str | Path | None = None,
    seed: int = 0,
) -> None:
    """Evaluate with safety wrapper + optional SEU-like corruption.

    If the model artifact hash changes (simulated corruption of critical weights),
    the guard reloads from disk (watchdog-style reset to known-good checkpoint).
    """
    base_env = SpaceDebrisAvoidanceEnv()
    env = RobustEnv(
        base_env,
        cfg=RobustEnvConfig(corruption=CorruptionConfig(obs_bitflip_p=float(obs_bitflip_p))),
        seed=seed,
    )

    integrity: ModelIntegrityGuard | None = None
    if model_path_for_hash is not None:
        integrity = ModelIntegrityGuard(model_path_for_hash)
        integrity.establish_baseline()

    fallback = RuleBasedFallbackPolicy()

    if render:
        require("matplotlib", extra="rl")
        import matplotlib

    for episode in range(int(num_episodes)):
        obs, _info = env.reset()
        done = False
        total_reward = 0.0
        used_fallback = 0
        watchdog_resets = 0

        while not done:
            if integrity is not None:
                res = integrity.verify()
                if not res.ok:
                    # Reload from checkpoint.
                    model = load_model(Path(model_path_for_hash))
                    integrity.establish_baseline()
                    watchdog_resets += 1

            # Policy action
            action, _states = model.predict(obs, deterministic=True)
            action_i = int(action)

            # If wrapper flagged obs issues previously, or action is nonsensical, switch to fallback.
            # (Action bounds are enforced by RobustEnv/SafetyMonitor, but we keep fallback here
            #  to represent a separately certified safe-mode controller.)
            if action_i < 0 or action_i >= int(env.action_space.n):
                action_i = fallback.predict(obs).action
                used_fallback += 1

            obs, reward, terminated, truncated, info = env.step(action_i)
            if info.get("safety_obs_rejected"):
                action_i = fallback.predict(obs).action
                used_fallback += 1

            done = bool(terminated or truncated)
            total_reward += float(reward)
            if render:
                env.render()

        print(
            f"Episode {episode + 1} total reward: {total_reward:.2f} | "
            f"fallback_actions: {used_fallback} | watchdog_resets: {watchdog_resets}"
        )

    env.close()
    if render and matplotlib.get_backend() != "Agg":
        import matplotlib.pyplot as plt

        plt.show()
