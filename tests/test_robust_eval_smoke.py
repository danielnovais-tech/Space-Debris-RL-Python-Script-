import numpy as np

from space_debris_rl.env import SpaceDebrisAvoidanceEnv
from space_debris_rl.robust_env import RobustEnv


def test_robust_env_reset_and_step_smoke():
    env = RobustEnv(SpaceDebrisAvoidanceEnv(), seed=123)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert info == {}

    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert isinstance(obs2, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info2, dict)
