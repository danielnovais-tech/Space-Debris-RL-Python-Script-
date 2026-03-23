import numpy as np

from space_debris_rl.distributed_env import DistributedServiceEnv


def test_distributed_env_shapes():
    env = DistributedServiceEnv(num_nodes=3, max_steps=5, seed=0)
    obs, info = env.reset()
    assert info == {}
    assert obs.shape == (9,)

    actions = np.array([0, 1, 2], dtype=np.int64)
    obs2, reward, terminated, truncated, info2 = env.step(actions)
    assert obs2.shape == (9,)
    assert isinstance(reward, float)
    assert terminated is False
    assert isinstance(truncated, bool)
    assert info2 == {}
