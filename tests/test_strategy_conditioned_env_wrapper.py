import numpy as np

from space_debris_rl.distributed_env import DistributedServiceEnv
from space_debris_rl.envs import StrategyConditionedEnv


def test_strategy_conditioned_env_augments_observation():
    base = DistributedServiceEnv(num_nodes=2, max_steps=3, seed=0)
    env = StrategyConditionedEnv(base, num_strategies=4)

    obs, _info = env.reset()
    assert obs.shape == (6 + 4,)

    env.set_strategy(3)
    obs2, _reward, _terminated, _truncated, _info2 = env.step(env.action_space.sample())
    assert obs2.shape == (6 + 4,)
    assert np.allclose(obs2[-4:], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
