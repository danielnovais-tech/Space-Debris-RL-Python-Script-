import numpy as np

from space_debris_rl.distributed_env import DistributedServiceEnv
from space_debris_rl.strategy_conditioning import StrategyConditionedObs


def test_strategy_one_hot_appended():
    base = DistributedServiceEnv(num_nodes=2, max_steps=3, seed=0)
    env = StrategyConditionedObs(base, strategy_n=3)

    obs, _info = env.reset()
    assert obs.shape == (6 + 3,)

    env.set_strategy(2)
    obs2 = env.observation(np.zeros((6,), dtype=np.float32))
    assert obs2.shape == (9,)
    assert np.allclose(obs2[-3:], np.array([0.0, 0.0, 1.0], dtype=np.float32))
