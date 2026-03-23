import numpy as np

from space_debris_rl.strategy_worker import StrategyWorker


def test_strategy_worker_outputs_valid_actions():
    w = StrategyWorker(num_nodes=3)
    # obs = [cpu, err, mem] * 3
    obs = np.array(
        [
            10.0,
            1.0,
            20.0,
            50.0,
            9.0,
            10.0,
            90.0,
            2.0,
            99.0,
        ],
        dtype=np.float32,
    )

    out = w.act(obs, strategy=1)
    assert out.actions.shape == (3,)
    assert set(out.actions.tolist()).issubset({0, 1, 2, 3})
    assert out.actions.sum() == 1  # single node targeted

    out2 = w.act(obs, strategy=3)
    assert out2.actions.sum() == 3  # one node gets action=3
