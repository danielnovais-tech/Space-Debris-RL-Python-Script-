from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def sha256_array(arr: np.ndarray) -> str:
    """Stable SHA-256 over array bytes + shape + dtype."""
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def bitflip_float32(arr: np.ndarray, *, p: float, rng: np.random.Generator) -> np.ndarray:
    """Randomly flip bits in a float32 array with probability p per element.

    This simulates SEU-like corruption in telemetry or state.
    """
    if p <= 0:
        return arr

    x = np.array(arr, dtype=np.float32, copy=True)
    mask = rng.random(x.shape) < float(p)
    if not mask.any():
        return x

    # View as uint32, flip a random bit in selected elements.
    u = x.view(np.uint32)
    bit = rng.integers(0, 32, size=x.shape, dtype=np.int64).astype(np.uint32)
    flip = (np.uint32(1) << bit)
    u[mask] ^= flip[mask]
    return u.view(np.float32)


@dataclass
class CorruptionConfig:
    obs_bitflip_p: float = 0.0


class ObservationCorruptor:
    def __init__(self, cfg: CorruptionConfig, *, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))

    def corrupt(self, obs: np.ndarray) -> np.ndarray:
        return bitflip_float32(obs, p=self.cfg.obs_bitflip_p, rng=self.rng)
