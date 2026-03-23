from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._deps import require


@dataclass(frozen=True)
class FedResult:
    ok: bool
    reason: str | None = None


class SB3FederatedAverager:
    """Federated averaging stub for Stable-Baselines3 PPO-like models.

    SB3 doesn't expose a stable, framework-agnostic 'state_dict' API. The most
    reliable approach here is to:
      - load each node model from disk
      - average torch parameters directly
      - save a new global model

    This is a demo-quality coordinator, not a production FL system.
    """

    def __init__(self, *, model_class_name: str = "PPO"):
        self.model_class_name = model_class_name

    def average(self, model_paths: list[str | Path], *, out_path: str | Path) -> FedResult:
        require("stable_baselines3", extra="rl", pip_name="stable-baselines3")
        require("torch", extra="rl")

        import torch  # type: ignore
        from stable_baselines3 import PPO  # type: ignore

        paths = [Path(p) for p in model_paths]
        if len(paths) == 0:
            return FedResult(False, "no_models")

        models = [PPO.load(str(p)) for p in paths]
        policies = [m.policy for m in models]

        with torch.no_grad():
            # Clone first policy as accumulator.
            acc = {k: v.detach().clone() for k, v in policies[0].state_dict().items()}
            for p in policies[1:]:
                sd = p.state_dict()
                for k in acc:
                    acc[k] += sd[k].detach()
            for k in acc:
                acc[k] /= float(len(policies))

            # Write averaged weights back to first model.
            policies[0].load_state_dict(acc)

        out_path = Path(out_path)
        models[0].save(str(out_path))
        return FedResult(True, None)
