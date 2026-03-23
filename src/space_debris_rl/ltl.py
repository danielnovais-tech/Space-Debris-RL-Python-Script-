from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LTLResult:
    ok: bool
    reason: str | None = None


class LTLMonitor:
    """Very small, pattern-based temporal constraint monitor.

    This is *not* a full LTL engine. It's a pragmatic monitor for a few
    certification-friendly rules like rate limits.
    """

    def __init__(self, formulas: list[str] | None = None, *, max_history: int = 10_000):
        self.formulas = list(formulas or [])
        self.max_history = int(max_history)
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history = []

    def update(self, state: dict, *, strategy: int | None = None) -> None:
        s = dict(state)
        if strategy is not None:
            s["strategy"] = int(strategy)
        self.history.append(s)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def check(self, state: dict, *, strategy: int | None = None) -> LTLResult:
        self.update(state, strategy=strategy)

        for formula in self.formulas:
            if formula == "always_restart_less_than_3_per_hour":
                # Interpreting 1 step ~= 1 second.
                window = self.history[-3600:]
                restarts = sum(1 for s in window if s.get("action_taken") == "restart")
                if restarts >= 3:
                    return LTLResult(False, f"Exceeded {restarts} restarts in last hour")

            elif formula == "strategy_1_restart_less_than_3_per_hour":
                if strategy is None or int(strategy) != 1:
                    continue
                window = self.history[-3600:]
                restarts = sum(1 for s in window if s.get("action_taken") == "restart")
                if restarts >= 3:
                    return LTLResult(False, f"Strategy 1 exceeded {restarts} restarts in last hour")

            elif formula == "never_restart_all_nodes_at_once":
                # Expect action_taken to optionally hold per-node actions.
                acts = state.get("actions")
                if isinstance(acts, (list, tuple)) and len(acts) > 0:
                    if all(a == 1 for a in acts):
                        return LTLResult(False, "Attempted restart of all nodes")

        return LTLResult(True, None)
