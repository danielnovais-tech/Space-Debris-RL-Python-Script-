from space_debris_rl.ltl import LTLMonitor


def test_ltl_restart_rate_limit():
    mon = LTLMonitor(["always_restart_less_than_3_per_hour"])
    # simulate 3 restarts within window
    for _ in range(2):
        ok = mon.check({"action_taken": "restart"})
        assert ok.ok
    bad = mon.check({"action_taken": "restart"})
    assert bad.ok is False
