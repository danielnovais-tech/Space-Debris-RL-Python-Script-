from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque

import numpy as np

from ._deps import MissingDependencyError, require


class ServiceSimulator:
    def __init__(self):
        self.state = "normal"
        self.fault_type = None
        self.time = 0
        self.metrics_history = {"response_time": [], "error_rate": [], "cpu_usage": []}
        self.faults_injected: list[tuple[int, str]] = []
        self.healing_actions: list[tuple[int, str]] = []

    def step(self):
        self.time += 1

        if self.state == "normal":
            response_time = 50 + np.random.normal(0, 5)
            error_rate = max(0, 1 + np.random.normal(0, 0.5))
            cpu_usage = 30 + np.random.normal(0, 3)
        else:
            if self.fault_type == "high_load":
                response_time = 200 + np.random.normal(0, 30)
                error_rate = 5 + np.random.normal(0, 2)
                cpu_usage = 95 + np.random.normal(0, 5)
            elif self.fault_type == "memory_leak":
                response_time = 150 + np.random.normal(0, 20)
                error_rate = 3 + np.random.normal(0, 1)
                cpu_usage = 80 + np.random.normal(0, 10)
            else:
                response_time = 300 + np.random.normal(0, 50)
                error_rate = 15 + np.random.normal(0, 5)
                cpu_usage = 70 + np.random.normal(0, 15)

        response_time = max(10, min(500, response_time))
        error_rate = max(0, min(30, error_rate))
        cpu_usage = max(5, min(100, cpu_usage))

        metrics = [float(response_time), float(error_rate), float(cpu_usage)]
        self.metrics_history["response_time"].append(metrics[0])
        self.metrics_history["error_rate"].append(metrics[1])
        self.metrics_history["cpu_usage"].append(metrics[2])
        return metrics

    def inject_fault(self, fault_type="high_load"):
        self.state = "fault"
        self.fault_type = fault_type
        self.faults_injected.append((self.time, fault_type))
        print(f"[{self.time}] Fault injected: {fault_type}")

    def heal(self, action="restart"):
        print(f"[{self.time}] Healing action: {action}")
        self.healing_actions.append((self.time, action))

        if action in {"restart", "scale_up"}:
            self.state = "normal"
            self.fault_type = None
            return True
        return False


def generate_normal_data(sim: ServiceSimulator, steps: int = 1000) -> np.ndarray:
    data = []
    for _ in range(int(steps)):
        data.append(sim.step())
    return np.array(data, dtype=np.float32)


def run_demo(steps_normal_train: int = 1500, threshold: float = -0.1) -> int:
    require("sklearn", extra="self-healing", pip_name="scikit-learn")
    require("matplotlib", extra="self-healing")

    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest  # type: ignore

    print("=== AI Self-Healing Service Simulator ===\n")

    sim = ServiceSimulator()
    print("Generating normal-operation data...")
    normal_data = generate_normal_data(sim, steps=steps_normal_train)

    print("Training IsolationForest...")
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(normal_data)

    print("Starting realtime simulation (Ctrl+C to stop)...\n")
    sim = ServiceSimulator()

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    lines = []
    for ax, metric in zip(axes, ["response_time", "error_rate", "cpu_usage"], strict=True):
        (line,) = ax.plot([], [], lw=1)
        lines.append(line)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True)
    axes[2].set_xlabel("Time (steps)")
    plt.tight_layout()

    window_size = 50
    anomaly_scores = deque(maxlen=window_size)

    step = 0
    try:
        while True:
            step += 1
            metrics = sim.step()
            score = float(model.decision_function([metrics])[0])
            anomaly_scores.append(score)

            is_anomaly = score < float(threshold)
            if is_anomaly and sim.state == "normal":
                print(f"[{step}] Anomaly detected! Score: {score:.3f}")
                action = "scale_up" if metrics[2] > 90 else "restart"
                sim.heal(action)

            if step % 200 == 0:
                fault = random.choice(["high_load", "memory_leak"])
                sim.inject_fault(fault)

            if step % 5 == 0:
                for i, (metric_name, line) in enumerate(
                    zip(["response_time", "error_rate", "cpu_usage"], lines, strict=True)
                ):
                    data = sim.metrics_history[metric_name]
                    line.set_xdata(range(len(data)))
                    line.set_ydata(data)
                    axes[i].relim()
                    axes[i].autoscale_view()
                plt.pause(0.01)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    print("\n=== Report ===")
    print(f"Total steps: {step}")
    print(f"Faults injected: {len(sim.faults_injected)}")
    print(f"Healing actions: {len(sim.healing_actions)}")

    plt.ioff()
    plt.show()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="service-self-healing", description="AI self-healing service simulator"
    )
    p.add_argument("--train-steps", type=int, default=1500)
    p.add_argument("--threshold", type=float, default=-0.1)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return run_demo(steps_normal_train=args.train_steps, threshold=args.threshold)
    except MissingDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
