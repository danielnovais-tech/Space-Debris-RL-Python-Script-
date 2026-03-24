# Space Debris RL

Reinforcement Learning enables satellites to learn optimal maneuvers by trial and error, receiving rewards for avoiding collisions and penalties for risky moves. This repo packages two demos as an installable Python project:

- **RL demo:** 2D space-debris collision avoidance with PPO
- **Self-healing demo:** anomaly detection + automated recovery loop simulator

## Quickstart (RL demo)

### 1) Create a virtualenv

Windows (PowerShell):

	python -m venv .venv
	.\.venv\Scripts\Activate.ps1

### 2) Install

	python -m pip install --upgrade pip
	pip install -e ".[rl]"

### 3) Run

	space-debris-rl run

What happens:

- Trains for **100,000** timesteps by default.
- Evaluates **5 episodes** and renders trajectories in a Matplotlib window.

## Training vs evaluation (inference)

Train and save a model (Stable-Baselines3 saves `*.zip` automatically):

	space-debris-rl train --timesteps 100000 --model space_debris_ppo

Evaluate a saved model:

	space-debris-rl evaluate --model space_debris_ppo.zip --episodes 5

Robust evaluation (simulated radiation-induced corruption):

	space-debris-rl evaluate --model space_debris_ppo.zip --robust --obs-bitflip-p 0.001

## Hierarchical distributed demo (manager/worker)

This repo also includes a **distributed self-healing** environment where a **manager** picks a discrete *strategy* and a **worker** produces low-level multi-node actions.

### New capabilities

- **Hierarchical training** (`train --hierarchical`) trains a manager policy that selects high-level strategies (e.g., noop, restart, scale up) on the distributed service environment.
- Optional worker training (`--train-worker`) learns a worker policy that translates strategies into low-level actions (otherwise a deterministic mapping is used).
- Supports customizing the number of nodes (`--nodes`) and strategies (`--strategies`).
- Robust mode (`--robust`) wraps the environment in `RobustEnv` with observation bit-flip injection and watchdog resets.

### Strategy-aware LTL constraints

- LTL-like formulas can reference the chosen strategy (example: `strategy_1_restart_less_than_3_per_hour`).
- Pass formulas via repeatable `--ltl` flags (e.g., `--ltl strategy_1_restart_less_than_3_per_hour`).
- `SafetyMonitor` enforces these constraints during both training and evaluation, vetoing actions that would violate them.

### Robust evaluation with decision logging

- `evaluate --robust --obs-bitflip-p ...` tests the agent under simulated radiation effects.
- Adding `--decision-log out/decision_log.json` produces a structured log of each step, including manager strategy context, final action taken, veto/fallback info, and watchdog resets.

### Federated manager aggregation

- `federated aggregate-manager` averages multiple manager checkpoints (e.g., from different nodes) and saves a global manager model.

### Train (hierarchical)

Recommended: use the existing `train` command with the `--hierarchical` flag (keeps the base UX intact):

	space-debris-rl train --hierarchical --timesteps 50000 --manager-model manager_ppo

Optionally also train a learned worker (otherwise the worker uses a deterministic strategy->action mapping):

	space-debris-rl train --hierarchical --timesteps 50000 --manager-model manager_ppo --train-worker --worker-model worker_ppo

Learned worker (explicit example):

	# Train manager
	space-debris-rl train --hierarchical --strategies 5 --timesteps 100000 --manager-model my_manager

	# Train worker conditioned on strategies (samples a random strategy per episode)
	space-debris-rl train --hierarchical --train-worker --worker-model my_worker \
	  --strategies 5 --timesteps 100000 --manager-model my_manager

Equivalent explicit command (same behavior as the alias above):

	space-debris-rl train-hierarchical --timesteps 50000 --manager-model manager_ppo

Useful options:

Flags:

- `--train-worker` also train the worker policy
- `--manager-model PATH` path/prefix to save the manager model
- `--worker-model PATH` path/prefix to save the worker model (requires `--train-worker`)
- `--ltl NAME` repeatable LTL-like formulas (enforced by `SafetyMonitor`; pass multiple times)
- `--robust` enable observation corruption + safety envelope behavior (including watchdog resets)
- `--obs-bitflip-p P` bitflip probability in robust mode (default: `0.001` for `train --hierarchical`)

Other knobs:

- `--strategies N` number of discrete strategies (default: 5 for `train --hierarchical`)
- `--nodes N` number of service nodes (distributed env)

### Evaluate (hierarchical)

Evaluate a trained manager+worker pair via the top-level `evaluate` command:

	space-debris-rl evaluate --hierarchical --manager-model manager_ppo.zip --worker-model worker_ppo.zip --episodes 5

Robust evaluation with decision-log export:

	space-debris-rl evaluate --hierarchical --robust --obs-bitflip-p 0.001 \
	  --manager-model manager_ppo.zip --worker-model worker_ppo.zip \
	  --decision-log out/decision_log.json

The decision log is designed for auditability and can include manager strategy context and veto reasons when the safety envelope intervenes (decision logs are emitted in robust mode).

Evaluate with a learned worker (uses strategy-conditioned observations):

	space-debris-rl evaluate --hierarchical --learned-worker \
	  --manager-model my_manager.zip --worker-model my_worker.zip \
	  --robust --obs-bitflip-p 0.001 \
	  --decision-log logs/out.json

### Federated averaging (manager-only)

Average multiple manager checkpoints into a single global manager model:

	space-debris-rl federated aggregate-manager --models manager_a.zip manager_b.zip --output manager_global.zip

Tip: use `--no-render` on headless machines.

### End-to-end checklist

Reproduce the full hierarchical flow (train -> eval -> logs -> federated aggregate):

1) Train manager-only (fast path):

	space-debris-rl train --hierarchical --timesteps 50000 --manager-model manager_ppo

2) (Optional) Train a learned worker:

	space-debris-rl train --hierarchical --timesteps 50000 \
	  --manager-model manager_ppo --train-worker --worker-model worker_ppo

3) Evaluate manager+worker (robust + decision log):

	space-debris-rl evaluate --hierarchical --episodes 5 --robust --obs-bitflip-p 0.001 \
	  --manager-model manager_ppo.zip --worker-model worker_ppo.zip \
	  --decision-log out/decision_log.json

4) Evaluate with runtime constraints (repeat `--ltl` as needed):

	space-debris-rl train --hierarchical --timesteps 50000 --robust --obs-bitflip-p 0.001 \
	  --ltl strategy_1_restart_less_than_3_per_hour \
	  --manager-model manager_ltl

5) Federated aggregation (manager-only):

	space-debris-rl federated aggregate-manager \
	  --models manager_a.zip manager_b.zip manager_c.zip \
	  --output manager_global.zip

## Quickstart (self-healing demo)

	pip install -e ".[self-healing]"
	service-self-healing

## Product‑Ready: key development areas

### 1) High‑fidelity environment

- Upgrade to 3D orbital dynamics: replace the 2D Cartesian model with a realistic propagator (e.g., SGP4) using Two‑Line Element (TLE) data for both the spacecraft and debris.
- Add perturbation models: include J2 effects, atmospheric drag, solar radiation pressure, and third‑body gravity.
- Incorporate sensor models: simulate noisy measurements of debris positions/velocities (radar, optical) with realistic update rates and uncertainties.
- Implement continuous action space: replace discrete thrust impulses with continuous thrust vectors with magnitude limits.

### 2) RL algorithm enhancements

- Multi‑objective reward shaping: balance collision probability, fuel consumption, time to manoeuvre, and compliance with coordination rules.
- Incorporate uncertainty: train with partially observable states (POMDP) using recurrent policies (e.g., LSTM) to handle measurement noise and occlusion.
- Safety‑constrained exploration: use shielding/safe‑RL techniques to avoid catastrophic actions during training.
- Transfer learning / fine‑tuning: pre‑train in simulation, then fine‑tune with higher‑fidelity models or historical encounter data.

### 3) Operational integration

- Real‑time performance: optimise inference for flight‑like hardware with latency < 1 second.
- Human‑in‑the‑loop interface: dashboard showing manoeuvres, risk metrics, and allowing human override; add explainability outputs.
- Integration with Flight Dynamics Systems: connect to orbit determination, conjunction assessment, and telemetry/telecommand.
- Compliance with space traffic management guidelines: keep‑out zones, right‑of‑way rules, coordination logic.

### 4) Validation & verification

- Extensive Monte Carlo simulations: test over thousands of encounter scenarios (varying geometry, sizes, uncertainties).
- Adversarial testing: worst‑case debris behaviour (untracked manoeuvres, fragmentation).
- Formal verification: prove policy never outputs a dangerous command within a defined operational envelope.
- Hardware‑in‑the‑loop: run real‑time simulations with simulated sensor feeds on representative flight compute.

### 5) Deployment & maintenance

- Continuous learning pipeline: retrain as debris catalogues update and after each real manoeuvre.
- Fallback modes: degrade gracefully to classical rule‑based algorithms when uncertain/out‑of‑distribution.
- Logging & auditing: record decisions and reasoning for post‑flight analysis and reporting.

### Sprint deliverables

| Theme | Deliverable |
|---|---|
| Environment | 3DOF orbital simulator with SGP4, realistic sensor noise, and configurable perturbations |
| RL Core | Trained policy handling partial observability, with safety constraints and fuel optimisation |
| Integration | API connecting the RL module to existing flight dynamics software; real‑time dashboard prototype |
| Validation | Test report covering 10^5 Monte Carlo runs and adversarial scenarios; formal bounds on collision probability |
| Documentation | Design documents, operator manual, and compliance checklist |

## Extra script: AI self-healing service simulator

This repo also includes a standalone anomaly-detection + auto-repair simulation.

This demo is now exposed via the `service-self-healing` CLI.

### Next sprint (product-ready)

**Goal:** move from a demo loop to an operator-safe, deployable prototype with real inputs, safer recovery actions, and measurable accuracy.

**Scope (next sprint)**

- Data pipeline: ingest real metrics (or recorded traces) in a consistent schema; add replay mode for deterministic testing.
- Detection: switch from single-point scoring to window/temporal features; calibrate thresholds and add basic drift monitoring.
- Recovery engine: define an action library with pre/post-conditions and a safety guard (don’t restart critical components blindly).
- Observability: structured logs + event timeline (detections, actions, recoveries), and an exportable report.
- Validation: staged fault-injection suite and a metrics report (false positives/negatives, time-to-detect, time-to-recover).

**Sprint deliverables**

| Theme | Deliverable |
|---|---|
| Data | Metric schema + replayable dataset loader |
| AI Core | Temporal anomaly scoring + calibrated thresholds |
| Recovery | Action library + safety guard + human approval mode |
| Ops | Structured logging + runbook-style report output |
| Validation | Fault-injection harness + summary report of results |

Out of scope for this sprint: full online learning, deep root-cause analysis, and production integrations (Prometheus/Grafana/PagerDuty) beyond simple adapters.

## Development

Install dev tooling:

	pip install -e ".[dev]"

Run checks:

	ruff check .
	ruff format --check .
	pytest

Enable pre-commit:

	pre-commit install

## Releases

This repo includes a GitHub Actions release workflow that runs on version tags.

- Create a tag like `v0.1.0` and push it.
- The workflow builds `sdist` + `wheel` and uploads them to a GitHub Release.

## Model artifact storage policy

The example trained model is stored as `space_debris_ppo.zip`.

For a "product" workflow, it’s usually better to store large artifacts as:

- a GitHub Release asset, or
- Git LFS, or
- downloaded at runtime (with a checksum).

If you distribute a model file, consider publishing a SHA-256 checksum. On Windows:

	certutil -hashfile space_debris_ppo.zip SHA256
