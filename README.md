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

Tip: use `--no-render` on headless machines.

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
