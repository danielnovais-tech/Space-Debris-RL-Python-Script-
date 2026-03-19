# Space-Debris-RL-Python-Script-
Reinforcement Learning enables satellites to learn optimal maneuvers by trial and error, receiving rewards for avoiding collisions and penalties for risky moves. Applied to constellations like Starlink, AI can minimize fuel use while safely dodging space debris in real time.

## RL demo: space debris collision avoidance

The main RL demo is implemented in `space_debris_rl.py`. It defines a custom (Gym/Gymnasium) environment and trains a PPO agent (Stable-Baselines3) to reach a goal while avoiding moving debris.

**Install (typical)**

	pip install gymnasium numpy matplotlib stable-baselines3 torch

Note: `torch`/`stable-baselines3` wheels may not be available for very new Python versions yet. If installation fails, try Python 3.10–3.12.

**Run**

	python space_debris_rl.py

## Extra script: AI self-healing service simulator

This repo also includes a standalone anomaly-detection + auto-repair simulation.

**Install**

	pip install numpy matplotlib scikit-learn

**Run**

	python service_self_healing.py
