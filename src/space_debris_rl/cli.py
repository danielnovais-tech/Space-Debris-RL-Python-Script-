from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ._deps import MissingDependencyError
from .rl import evaluate, evaluate_robust, load_model, train


def decision_log_summary(entries: list[dict[str, Any]]) -> str:
    counts = {
        "steps": len(entries),
        "veto_action": 0,
        "fallback_used": 0,
        "watchdog_reset": 0,
        "obs_rejected": 0,
    }
    for e in entries:
        if e.get("veto", False):
            counts["veto_action"] += 1
        if e.get("fallback_used", False):
            counts["fallback_used"] += 1
        if e.get("watchdog_reset", False):
            counts["watchdog_reset"] += 1
        if e.get("obs_rejected", False):
            counts["obs_rejected"] += 1
    return (
        "Decision log summary: "
        f"steps={counts['steps']} "
        f"veto_action={counts['veto_action']} "
        f"fallback_used={counts['fallback_used']} "
        f"watchdog_reset={counts['watchdog_reset']} "
        f"obs_rejected={counts['obs_rejected']}"
    )


def write_decision_log_json(entries: list[dict[str, Any]], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")


def _add_distributed_subcommands(sub: argparse._SubParsersAction) -> None:
    p_dist = sub.add_parser("distributed", help="Distributed service self-healing demo")
    p_dist_sub = p_dist.add_subparsers(dest="dist_cmd")

    p_dist_train = p_dist_sub.add_parser("train", help="Train PPO on distributed env")
    p_dist_train.add_argument("--timesteps", type=int, default=50_000)
    p_dist_train.add_argument("--seed", type=int, default=0)
    p_dist_train.add_argument("--model", type=str, default="distributed_ppo")

    p_dist_eval = p_dist_sub.add_parser("evaluate", help="Evaluate PPO on distributed env")
    p_dist_eval.add_argument("--model", type=str, default="distributed_ppo.zip")
    p_dist_eval.add_argument("--episodes", type=int, default=5)
    p_dist_eval.add_argument("--robust", action="store_true")
    p_dist_eval.add_argument("--obs-bitflip-p", type=float, default=0.0)
    p_dist_eval.add_argument("--no-render", action="store_true")
    p_dist_eval.add_argument(
        "--decision-log",
        type=str,
        default=None,
        help="Write RobustEnv decision log to a JSON file (robust mode)",
    )
    p_dist.set_defaults(dist_cmd="train")


def _add_hierarchical_subcommands(sub: argparse._SubParsersAction) -> None:
    p_h = sub.add_parser("hierarchical", help="Hierarchical (manager/worker) distributed demo")
    p_h_sub = p_h.add_subparsers(dest="h_cmd")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--nodes", type=int, default=4)
    common.add_argument("--timesteps", type=int, default=50_000)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--strategies", type=int, default=3)
    common.add_argument("--robust", action="store_true")
    common.add_argument("--obs-bitflip-p", type=float, default=0.0)

    p_m = p_h_sub.add_parser("train-manager", parents=[common], help="Train manager policy")
    p_m.add_argument("--model", type=str, default="manager_ppo")

    p_w = p_h_sub.add_parser("train-worker", parents=[common], help="Train worker policy")
    p_w.add_argument("--model", type=str, default="worker_ppo")

    p_e = p_h_sub.add_parser("evaluate", parents=[common], help="Evaluate manager+worker")
    p_e.add_argument("--manager", type=str, default="manager_ppo.zip")
    p_e.add_argument("--worker", type=str, default="worker_ppo.zip")
    p_e.add_argument("--episodes", type=int, default=5)

    p_iface = p_h_sub.add_parser(
        "interface-eval",
        parents=[common],
        help="Evaluate manager(strategy)->worker(actions) interface (deterministic worker)",
    )
    p_iface.add_argument("--episodes", type=int, default=5)
    p_iface.add_argument("--manager", type=str, default=None)
    p_iface.add_argument("--fixed-strategy", type=int, default=None)
    p_iface.add_argument(
        "--ltl",
        action="append",
        default=[],
        help="Add LTL-like constraint name (repeatable)",
    )

    p_h.set_defaults(h_cmd="evaluate")


def _add_train_hierarchical(sub: argparse._SubParsersAction) -> None:
    p_th = sub.add_parser(
        "train-hierarchical",
        help="Train hierarchical manager (and optional worker) for distributed env",
    )
    p_th.add_argument("--nodes", type=int, default=4)
    p_th.add_argument("--strategies", type=int, default=4)
    p_th.add_argument("--timesteps", type=int, default=50_000)
    p_th.add_argument("--seed", type=int, default=0)
    p_th.add_argument("--robust", action="store_true")
    p_th.add_argument("--obs-bitflip-p", type=float, default=0.0)
    p_th.add_argument(
        "--ltl",
        action="append",
        default=[],
        help="Add LTL-like constraint name (repeatable)",
    )
    p_th.add_argument(
        "--manager-model",
        type=str,
        default="manager_ppo",
        help="Output path (prefix) for manager model",
    )
    p_th.add_argument(
        "--train-worker",
        action="store_true",
        help="Also train a worker policy (otherwise keep deterministic mapping)",
    )
    p_th.add_argument(
        "--worker-model",
        type=str,
        default="worker_ppo",
        help="Output path (prefix) for worker model (if --train-worker)",
    )


def _add_federated_subcommands(sub: argparse._SubParsersAction) -> None:
    p_f = sub.add_parser("federated", help="Federated averaging utilities")
    p_f_sub = p_f.add_subparsers(dest="fed_cmd")

    p_agg = p_f_sub.add_parser(
        "aggregate-manager",
        help="Average multiple manager checkpoints into a global manager model",
    )
    p_agg.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Manager model .zip files to average",
    )
    p_agg.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for averaged manager model",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="space-debris-rl", description="Space debris RL demo (train + evaluate)"
    )
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Train then evaluate (default)")
    p_run.add_argument("--timesteps", type=int, default=100_000)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--episodes", type=int, default=5)
    p_run.add_argument("--model", type=str, default="space_debris_ppo")
    p_run.add_argument("--no-render", action="store_true")

    p_train = sub.add_parser("train", help="Train PPO and save model")
    p_train.add_argument("--timesteps", type=int, default=100_000)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--model", type=str, default="space_debris_ppo")

    # Alias path: allow hierarchical training without adding clutter to base UX.
    p_train.add_argument(
        "--hierarchical",
        action="store_true",
        help="Train hierarchical manager (and optional worker) for distributed env",
    )
    p_train.add_argument("--nodes", type=int, default=4)
    p_train.add_argument("--strategies", type=int, default=5)
    p_train.add_argument("--robust", action="store_true")
    p_train.add_argument("--obs-bitflip-p", type=float, default=0.001)
    p_train.add_argument(
        "--ltl",
        action="append",
        default=[],
        help="Add LTL-like constraint name (repeatable)",
    )
    p_train.add_argument(
        "--manager-model",
        type=str,
        default="manager_ppo",
        help="Output path (prefix) for manager model (hierarchical mode)",
    )
    p_train.add_argument(
        "--train-worker",
        action="store_true",
        help="Also train a worker policy (hierarchical mode)",
    )
    p_train.add_argument(
        "--worker-model",
        type=str,
        default="worker_ppo",
        help="Output path (prefix) for worker model (hierarchical mode)",
    )

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model")
    p_eval.add_argument("--model", type=str, default="space_debris_ppo.zip")
    p_eval.add_argument("--episodes", type=int, default=5)
    p_eval.add_argument("--no-render", action="store_true")
    p_eval.add_argument(
        "--robust",
        action="store_true",
        help="Enable safety wrapper + corruption simulation + hash-based reload",
    )
    p_eval.add_argument(
        "--obs-bitflip-p",
        type=float,
        default=0.0,
        help="Probability of SEU-like bitflip per observation element (robust mode)",
    )
    p_eval.add_argument(
        "--decision-log",
        type=str,
        default=None,
        help="Write RobustEnv decision log to a JSON file (robust mode)",
    )

    p_eval.add_argument(
        "--hierarchical",
        action="store_true",
        help="Use hierarchical agent (manager + worker) instead of a single policy",
    )
    p_eval.add_argument(
        "--manager-model",
        type=str,
        default=None,
        help="Path to manager model (required for --hierarchical)",
    )
    p_eval.add_argument(
        "--worker-model",
        type=str,
        default=None,
        help="Path to worker model (required for --hierarchical)",
    )

    p.set_defaults(cmd="run")

    _add_distributed_subcommands(sub)
    _add_hierarchical_subcommands(sub)
    _add_train_hierarchical(sub)
    _add_federated_subcommands(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.cmd == "hierarchical":
            from .hierarchical_rl import (
                HierarchicalConfig,
                evaluate_hierarchical,
                evaluate_strategy_manager_worker,
                train_manager,
                train_worker,
            )

            cfg = HierarchicalConfig(
                num_nodes=int(args.nodes),
                strategy_n=int(args.strategies),
                seed=int(args.seed),
                robust=bool(args.robust),
                obs_bitflip_p=float(args.obs_bitflip_p),
            )

            if args.h_cmd == "train-manager":
                train_manager(total_timesteps=int(args.timesteps), cfg=cfg, model_path=args.model)
                return 0
            if args.h_cmd == "train-worker":
                train_worker(total_timesteps=int(args.timesteps), cfg=cfg, model_path=args.model)
                return 0
            if args.h_cmd == "evaluate":
                evaluate_hierarchical(
                    manager_model_path=args.manager,
                    worker_model_path=args.worker,
                    episodes=int(args.episodes),
                    cfg=cfg,
                )
                return 0

            if args.h_cmd == "interface-eval":
                evaluate_strategy_manager_worker(
                    episodes=int(args.episodes),
                    cfg=cfg,
                    manager_model_path=args.manager,
                    fixed_strategy=args.fixed_strategy,
                    ltl_formulas=list(args.ltl) if args.ltl else None,
                )
                return 0

        if args.cmd == "distributed":
            from .distributed_env import DistributedServiceEnv
            from .robust_env import RobustEnv, RobustEnvConfig
            from .corruption import CorruptionConfig

            from ._deps import require

            require("stable_baselines3", extra="rl", pip_name="stable-baselines3")
            from stable_baselines3 import PPO  # type: ignore
            from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

            def make_env():
                base = DistributedServiceEnv(seed=int(args.seed))
                if getattr(args, "robust", False):
                    return RobustEnv(
                        base,
                        cfg=RobustEnvConfig(
                            corruption=CorruptionConfig(obs_bitflip_p=float(args.obs_bitflip_p))
                        ),
                        seed=int(args.seed),
                    )
                return base

            env = DummyVecEnv([make_env])

            if args.dist_cmd == "train":
                model = PPO("MlpPolicy", env, verbose=1, seed=int(args.seed))
                model.learn(total_timesteps=int(args.timesteps))
                model.save(str(args.model))
                return 0

            if args.dist_cmd == "evaluate":
                model = PPO.load(str(args.model))
                # simple rollout without rendering
                for ep in range(int(args.episodes)):
                    obs = env.reset()
                    done = False
                    total = 0.0
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, _info = env.step(action)
                        total += float(reward)
                    print(f"Episode {ep + 1} total reward: {total:.2f}")

                if getattr(args, "decision_log", None):
                    try:
                        robust = env.envs[0]  # type: ignore[attr-defined]
                        log = robust.get_decision_log()  # type: ignore[attr-defined]
                    except Exception:
                        log = []
                    if log:
                        write_decision_log_json(log, str(args.decision_log))
                        print(decision_log_summary(log))
                return 0

        if args.cmd == "train-hierarchical":
            from .hierarchical_training import TrainHierarchicalArgs, train_hierarchical

            train_hierarchical(
                TrainHierarchicalArgs(
                    nodes=int(args.nodes),
                    strategies=int(args.strategies),
                    timesteps=int(args.timesteps),
                    seed=int(args.seed),
                    robust=bool(args.robust),
                    obs_bitflip_p=float(args.obs_bitflip_p),
                    ltl=list(args.ltl) if getattr(args, "ltl", None) else (),
                    manager_model=str(args.manager_model),
                    train_worker=bool(getattr(args, "train_worker", False)),
                    worker_model=str(args.worker_model),
                )
            )
            return 0

        if args.cmd == "federated":
            if args.fed_cmd == "aggregate-manager":
                from .federated import SB3FederatedAverager

                avg = SB3FederatedAverager(model_class_name="PPO")
                res = avg.average(list(args.models), out_path=str(args.output))
                if not res.ok:
                    raise ValueError(res.reason or "federated_aggregate_failed")
                print(f"Wrote averaged manager model to: {args.output}")
                return 0

        if args.cmd == "train":
            if getattr(args, "hierarchical", False):
                from .hierarchical_training import TrainHierarchicalArgs, train_hierarchical

                train_hierarchical(
                    TrainHierarchicalArgs(
                        nodes=int(args.nodes),
                        strategies=int(args.strategies),
                        timesteps=int(args.timesteps),
                        seed=int(args.seed),
                        robust=bool(getattr(args, "robust", False)),
                        obs_bitflip_p=float(getattr(args, "obs_bitflip_p", 0.001)),
                        ltl=list(args.ltl) if getattr(args, "ltl", None) else (),
                        manager_model=str(args.manager_model),
                        train_worker=bool(getattr(args, "train_worker", False)),
                        worker_model=str(args.worker_model),
                    )
                )
                return 0

            print("Training PPO agent on space debris avoidance...")
            train(total_timesteps=args.timesteps, seed=args.seed, model_path=args.model)
            return 0

        if args.cmd == "evaluate":
            if getattr(args, "hierarchical", False):
                if not getattr(args, "manager_model", None) or not getattr(args, "worker_model", None):
                    raise ValueError("--manager-model and --worker-model are required with --hierarchical")

                from .hier_eval import evaluate_hierarchical as evaluate_hierarchical_agent

                _totals, log = evaluate_hierarchical_agent(
                    manager_path=str(args.manager_model),
                    worker_path=str(args.worker_model),
                    episodes=int(args.episodes),
                    robust=bool(args.robust),
                    obs_bitflip_p=float(args.obs_bitflip_p),
                    num_nodes=4,
                    seed=0,
                )
                if getattr(args, "decision_log", None) and log:
                    write_decision_log_json(log, str(args.decision_log))
                    print(decision_log_summary(log))
                return 0

            model = load_model(Path(args.model))
            if args.robust:
                evaluate_robust(
                    model,
                    num_episodes=args.episodes,
                    render=not args.no_render,
                    obs_bitflip_p=float(args.obs_bitflip_p),
                    model_path_for_hash=Path(args.model),
                    seed=0,
                )
            else:
                evaluate(model, num_episodes=args.episodes, render=not args.no_render)
            return 0

        # run (default)
        print("Training PPO agent on space debris avoidance...")
        model = train(total_timesteps=args.timesteps, seed=args.seed, model_path=args.model)
        print("\nEvaluating trained agent...")
        evaluate(model, num_episodes=args.episodes, render=not args.no_render)
        return 0

    except MissingDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
