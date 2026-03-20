from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._deps import MissingDependencyError
from .rl import evaluate, load_model, train


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

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model")
    p_eval.add_argument("--model", type=str, default="space_debris_ppo.zip")
    p_eval.add_argument("--episodes", type=int, default=5)
    p_eval.add_argument("--no-render", action="store_true")

    p.set_defaults(cmd="run")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.cmd == "train":
            print("Training PPO agent on space debris avoidance...")
            train(total_timesteps=args.timesteps, seed=args.seed, model_path=args.model)
            return 0

        if args.cmd == "evaluate":
            model = load_model(Path(args.model))
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
