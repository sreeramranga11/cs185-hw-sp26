import argparse
import csv
from pathlib import Path

from scripts.train_offline_online import main, setup_arguments


ENV_NAMES = {
    "cube-single": "cube-single-play-singletask-task1-v0",
    "cube-double": "cube-double-play-singletask-task1-v0",
    "antsoccer-arena": "antsoccer-arena-navigate-singletask-task1-v0",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["cube-double"],
        choices=sorted(ENV_NAMES),
        help="Use cube-double first; add antsoccer-arena once a profile has signal.",
    )
    parser.add_argument("--noise_scales", nargs="+", type=float, default=[0.3, 0.5, 0.8, 1.0, 1.2])
    parser.add_argument("--flow_steps", nargs="+", type=int, default=[10])
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[3e-4])
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline_training_steps", type=int, default=100000)
    parser.add_argument("--online_training_steps", type=int, default=20000)
    parser.add_argument("--eval_interval", type=int, default=20000)
    parser.add_argument("--log_interval", type=int, default=20000)
    parser.add_argument("--num_eval_trajectories", type=int, default=5)
    parser.add_argument("--run_group", default="debug_dsrl_sweep")
    parser.add_argument("--wandb_mode", default="offline")
    parser.add_argument("--use_prior_policy", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_jobs(args):
    jobs = []
    for env_key in args.envs:
        for noise_scale in args.noise_scales:
            for flow_steps in args.flow_steps:
                for learning_rate in args.learning_rates:
                    jobs.append(
                        [
                            "--run_group", args.run_group,
                            "--base_config", "dsrl",
                            "--env_name", ENV_NAMES[env_key],
                            "--seed", str(args.seed),
                            "--noise_scale", str(noise_scale),
                            "--flow_steps", str(flow_steps),
                            "--learning_rate", str(learning_rate),
                            "--hidden_size", str(args.hidden_size),
                            "--num_layers", str(args.num_layers),
                            "--offline_training_steps", str(args.offline_training_steps),
                            "--online_training_steps", str(args.online_training_steps),
                            "--eval_interval", str(args.eval_interval),
                            "--log_interval", str(args.log_interval),
                            "--num_eval_trajectories", str(args.num_eval_trajectories),
                            "--wandb_mode", args.wandb_mode,
                        ]
                    )
                    if args.use_prior_policy:
                        jobs[-1].append("--use_prior_policy")
    return jobs


def latest_results(run_group: str):
    rows = []
    for eval_csv in sorted(Path("exp").glob(f"{run_group}/*/eval.csv")):
        with eval_csv.open(newline="") as f:
            eval_rows = list(csv.DictReader(f))
        vals = [
            float(row["eval/success_rate"])
            for row in eval_rows
            if row.get("eval/success_rate", "") != ""
        ]
        rows.append((eval_csv.parent.name, vals[-1] if vals else None, max(vals) if vals else None))
    return rows


def main_script():
    args = parse_args()
    jobs = build_jobs(args)

    print("DSRL debug sweep jobs:")
    for job in jobs:
        print("uv run src/scripts/train_offline_online.py " + " ".join(job))

    if args.dry_run:
        return

    for job in jobs:
        main(setup_arguments(job))

    print("\nSummary:")
    for name, final, best in latest_results(args.run_group):
        print(f"{name}: final={final}, best={best}")


if __name__ == "__main__":
    main_script()
