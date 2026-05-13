import argparse
import itertools
import os
import subprocess
import sys


SUITES = {
    "sanity": ["cube-single-play-singletask-task1-v0"],
    "deliverable": [
        "cube-double-play-singletask-task1-v0",
        "antsoccer-arena-navigate-singletask-task1-v0",
    ],
    "all": [
        "cube-single-play-singletask-task1-v0",
        "cube-double-play-singletask-task1-v0",
        "antsoccer-arena-navigate-singletask-task1-v0",
    ],
}


def parse_csv_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x]


def parse_csv_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(SUITES), default="sanity")
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alphas", type=str, default="30,100,300")
    parser.add_argument("--offline_data", type=str, default="0,0.1,0.25")
    parser.add_argument("--wsrl_steps", type=str, default="0,10000")
    parser.add_argument("--num_action_samples", type=str, default="32,64")
    parser.add_argument("--target_num_action_samples", type=int, default=8)
    parser.add_argument("--offline_training_steps", type=int, default=100000)
    parser.add_argument("--online_training_steps", type=int, default=20000)
    parser.add_argument("--eval_interval", type=int, default=20000)
    parser.add_argument("--log_interval", type=int, default=20000)
    parser.add_argument("--num_eval_trajectories", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    env_names = [args.env_name] if args.env_name is not None else SUITES[args.suite]

    jobs = []
    for env_name in env_names:
        for alpha, offline_data, wsrl_steps, num_samples in itertools.product(
            parse_csv_floats(args.alphas),
            parse_csv_floats(args.offline_data),
            parse_csv_ints(args.wsrl_steps),
            parse_csv_ints(args.num_action_samples),
        ):
            jobs.append(
                [
                    sys.executable,
                    "src/scripts/train_offline_online.py",
                    "--group=debug_s3_custom",
                    "--base_config=custom",
                    f"--env_name={env_name}",
                    f"--seed={args.seed}",
                    f"--alpha={alpha}",
                    f"--offline_data={offline_data}",
                    f"--wsrl_steps={wsrl_steps}",
                    f"--num_action_samples={num_samples}",
                    f"--target_num_action_samples={args.target_num_action_samples}",
                    f"--offline_training_steps={args.offline_training_steps}",
                    f"--online_training_steps={args.online_training_steps}",
                    f"--eval_interval={args.eval_interval}",
                    f"--log_interval={args.log_interval}",
                    f"--num_eval_trajectories={args.num_eval_trajectories}",
                    "--wandb_mode=offline",
                ]
            )

    if args.limit > 0:
        jobs = jobs[: args.limit]

    print("Running custom debug sweep jobs:")
    for job in jobs:
        print(" ".join(job))

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    for job in jobs:
        subprocess.run(job, check=True, env=env)

    print("\nLogs saved under exp/debug_s3_custom/")


if __name__ == "__main__":
    main()
