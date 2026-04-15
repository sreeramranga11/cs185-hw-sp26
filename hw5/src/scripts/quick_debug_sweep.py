import argparse
import shlex

from scripts.run import main, setup_arguments


def make_jobs(args: argparse.Namespace) -> list[str]:
    jobs: list[str] = []

    if args.suite in {"failing_cases", "q1", "q1_antmaze"}:
        env_name = (
            "antmaze-medium-navigate-singletask-task1-v0"
            if args.suite == "q1_antmaze"
            else "antsoccer-arena-navigate-singletask-task1-v0"
        )
        for alpha in args.sacbc_alphas:
            jobs.append(
                "JOB "
                f"--run_group={args.run_group} "
                "--base_config=sacbc "
                f"--env_name={env_name} "
                f"--seed={args.seed} "
                f"--alpha={alpha} "
                f"--training_steps={args.training_steps} "
                f"--log_interval={args.log_interval} "
                f"--eval_interval={args.eval_interval} "
                f"--num_eval_trajectories={args.num_eval_trajectories} "
                f"--logdir_prefix={args.logdir_prefix} "
                f"--wandb_mode={args.wandb_mode}"
            )

    if args.suite in {"failing_cases", "q3", "q3_antmaze"}:
        if args.suite == "q3_antmaze":
            env_to_alphas = {
                "antmaze-medium-navigate-singletask-task1-v0": args.fql_antmaze_alphas,
            }
        elif args.suite == "q3":
            env_to_alphas = {
                "cube-single-play-singletask-task1-v0": args.fql_cube_alphas,
                "antsoccer-arena-navigate-singletask-task1-v0": args.fql_antsoccer_alphas,
            }
        else:
            env_to_alphas = {
                "cube-single-play-singletask-task1-v0": args.fql_cube_alphas,
                "antsoccer-arena-navigate-singletask-task1-v0": args.fql_antsoccer_alphas,
            }

        for env_name, alphas in env_to_alphas.items():
            for alpha in alphas:
                jobs.append(
                    "JOB "
                    f"--run_group={args.run_group} "
                    "--base_config=fql "
                    f"--env_name={env_name} "
                    f"--seed={args.seed} "
                    f"--alpha={alpha} "
                    f"--training_steps={args.training_steps} "
                    f"--log_interval={args.log_interval} "
                    f"--eval_interval={args.eval_interval} "
                    f"--num_eval_trajectories={args.num_eval_trajectories} "
                    f"--logdir_prefix={args.logdir_prefix} "
                    f"--wandb_mode={args.wandb_mode}"
                )

    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["failing_cases", "q1", "q3", "q1_antmaze", "q3_antmaze"],
        default="failing_cases",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training_steps", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", type=int, default=5)
    parser.add_argument("--run_group", type=str, default="debug_fix")
    parser.add_argument("--logdir_prefix", type=str, default="debug_exp")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--sacbc_alphas", type=float, nargs="+", default=[1.0, 3.0, 10.0, 30.0])
    parser.add_argument("--fql_cube_alphas", type=float, nargs="+", default=[30.0, 100.0, 300.0, 1000.0])
    parser.add_argument("--fql_antsoccer_alphas", type=float, nargs="+", default=[1.0, 3.0, 10.0, 30.0])
    parser.add_argument("--fql_antmaze_alphas", type=float, nargs="+", default=[1.0, 3.0, 10.0, 30.0])
    return parser.parse_args()


def main_script() -> None:
    args = parse_args()
    jobs = make_jobs(args)
    if not jobs:
        raise SystemExit("No jobs selected.")

    print("Running quick debug jobs:")
    for job in jobs:
        print(job)

    for job in jobs:
        job_args = setup_arguments(args=shlex.split(job)[1:])
        main(job_args)


if __name__ == "__main__":
    main_script()
