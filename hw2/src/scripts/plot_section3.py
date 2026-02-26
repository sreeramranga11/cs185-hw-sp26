import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt


SMALL_BATCH_EXPERIMENTS = [
    "cartpole",
    "cartpole_rtg",
    "cartpole_na",
    "cartpole_rtg_na",
]

LARGE_BATCH_EXPERIMENTS = [
    "cartpole_lb",
    "cartpole_lb_rtg",
    "cartpole_lb_na",
    "cartpole_lb_rtg_na",
]


def get_hw2_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_latest_log_csv(exp_root: str, env_name: str, exp_name: str) -> str:
    pattern = os.path.join(exp_root, f"{env_name}_{exp_name}_sd*", "log.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"No log.csv found for experiment '{exp_name}' with pattern: {pattern}"
        )
    return max(candidates, key=os.path.getmtime)


def load_curve(csv_path: str, x_key: str, y_key: str):
    x_vals, y_vals = [], []
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not row.get(x_key) or not row.get(y_key):
                continue
            x_vals.append(float(row[x_key]))
            y_vals.append(float(row[y_key]))
    if not x_vals:
        raise ValueError(f"No valid points found in {csv_path} for {x_key}/{y_key}")
    return x_vals, y_vals


def plot_group(
    exp_root: str,
    env_name: str,
    exp_names,
    x_key: str,
    y_key: str,
    title: str,
    output_path: str,
) -> None:
    plt.figure(figsize=(8, 5))
    for exp_name in exp_names:
        csv_path = find_latest_log_csv(exp_root, env_name, exp_name)
        x_vals, y_vals = load_curve(csv_path, x_key, y_key)
        plt.plot(x_vals, y_vals, label=exp_name)
        print(f"{exp_name}: using {csv_path}")

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote plot: {output_path}")


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v0")
    parser.add_argument("--exp_root", type=str, default=os.path.join(get_hw2_root(), "exp"))
    parser.add_argument("--x_key", type=str, default="Train_EnvstepsSoFar")
    parser.add_argument("--y_key", type=str, default="Eval_AverageReturn")
    parser.add_argument(
        "--small_plot_path",
        type=str,
        default=os.path.join(get_hw2_root(), "section3_small_batch.png"),
    )
    parser.add_argument(
        "--large_plot_path",
        type=str,
        default=os.path.join(get_hw2_root(), "section3_large_batch.png"),
    )
    return parser.parse_args(args=args)


def main(args):
    plot_group(
        exp_root=args.exp_root,
        env_name=args.env_name,
        exp_names=SMALL_BATCH_EXPERIMENTS,
        x_key=args.x_key,
        y_key=args.y_key,
        title="CartPole Section 3: Small Batch (b=1000)",
        output_path=args.small_plot_path,
    )
    plot_group(
        exp_root=args.exp_root,
        env_name=args.env_name,
        exp_names=LARGE_BATCH_EXPERIMENTS,
        x_key=args.x_key,
        y_key=args.y_key,
        title="CartPole Section 3: Large Batch (b=4000)",
        output_path=args.large_plot_path,
    )


if __name__ == "__main__":
    args = setup_arguments()
    main(args)
