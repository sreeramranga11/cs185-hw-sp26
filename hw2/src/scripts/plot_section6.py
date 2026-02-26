import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt


def get_hw2_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_exp_names(raw_names: str):
    names = [name.strip() for name in raw_names.split(",")]
    return [name for name in names if name]


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
            x_raw = row.get(x_key)
            y_raw = row.get(y_key)
            if not x_raw or not y_raw:
                continue
            x_vals.append(float(x_raw))
            y_vals.append(float(y_raw))
    if not x_vals:
        raise ValueError(f"No valid points found in {csv_path} for {x_key}/{y_key}")
    return x_vals, y_vals


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="InvertedPendulum-v4")
    parser.add_argument(
        "--exp_root", type=str, default=os.path.join(get_hw2_root(), "exp")
    )
    parser.add_argument(
        "--exp_names",
        type=str,
        default="pendulum_default,pendulum_best",
        help=(
            "Comma-separated experiment names for the plot, "
            "e.g. pendulum_default,pendulum_best"
        ),
    )
    parser.add_argument("--x_key", type=str, default="Train_EnvstepsSoFar")
    parser.add_argument("--y_key", type=str, default="Eval_AverageReturn")
    parser.add_argument(
        "--plot_path",
        type=str,
        default=os.path.join(get_hw2_root(), "section6_pendulum_compare.png"),
    )
    return parser.parse_args(args=args)


def main(args):
    exp_names = parse_exp_names(args.exp_names)

    plt.figure(figsize=(8, 5))
    for exp_name in exp_names:
        csv_path = find_latest_log_csv(args.exp_root, args.env_name, exp_name)
        x_vals, y_vals = load_curve(csv_path, args.x_key, args.y_key)
        plt.plot(x_vals, y_vals, label=exp_name)
        print(f"{exp_name}: using {csv_path}")

    plt.xlabel(args.x_key)
    plt.ylabel(args.y_key)
    plt.title("InvertedPendulum-v4 Section 6: Default vs Tuned")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=200)
    plt.close()
    print(f"Wrote plot: {args.plot_path}")


if __name__ == "__main__":
    args = setup_arguments()
    main(args)
