import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class CurveRun:
    label: str
    run_dir: Path
    steps: list[int]
    success_rates: list[float]
    offline_training_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, default=Path("exp"))
    parser.add_argument("--run_dir", action="append", default=[])
    parser.add_argument("--run_group", action="append", default=[])
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--title", type=str, default="Offline-to-Online Training Curves")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve_latest_run_dir(exp_dir: Path, run_group: str) -> Path:
    group_dir = exp_dir / run_group
    if not group_dir.exists():
        raise FileNotFoundError(f"Run group not found: {group_dir}")

    candidates = [path for path in group_dir.iterdir() if path.is_dir() and (path / "eval.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"No completed runs found under {group_dir}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_eval_csv(eval_csv: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    success_rates: list[float] = []
    with eval_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(float(row["step"])))
            success_rates.append(float(row["eval/success_rate"]))
    return steps, success_rates


def load_offline_training_steps(run_dir: Path) -> int:
    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    return int(flags["offline_training_steps"])


def build_runs(args: argparse.Namespace) -> list[CurveRun]:
    run_dirs = [Path(path) for path in args.run_dir]
    run_dirs.extend(resolve_latest_run_dir(args.exp_dir, run_group) for run_group in args.run_group)

    if not run_dirs:
        raise ValueError("Provide at least one --run_dir or --run_group.")

    if args.label and len(args.label) != len(run_dirs):
        raise ValueError("Number of --label values must match the number of plotted runs.")

    runs: list[CurveRun] = []
    for index, run_dir in enumerate(run_dirs):
        eval_csv = run_dir / "eval.csv"
        if not eval_csv.exists():
            raise FileNotFoundError(f"Missing eval.csv in {run_dir}")

        steps, success_rates = load_eval_csv(eval_csv)
        if not steps:
            raise ValueError(f"No evaluation points found in {eval_csv}")

        label = args.label[index] if args.label else run_dir.parent.name
        runs.append(
            CurveRun(
                label=label,
                run_dir=run_dir,
                steps=steps,
                success_rates=success_rates,
                offline_training_steps=load_offline_training_steps(run_dir),
            )
        )
    return runs


def plot_eval_curves(runs: list[CurveRun], title: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        ax.plot(
            run.steps,
            run.success_rates,
            marker="o",
            linewidth=2,
            markersize=4,
            label=run.label,
        )

    offline_steps = {run.offline_training_steps for run in runs}
    if len(offline_steps) == 1:
        boundary = offline_steps.pop()
        ax.axvline(boundary, color="black", linestyle="--", linewidth=1.5, label="offline -> online")
    else:
        for run in runs:
            ax.axvline(run.offline_training_steps, linestyle="--", linewidth=1.0, alpha=0.4)

    ax.set_title(title)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = build_runs(args)
    plot_eval_curves(runs, args.title, args.output)
    print(f"Saved {args.output}")
    for run in runs:
        print(f"  {run.label}: {run.run_dir}")


if __name__ == "__main__":
    main()
