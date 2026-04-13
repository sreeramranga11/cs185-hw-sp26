import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


QUESTION_TO_ALGO = {
    "q1": "sacbc",
    "q2": "iql",
    "q3": "fql",
}

ENV_LABELS = {
    "cube-single-play-singletask-task1-v0": "cube-single",
    "antsoccer-arena-navigate-singletask-task1-v0": "antsoccer",
}

QUESTION_TITLES = {
    "q1": "SAC+BC",
    "q2": "IQL",
    "q3": "FQL",
}


@dataclass
class RunLog:
    question: str
    algo: str
    env_name: str
    env_label: str
    alpha: float | None
    run_dir: Path
    eval_csv: Path
    steps: list[int]
    success_rates: list[float]

    @property
    def final_success(self) -> float:
        return self.success_rates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, default=Path("exp"))
    parser.add_argument("--out_dir", type=Path, default=Path("plots"))
    parser.add_argument("--min_alpha_curves", type=int, default=3)
    return parser.parse_args()


def extract_env_name(run_dir_name: str) -> str | None:
    for env_name in ENV_LABELS:
        if env_name in run_dir_name:
            return env_name
    return None


def extract_alpha(run_dir_name: str) -> float | None:
    matches = re.findall(r"_a(-?\d+(?:\.\d+)?)", run_dir_name)
    if not matches:
        return None
    return float(matches[-1])


def load_eval_csv(eval_csv: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    success_rates: list[float] = []
    with eval_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(float(row["step"])))
            success_rates.append(float(row["eval/success_rate"]))
    return steps, success_rates


def discover_runs(exp_dir: Path) -> list[RunLog]:
    runs: list[RunLog] = []
    for question, algo in QUESTION_TO_ALGO.items():
        question_dir = exp_dir / question
        if not question_dir.exists():
            continue

        for run_dir in sorted(question_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            eval_csv = run_dir / "eval.csv"
            if not eval_csv.exists():
                continue

            env_name = extract_env_name(run_dir.name)
            if env_name is None:
                continue

            steps, success_rates = load_eval_csv(eval_csv)
            if not steps:
                continue

            runs.append(
                RunLog(
                    question=question,
                    algo=algo,
                    env_name=env_name,
                    env_label=ENV_LABELS[env_name],
                    alpha=extract_alpha(run_dir.name),
                    run_dir=run_dir,
                    eval_csv=eval_csv,
                    steps=steps,
                    success_rates=success_rates,
                )
            )
    return runs


def choose_best_run(runs: list[RunLog]) -> RunLog:
    return max(runs, key=lambda run: (run.final_success, run.eval_csv.stat().st_mtime))


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)


def save_best_task_plot(question: str, runs: list[RunLog], out_dir: Path) -> tuple[Path, list[Path]]:
    title = QUESTION_TITLES[question]
    relevant = [run for run in runs if run.question == question]

    best_runs: list[RunLog] = []
    for env_name in ENV_LABELS:
        candidates = [run for run in relevant if run.env_name == env_name]
        if candidates:
            best_runs.append(choose_best_run(candidates))

    if not best_runs:
        raise ValueError(f"No completed runs found for {question}.")

    fig, ax = plt.subplots(figsize=(8, 5))
    used_logs: list[Path] = []
    for run in best_runs:
        ax.plot(
            run.steps,
            run.success_rates,
            marker="o",
            linewidth=2,
            markersize=4,
            label=f"{run.env_label} (alpha={run.alpha:g})" if run.alpha is not None else run.env_label,
        )
        used_logs.append(run.eval_csv)

    style_axes(ax)
    ax.set_title(f"{title}: Best Runs on Both Tasks")
    ax.legend()
    fig.tight_layout()

    output_path = out_dir / f"{question}_best_tasks.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path, used_logs


def save_alpha_sweep_plot(
    question: str,
    env_name: str,
    runs: list[RunLog],
    out_dir: Path,
    min_alpha_curves: int,
) -> tuple[Path | None, list[Path], str | None]:
    title = QUESTION_TITLES[question]
    relevant = [
        run for run in runs
        if run.question == question and run.env_name == env_name and run.alpha is not None
    ]

    best_by_alpha: dict[float, RunLog] = {}
    for run in relevant:
        current = best_by_alpha.get(run.alpha)
        if current is None or (run.final_success, run.eval_csv.stat().st_mtime) > (
            current.final_success,
            current.eval_csv.stat().st_mtime,
        ):
            best_by_alpha[run.alpha] = run

    if len(best_by_alpha) < min_alpha_curves:
        found = ", ".join(f"{alpha:g}" for alpha in sorted(best_by_alpha)) or "none"
        warning = (
            f"Skipping {question} alpha sweep for {ENV_LABELS[env_name]}: "
            f"need at least {min_alpha_curves} distinct alphas, found {found}."
        )
        return None, [run.eval_csv for run in best_by_alpha.values()], warning

    fig, ax = plt.subplots(figsize=(8, 5))
    used_logs: list[Path] = []
    for alpha in sorted(best_by_alpha):
        run = best_by_alpha[alpha]
        ax.plot(
            run.steps,
            run.success_rates,
            marker="o",
            linewidth=2,
            markersize=4,
            label=f"alpha={alpha:g}",
        )
        used_logs.append(run.eval_csv)

    style_axes(ax)
    ax.set_title(f"{title}: Alpha Sweep on {ENV_LABELS[env_name]}")
    ax.legend()
    fig.tight_layout()

    output_path = out_dir / f"{question}_alpha_sweep_{ENV_LABELS[env_name]}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path, used_logs, None


def write_manifest(manifest_path: Path, sections: list[tuple[str, list[str]]]) -> None:
    lines: list[str] = []
    for title, entries in sections:
        lines.append(f"{title}:")
        if entries:
            lines.extend(entries)
        else:
            lines.append("  (none)")
        lines.append("")
    manifest_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.exp_dir)
    if not runs:
        raise SystemExit(f"No completed eval logs found under {args.exp_dir}.")

    manifest_sections: list[tuple[str, list[str]]] = []

    for question in QUESTION_TO_ALGO:
        output_path, used_logs = save_best_task_plot(question, runs, args.out_dir)
        print(f"Saved {output_path}")
        manifest_sections.append(
            (
                f"{question} best-task plot logs",
                [f"  {log}" for log in used_logs],
            )
        )

    for question in ("q1", "q2"):
        output_path, used_logs, warning = save_alpha_sweep_plot(
            question=question,
            env_name="cube-single-play-singletask-task1-v0",
            runs=runs,
            out_dir=args.out_dir,
            min_alpha_curves=args.min_alpha_curves,
        )
        if warning is not None:
            print(warning)
            manifest_sections.append(
                (
                    f"{question} alpha-sweep logs found",
                    [f"  {log}" for log in used_logs],
                )
            )
        else:
            assert output_path is not None
            print(f"Saved {output_path}")
            manifest_sections.append(
                (
                    f"{question} alpha-sweep plot logs",
                    [f"  {log}" for log in used_logs],
                )
            )

    manifest_path = args.out_dir / "plot_manifest.txt"
    write_manifest(manifest_path, manifest_sections)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
