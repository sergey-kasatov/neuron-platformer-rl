"""Plot a curriculum run: medium and hard eval curves with stage marks.

Same layout as plot_pixel_eval.py (success panel + reward panel, one axis
each), but two series - the medium and hard EvalCallback logs - plus the
curriculum stage boundaries as vertical marks. Serves both curriculum runs;
--run selects which one, and the default reproduces the published v1.6 figure.

Run:  python scripts/plot_curriculum_eval.py [--run curriculum|lstm] [--out PATH]
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Fixed categorical assignment: medium keeps the project's series blue,
# hard gets the orange counterpart (CVD-safe pair on white).
SERIES = [("medium", "#2563eb"), ("hard", "#ea580c")]
GRAY = "#6b7280"
INK = "#1f2937"

# Stage tables mirror the STAGES constants of the two training scripts.
# The v3 curriculum starts from the v2 easy-trained weights, so its ladder
# opens on demo; the v4 LSTM run has no warm start and opens on easy.
RUNS = {
    "curriculum": dict(
        eval_dir="eval_pixels",
        out="curriculum_training_curve.png",
        title="Curriculum (v1.6): from the v2 easy-trained weights to medium and hard",
        stages=[(0.0, "demo x24"), (2.0, "medium 16 + demo 8"),
                (5.5, "hard 16 + medium 8")],
    ),
    "lstm": dict(
        eval_dir="eval_lstm",
        out="lstm_training_curve.png",
        title="Recurrence (v4): CnnLstmPolicy on single frames, curriculum from scratch",
        # Short labels: this run spans 16M, so the 4M and 6M marks sit close
        # together and the full "medium 16 + demo 8" spelling collides.
        stages=[(0.0, "easy"), (2.5, "demo"), (4.0, "medium+demo"),
                (6.0, "hard+medium")],
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=tuple(RUNS), default="curriculum")
    ap.add_argument("--out", default=None,
                    help="defaults to the run's assets/<name>_training_curve.png")
    args = ap.parse_args()
    cfg = RUNS[args.run]
    out_path = args.out or str(ROOT / "assets" / cfg["out"])

    data = {}
    for name, _ in SERIES:
        d = np.load(ROOT / "logs" / f"{cfg['eval_dir']}_{name}" / "evaluations.npz")
        data[name] = (d["timesteps"] / 1e6, d["successes"].mean(axis=1) * 100,
                      d["results"].mean(axis=1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, dpi=150)
    fig.suptitle(cfg["title"], fontsize=10.5, color=INK, x=0.02, ha="left")

    for name, colour in SERIES:
        steps, success, rew = data[name]
        ax1.plot(steps, success, color=colour, lw=2, marker="o", ms=3.5, label=name)
        ax1.annotate(f"{success[-1]:.0f}%", (steps[-1], success[-1]),
                     textcoords="offset points", xytext=(4, 4),
                     fontsize=9, fontweight="bold", color=colour)
        ax2.plot(steps, rew, color=colour, lw=2, marker="o", ms=3.5, label=name)

    for x, label in cfg["stages"]:
        for ax in (ax1, ax2):
            if x > 0:
                ax.axvline(x, color=GRAY, lw=1, ls=":")
        ax1.text(x + 0.08, 97, label, fontsize=7.5, color=GRAY, va="top")

    ax1.set_ylabel("success rate, %", fontsize=9, color=INK)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="lower right", fontsize=8, frameon=False)
    ax2.set_ylabel("mean eval reward", fontsize=9, color=INK)
    ax2.set_xlabel("training steps, millions", fontsize=9, color=INK)
    ax2.legend(loc="lower right", fontsize=8, frameon=False)

    for ax in (ax1, ax2):
        ax.grid(True, color="#e5e7eb", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    for name, _ in SERIES:
        steps, success, rew = data[name]
        print(f"{name}: {len(steps)} points, final success {success[-1]:.1f}%, "
              f"best success {success.max():.1f}%")
    print("->", out)


if __name__ == "__main__":
    main()
