"""Plot the control run's evaluation curve: the v1 state agent at the pixel budget.

Same layout as plot_pixel_eval.py (two stacked panels over training steps,
never a dual axis): success rate on the 30 held-out seeds, and mean
evaluation reward with a +/- std band. The run was interrupted once and
resumed from a checkpoint, so two evaluations.npz files are stitched in
step order. Reward numbers are comparable only within this run.

Run:  python scripts/plot_state_control_eval.py
      [--npz logs/eval_state_control/evaluations.npz logs/eval_state_control_r2/evaluations.npz]
      [--out assets/state_control_curve.png]
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

BLUE = "#2563eb"      # the one series: the state agent (same entity, same hue)
GRAY = "#6b7280"
INK = "#1f2937"

PUBLISHED_V1 = 0.50        # the 2M v1 model as published, same 30 seeds, deterministic
PUBLISHED_V1_STEPS = 2.0   # its training budget, millions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+",
                    default=[str(ROOT / "logs" / "eval_state_control" / "evaluations.npz"),
                             str(ROOT / "logs" / "eval_state_control_r2" / "evaluations.npz")])
    ap.add_argument("--out", default=str(ROOT / "assets" / "state_control_curve.png"))
    args = ap.parse_args()

    # Stitch the segments in step order (the resume continues the counter).
    parts = [np.load(p) for p in args.npz]
    steps = np.concatenate([d["timesteps"] for d in parts])
    successes = np.concatenate([d["successes"] for d in parts])
    results = np.concatenate([d["results"] for d in parts])
    order = np.argsort(steps)
    steps, successes, results = steps[order] / 1e6, successes[order], results[order]
    success = successes.mean(axis=1) * 100
    rew_mean, rew_std = results.mean(axis=1), results.std(axis=1)
    best = int(rew_mean.argmax())   # EvalCallback's criterion: best mean reward

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, dpi=150)
    fig.suptitle("Control run: the v1 state agent at the pixel budget (10M), evaluation on 30 held-out seeds",
                 fontsize=10.5, color=INK, x=0.02, ha="left")

    ax1.plot(steps, success, color=BLUE, lw=2, marker="o", ms=3.5)
    ax1.axhline(PUBLISHED_V1 * 100, color=GRAY, lw=1.2, ls="--")
    # Label on the left, where the curve is still far below the line; on the
    # right it runs through the 47-67% band and the text would sit on the data.
    ax1.text(0.15, PUBLISHED_V1 * 100 + 2, "v1 as published, 50%",
             ha="left", fontsize=8, color=GRAY)
    ax1.annotate(f"best-on-eval snapshot {success[best]:.0f}%\n(64.0% on 200 seeds)",
                 (steps[best], success[best]), textcoords="offset points",
                 xytext=(14, 10), fontsize=8, color=BLUE,
                 arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax1.annotate(f"{success[-1]:.0f}%", (steps[-1], success[-1]),
                 textcoords="offset points", xytext=(-2, 8), ha="right",
                 fontsize=9, fontweight="bold", color=BLUE)
    ax1.set_ylabel("success rate, %", fontsize=9, color=INK)
    ax1.set_ylim(0, 100)

    ax2.plot(steps, rew_mean, color=BLUE, lw=2, marker="o", ms=3.5)
    ax2.fill_between(steps, rew_mean - rew_std, rew_mean + rew_std,
                     color=BLUE, alpha=0.12, lw=0)
    ax2.set_ylabel("mean eval reward (+/- std)", fontsize=9, color=INK)
    ax2.set_xlabel("training steps, millions", fontsize=9, color=INK)

    for ax in (ax1, ax2):
        ax.axvline(PUBLISHED_V1_STEPS, color=GRAY, lw=1.0, ls=":")
        ax.grid(True, color="#e5e7eb", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK)
    ax1.text(PUBLISHED_V1_STEPS + 0.08, 4, "original v1 budget", fontsize=8, color=GRAY)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"{len(steps)} eval points -> {out}")
    print(f"best-on-eval at {steps[best]:.2f}M: success {success[best]:.1f}%, reward {rew_mean[best]:.1f}")
    print(f"final: success {success[-1]:.1f}%, reward {rew_mean[-1]:.1f} +/- {rew_std[-1]:.1f}")


if __name__ == "__main__":
    main()
