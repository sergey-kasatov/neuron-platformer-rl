"""Plot the pixel-phase evaluation curve from EvalCallback's evaluations.npz.

Two stacked panels over training steps (never a dual-axis chart): success
rate on the 30 held-out seeds, and mean evaluation reward with a +/- std
band. Reward numbers are comparable only within this run.

Run:  python scripts/plot_pixel_eval.py [--npz logs/eval_pixels/evaluations.npz]
      [--out assets/pixel_training_curve.png]
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

BLUE = "#2563eb"      # the one series: the pixel agent (same entity, same hue)
GRAY = "#6b7280"
INK = "#1f2937"

STATE_BASELINE = 0.50   # v1 state-vector model, same 30 seeds, deterministic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(ROOT / "logs" / "eval_pixels" / "evaluations.npz"))
    ap.add_argument("--out", default=str(ROOT / "assets" / "pixel_training_curve.png"))
    args = ap.parse_args()

    d = np.load(args.npz)
    steps = d["timesteps"] / 1e6
    success = d["successes"].mean(axis=1) * 100
    rew_mean = d["results"].mean(axis=1)
    rew_std = d["results"].std(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, dpi=150)
    fig.suptitle("Pixel phase (v1.4): PPO on stacked 84x84 frames, evaluation on 30 held-out seeds",
                 fontsize=10.5, color=INK, x=0.02, ha="left")

    ax1.plot(steps, success, color=BLUE, lw=2, marker="o", ms=3.5)
    ax1.axhline(STATE_BASELINE * 100, color=GRAY, lw=1.2, ls="--")
    ax1.text(steps[-1], STATE_BASELINE * 100 + 2, "state-vector baseline 50%",
             ha="right", fontsize=8, color=GRAY)
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
        ax.grid(True, color="#e5e7eb", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"{len(steps)} eval points -> {out}")
    print(f"final: success {success[-1]:.1f}%, reward {rew_mean[-1]:.1f} +/- {rew_std[-1]:.1f}")


if __name__ == "__main__":
    main()
