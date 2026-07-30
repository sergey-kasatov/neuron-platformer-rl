"""Empirical solvability audit for the level generator.

For every consecutive pair of main-path platforms, simulate the best possible
jump with the env's exact integer physics (vx = int(4.2) = 4, v0 = -12.2,
g = 0.55, fall cap 15). A level fails if any main-path transition cannot be
cleared. Bonus ledges (platforms that start inside the span of the platform
below) are optional detours and are skipped.

Run:  python scripts/audit_solvability.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from neuron_platformer_rl.generators.level_generator import LevelGenerator

PLAYER_W = 32
SEEDS = range(1, 301)


def jump_lands(x0: int, y0: int, p2) -> bool:
    """One jump launched at x0 (player left edge), feet at y0, RIGHT held."""
    x, y = float(x0), float(y0)
    vy = -12.2
    for _ in range(400):
        vy = min(15.0, vy + 0.55)
        x += 4
        y += int(vy)
        if vy > 0:
            if y >= p2.y and x + PLAYER_W > p2.x and x < p2.x + p2.w:
                return y - int(vy) <= p2.y  # was above the floor a frame ago
            if y > p2.y + 400:
                return False
    return False


def transition_possible(p1, p2) -> bool:
    for launch in range(int(p1.x + p1.w - 1), int(p1.x + p1.w - 60), -4):
        if jump_lands(launch, p1.y, p2):
            return True
    return p2.x <= p1.x + p1.w and p2.y >= p1.y  # walk across


def main_chain(platforms):
    chain = []
    for p in sorted(platforms, key=lambda p: p.x):
        if chain and p.x < chain[-1].x + chain[-1].w:
            continue  # bonus ledge above a run, not part of the forward path
        chain.append(p)
    return chain


def main() -> int:
    gen = LevelGenerator(960, 540)
    failed = False
    print("%-8s %6s %10s %12s" % ("diff", "seeds", "bad seeds", "bad transit"))
    for diff in ("easy", "medium", "hard", "demo"):
        bad_seeds = bad_total = trans_total = 0
        for seed in SEEDS:
            lvl = gen.generate(seed, diff)
            chain = main_chain(lvl.platforms)
            bad_here = sum(
                1 for p1, p2 in zip(chain, chain[1:]) if not transition_possible(p1, p2)
            )
            trans_total += len(chain) - 1
            if bad_here:
                bad_seeds += 1
                bad_total += bad_here
        if bad_seeds:
            failed = True
        print("%-8s %6d %10s %12s" % (diff, len(SEEDS),
              "%d (%.0f%%)" % (bad_seeds, 100 * bad_seeds / len(SEEDS)),
              "%d/%d" % (bad_total, trans_total)))
    print("\n%s" % ("FAILED: unreachable transitions found" if failed else
                    "OK: every audited level is solvable"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
