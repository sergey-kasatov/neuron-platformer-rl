from __future__ import annotations
from dataclasses import dataclass, field
import random

TILE = 32          # platforms are laid out on a 32 px grid so tiles render cleanly
PLAYER_W = 32
JUMP_V0 = -12.2    # must match platformer_env physics
GRAVITY = 0.55
RUN_VX = 4         # int(4.2), the env truncates
GROUND_Y = 500


@dataclass
class Platform:
    x: int
    y: int
    w: int
    h: int = 32


@dataclass
class Enemy:
    x: int
    y: int
    w: int = 32
    h: int = 32
    direction: int = -1
    speed: float = 1.0
    alive: bool = True


@dataclass
class Chip:
    x: int
    y: int
    r: int = 8
    collected: bool = False


@dataclass
class Portal:
    x: int
    y: int
    w: int = 54
    h: int = 86


@dataclass
class Decor:
    x: int          # anchor: horizontal centre
    y: int          # anchor: platform top the prop stands on
    kind: str       # bush / grass / fence


@dataclass
class Level:
    seed: int
    difficulty: str
    width: int
    height: int
    platforms: list[Platform]
    enemies: list[Enemy]
    chips: list[Chip]
    portal: Portal
    spawn_x: int = 80
    spawn_y: int = 420
    decor: list[Decor] = field(default_factory=list)


def gap_ceiling(dy: int) -> int:
    """Longest horizontal gap a jump can clear onto a floor dy px lower
    (dy < 0 = target is higher), simulated with the env's exact integer
    physics. Launch happens on the last standable pixel, so the player
    width extends the reach by almost its full 32 px.
    """
    vy, x, y = JUMP_V0, 0, 0
    for _ in range(400):
        vy = min(15.0, vy + GRAVITY)
        x += RUN_VX
        y += int(vy)
        if vy > 0 and y >= dy:
            return x + PLAYER_W - 1
    return 0  # dy above the jump apex: not reachable at all


class LevelGenerator:
    """Generates Mario-style, provably solvable platformer levels.

    Every gap is capped by gap_ceiling(dy) minus a per-difficulty safety
    margin, where dy is the height change of the jump. Difficulty tightens
    the margin (harder jumps), adds enemies and raises the share of
    floating-platform sections over pits. scripts/audit_solvability.py
    re-checks the guarantee empirically across hundreds of seeds.
    """

    SETTINGS = {
        #          level width, safety margin, floating-section share, enemies per run, enemy speeds
        "easy":   dict(width=3200, margin=64, elevated=0.22, enemy_n=(0, 1), speeds=(0.8, 1.0)),
        "demo":   dict(width=4160, margin=48, elevated=0.35, enemy_n=(0, 1), speeds=(0.8, 1.0, 1.2)),
        "medium": dict(width=4480, margin=36, elevated=0.45, enemy_n=(1, 2), speeds=(1.0, 1.2)),
        "hard":   dict(width=5440, margin=24, elevated=0.55, enemy_n=(1, 2), speeds=(1.2, 1.5)),
    }

    def __init__(self, width: int = 960, height: int = 540):
        self.screen_width = width
        self.screen_height = height

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _snap(v: int) -> int:
        return (v // TILE) * TILE

    def _pick_gap(self, rng, dy: int, margin: int, lo: int = 64) -> int:
        """A tile-aligned gap that the physics can clear with `margin` px to spare."""
        hi = self._snap(gap_ceiling(dy) - margin)
        if hi < lo:
            return 0
        return rng.randrange(lo, hi + 1, TILE)

    # -- generation ----------------------------------------------------------

    def generate(self, seed: int | None = None, difficulty: str = "easy") -> Level:
        if seed is None:
            seed = random.randint(1, 10_000_000)
        rng = random.Random(seed)
        cfg = self.SETTINGS.get(difficulty, self.SETTINGS["easy"])
        level_w = cfg["width"]

        platforms: list[Platform] = []
        enemies: list[Enemy] = []
        chips: list[Chip] = []
        decor: list[Decor] = []

        def dress_run(p: Platform, allow_enemy: bool = True):
            """Enemies, coins and props on a ground run."""
            if allow_enemy and p.w >= 192:
                for _ in range(rng.randint(*cfg["enemy_n"])):
                    ex = rng.randrange(p.x + 64, p.x + p.w - 96)
                    enemies.append(Enemy(ex, p.y - 32, speed=rng.choice(cfg["speeds"])))
            if p.w >= 160 and rng.random() < 0.8:
                kind = rng.choice(("bush", "grass", "fence", "grass"))
                decor.append(Decor(rng.randrange(p.x + 32, p.x + p.w - 32), p.y, kind))
            if p.w >= 256 and rng.random() < 0.5:
                cx = rng.randrange(p.x + 64, p.x + p.w - 64)
                for i in range(3):
                    chips.append(Chip(cx + (i - 1) * 36, p.y - 96))

        # spawn run
        start = Platform(0, GROUND_Y, 544)
        platforms.append(start)
        dress_run(start, allow_enemy=False)     # no enemy right at spawn
        x = start.w

        while x < level_w - 900:
            if rng.random() < cfg["elevated"]:
                # floating platforms over a chasm, Mario cloud-walk style
                prev_y = GROUND_Y
                px = x
                for _ in range(rng.randint(2, 3)):
                    dy = rng.randrange(-64, 33, TILE)          # up to 2 tiles up, 1 down
                    ny = max(340, min(468, prev_y + dy))
                    gap = self._pick_gap(rng, ny - prev_y, cfg["margin"], lo=64)
                    if not gap:
                        break
                    pw = rng.randrange(96, 193, TILE)
                    plat = Platform(px + gap, ny, pw)
                    platforms.append(plat)
                    for i in range(pw // 64):
                        chips.append(Chip(plat.x + 32 + i * 64, plat.y - 40))
                    if pw >= 160 and rng.random() < 0.4:
                        enemies.append(Enemy(plat.x + pw // 2, plat.y - 32,
                                             speed=rng.choice(cfg["speeds"])))
                    px = plat.x + pw
                    prev_y = ny
                # descend back to the ground
                gap = self._pick_gap(rng, GROUND_Y - prev_y, cfg["margin"], lo=64)
                run = Platform(px + gap, GROUND_Y, rng.randrange(288, 545, TILE))
                platforms.append(run)
                dress_run(run)
                x = run.x + run.w
            else:
                # plain pit with a coin arc over it
                gap = self._pick_gap(rng, 0, cfg["margin"], lo=64)
                mid = x + gap // 2
                for i, dy in enumerate((-84, -116, -84)):
                    chips.append(Chip(mid + (i - 1) * 40, GROUND_Y + dy))
                run = Platform(x + gap, GROUND_Y, rng.randrange(288, 545, TILE))
                platforms.append(run)
                dress_run(run)
                # occasional low ledge on wide runs, with a coin row
                if run.w >= 416 and rng.random() < 0.5:
                    lw = rng.randrange(96, 161, TILE)
                    lx = run.x + (run.w - lw) // 2
                    platforms.append(Platform(lx, GROUND_Y - 96, lw))
                    for i in range(lw // 48):
                        chips.append(Chip(lx + 24 + i * 48, GROUND_Y - 136))
                x = run.x + run.w

        # final stretch: a modest gap, then the flag run. A long floating
        # section can overshoot the nominal width, so the level grows to fit
        # the flag run instead of ever truncating it.
        gap = self._pick_gap(rng, 0, max(cfg["margin"], 48), lo=64)
        fx = x + gap
        final = Platform(fx, GROUND_Y, max(416, level_w - fx))
        platforms.append(final)
        dress_run(final, allow_enemy=False)
        level_w = final.x + final.w
        portal = Portal(level_w - 190, GROUND_Y - 86)

        return Level(seed, difficulty, level_w, self.screen_height,
                     platforms, enemies, chips, portal,
                     spawn_x=80, spawn_y=GROUND_Y - 50, decor=decor)
