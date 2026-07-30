"""Kenney "New Platformer Pack" sprite loader (CC0, see assets/kenney/License.txt).

Sprites load once and are cached. convert_alpha() needs a display surface,
which only exists in human render mode, so headless (rgb_array) callers fall
back to the raw loaded surfaces: slower to blit but identical pixels.
"""
from __future__ import annotations

from pathlib import Path

import pygame

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "kenney"
TILE = 32

_cache: dict | None = None


def _load(name: str, size: tuple[int, int] | None = None) -> pygame.Surface:
    surf = pygame.image.load(str(ASSET_DIR / (name + ".png")))
    try:
        surf = surf.convert_alpha()
    except pygame.error:
        pass  # no display yet (headless rgb_array mode)
    if surf.get_bitsize() < 24:
        # smoothscale needs 24/32-bit; force 32-bit with per-pixel alpha
        out = pygame.Surface(surf.get_size(), pygame.SRCALPHA, 32)
        out.blit(surf, (0, 0))
        surf = out
    if size:
        surf = pygame.transform.smoothscale(surf, size)
    return surf


def load_assets() -> dict:
    global _cache
    if _cache is None:
        t = (TILE, TILE)
        _cache = {
            # ground terrain (top row + dirt fill)
            "g_top": _load("terrain_grass_block_top", t),
            "g_tl": _load("terrain_grass_block_top_left", t),
            "g_tr": _load("terrain_grass_block_top_right", t),
            "g_c": _load("terrain_grass_block_center", t),
            "g_l": _load("terrain_grass_block_left", t),
            "g_r": _load("terrain_grass_block_right", t),
            # thin floating platform
            "h_l": _load("terrain_grass_horizontal_left", t),
            "h_m": _load("terrain_grass_horizontal_middle", t),
            "h_r": _load("terrain_grass_horizontal_right", t),
            # pickups and goal
            "coin": [_load("coin_gold", (24, 24)), _load("coin_gold_side", (24, 24))],
            "flag": [_load("flag_green_a", (48, 48)), _load("flag_green_b", (48, 48))],
            "hud_coin": _load("hud_coin", (28, 28)),
            # decorations
            "bush": _load("bush", (40, 40)),
            "grass": _load("grass", (32, 32)),
            "fence": _load("fence", (40, 40)),
            # actors
            "player": {
                "idle": _load("character_beige_idle", (48, 48)),
                "walk_a": _load("character_beige_walk_a", (48, 48)),
                "walk_b": _load("character_beige_walk_b", (48, 48)),
                "jump": _load("character_beige_jump", (48, 48)),
                "hit": _load("character_beige_hit", (48, 48)),
            },
            "slime": [_load("slime_normal_walk_a", t), _load("slime_normal_walk_b", t)],
            "slime_flat": _load("slime_normal_flat", t),
            # 256x256 source scaled once to screen height for a parallax backdrop.
            # The "fade" variant melts into light blue at the bottom, so pits
            # read as sky haze instead of blending with solid ground green.
            "bg": _load("background_fade_hills", (540, 540)),
        }
    return _cache
