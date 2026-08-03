"""Compose the GitHub social-preview card (1280x640) from real project imagery.

Left: a live frame of the v3 agent mid-episode on a held-out level. Right:
the actual 84x84 observation and its Grad-CAM, reusing grad_cam()/overlay()
from saliency_maps.py so the card can never disagree with the README figures.
Bottom strip: the neuron mark and the title. No success-rate numbers on
purpose: those change with re-measurement, a preview card should not.

The card is uploaded by hand at GitHub Settings -> General -> Social preview
(there is no API for it). Fonts come from the Windows font folder with a
graceful fallback, so the exact text rendering is machine-dependent; the
game imagery is deterministic (fixed model, seed and step).

Run:  python scripts/make_social_preview.py [--seed 10000 --difficulty hard]
      [--step 150] [--out assets/social_preview.png]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv
from saliency_maps import grad_cam, overlay

ROOT = pathlib.Path(__file__).resolve().parents[1]

W, H = 1280, 640
GAME_W, GAME_H = 960, 540          # native env render size
BG = (24, 28, 40)
PANEL_BG = (17, 20, 30)
FG = (226, 230, 240)
DIM = (150, 158, 180)
ACCENT = (110, 200, 255)


def font(size, bold=False):
    names = (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/" + name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def neuron_mark(draw, cx, cy, r):
    # A soma with radiating dendrites ending in small nodes: the wordmark's dot.
    rng = np.random.default_rng(7)          # fixed layout, not decoration noise
    for ang in np.linspace(0, 2 * np.pi, 7)[:-1] + 0.3:
        length = r * (1.55 + 0.35 * rng.random())
        ex, ey = cx + length * np.cos(ang), cy + length * np.sin(ang)
        draw.line((cx, cy, ex, ey), fill=ACCENT, width=3)
        draw.ellipse((ex - 4, ey - 4, ex + 4, ey + 4), fill=ACCENT)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ACCENT)
    draw.ellipse((cx - r + 5, cy - r + 5, cx + r - 5, cy + r - 5), fill=(235, 245, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "ppo_neuron_platformer_v3_curriculum.zip"))
    ap.add_argument("--difficulty", default="hard")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--step", type=int, default=150, help="episode step to photograph")
    ap.add_argument("--out", default=str(ROOT / "assets" / "social_preview.png"))
    args = ap.parse_args()

    # Roll the deterministic episode forward to the chosen step
    model = PPO.load(args.model, device="cpu")
    make = lambda: NeuronPlatformerEnv(render_mode="rgb_array", difficulty=args.difficulty,
                                       seed=args.seed, observation_mode="rgb")
    venv = VecFrameStack(DummyVecEnv([make]), 4)
    base_env = venv.unwrapped.envs[0]
    obs = venv.reset()
    for _ in range(args.step):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = venv.step(act)
        if dones[0]:
            raise SystemExit("episode ended before --step; pick an earlier step")
    frame = base_env.render()
    cam, action, prob = grad_cam(model.policy, obs)
    current = obs[0, :, :, -3:]
    venv.close()

    # Compose the card. The game photograph is zoomed onto the action band
    # (aspect preserved) so the card is not half empty sky at thumbnail size;
    # the CNN-input panel keeps showing the true full observation.
    card = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(card)
    photo = Image.fromarray(frame).crop((178, 105, 925, 525)).resize((GAME_W, GAME_H), Image.LANCZOS)
    card.paste(photo, (0, 0))

    # Right column: what the CNN sees, and where it looks
    draw.rectangle((GAME_W, 0, W, H), fill=PANEL_BG)
    img_w = 230
    x = GAME_W + (W - GAME_W - img_w) // 2
    y = 12
    small = font(17)
    for caption, arr in (("CNN input 84x84", current),
                         ("Grad-CAM: where it looks", overlay(current, cam))):
        draw.text((x, y), caption, font=small, fill=DIM)
        card.paste(Image.fromarray(arr).resize((img_w, img_w), Image.NEAREST), (x, y + 24))
        y += 24 + img_w + 16
    draw.text((x, y + 2), "trained PPO policy,\nheld-out level", font=small, fill=DIM)

    # Bottom strip: mark, wordmark, tagline
    draw.rectangle((0, GAME_H, GAME_W, H), fill=BG)
    neuron_mark(draw, 62, GAME_H + 50, 17)
    title_f, tag_f = font(40, bold=True), font(19)
    draw.text((118, GAME_H + 10), "Neuron Platformer RL", font=title_f, fill=FG)
    draw.text((120, GAME_H + 64),
              "a neural net learns a platformer from raw pixels  -  PPO  -  curriculum  -  Grad-CAM",
              font=tag_f, fill=DIM)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    card.save(out)
    print("%s seed %d step %d action=%d p=%.2f -> %s"
          % (args.difficulty, args.seed, args.step, action, prob, out))


if __name__ == "__main__":
    main()
