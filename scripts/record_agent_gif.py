"""Record the trained agent with the live policy panel into a GIF.

Run:  python scripts/record_agent_gif.py [--seed 10000] [--difficulty easy]
      [--out assets/demo_agent_brain.gif]

The panel's argmax action is also the action stepped into the env, so the
recording shows exactly the decision the displayed distribution produced.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pygame
from PIL import Image
from stable_baselines3 import PPO

from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv
from neuron_platformer_rl.vision.brain_panel import BrainPanel

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--model", default=str(ROOT / "models" / "ppo_neuron_platformer_v1_state.zip"))
    ap.add_argument("--out", default=str(ROOT / "assets" / "demo_agent_brain.gif"))
    ap.add_argument("--every", type=int, default=3, help="record every Nth step")
    ap.add_argument("--scale", type=float, default=0.65)
    args = ap.parse_args()

    model = PPO.load(args.model)
    env = NeuronPlatformerEnv(render_mode="rgb_array", difficulty=args.difficulty,
                              seed=args.seed, observation_mode="state")
    panel = BrainPanel(model)

    obs, info = env.reset()
    # sanity: the panel's argmax must equal SB3's deterministic prediction
    assert panel.introspect(obs)["action"] == int(model.predict(obs, deterministic=True)[0])

    frames = []
    done = False
    while not done:
        panel_surf = panel.draw(obs)
        action = panel.introspect(obs)["action"]
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        if env.steps % args.every == 0 or done:
            game = env.render()
            h, w = game.shape[0], game.shape[1]
            canvas = pygame.Surface((w + panel.w, h))
            canvas.blit(pygame.surfarray.make_surface(game.swapaxes(0, 1)), (0, 0))
            canvas.blit(panel_surf, (w, 0))
            frames.append(pygame.surfarray.array3d(canvas).swapaxes(0, 1))

    env.close()
    size = (int(frames[0].shape[1] * args.scale), int(frames[0].shape[0] * args.scale))
    imgs = [Image.fromarray(f).resize(size, Image.LANCZOS) for f in frames]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=int(args.every * 16.7), loop=0, optimize=True)
    print("seed=%d success=%s x=%d frames=%d -> %s" %
          (args.seed, info["success"], info["x"], len(imgs), out))


if __name__ == "__main__":
    main()
