"""Record the pixel agent into a GIF: gameplay + what-it-sees + Grad-CAM.

Left: the human-facing Kenney frame. Right panel, stacked: the actual 84x84
observation the CNN receives, and the Grad-CAM overlay showing where it
looks, with the chosen action underneath. Reuses grad_cam()/overlay() from
saliency_maps.py, so the GIF and the static figure can never disagree.

Run:  python scripts/record_pixel_agent_gif.py [--difficulty hard --seed 10000]
      [--model models/ppo_neuron_platformer_v3_curriculum.zip]
      [--out assets/demo_pixel_agent.gif] [--recurrent]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from neuron_platformer_rl.envs.platformer_env import ACTIONS, NeuronPlatformerEnv
from saliency_maps import grad_cam, overlay

ROOT = pathlib.Path(__file__).resolve().parents[1]

PANEL_IMG = 238          # right-panel image size: 2 images + captions fit 540 px
PAD, CAP = 6, 15
BG = (24, 28, 40)
FG = (226, 230, 240)
DIM = (150, 158, 180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "ppo_neuron_platformer_v3_curriculum.zip"))
    ap.add_argument("--difficulty", default="hard")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--out", default=str(ROOT / "assets" / "demo_pixel_agent.gif"))
    ap.add_argument("--every", type=int, default=5, help="record every Nth step")
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--recurrent", action="store_true",
                    help="v4 LSTM model: single frames, hidden state threaded")
    args = ap.parse_args()

    model = (RecurrentPPO if args.recurrent else PPO).load(args.model, device="cpu")
    make = lambda: NeuronPlatformerEnv(render_mode="rgb_array", difficulty=args.difficulty,
                                       seed=args.seed, observation_mode="rgb")
    venv = DummyVecEnv([make])
    if not args.recurrent:
        venv = VecFrameStack(venv, 4)          # the LSTM replaces the stack
    base_env = venv.unwrapped.envs[0]

    panel_w = PANEL_IMG + 2 * PAD
    canvas_w, canvas_h = 960 + panel_w, 540
    frames = []
    obs = venv.reset()
    done, steps = False, 0
    info = {}
    state, episode_start = None, np.ones((1,), dtype=bool)
    while not done:
        record = steps % args.every == 0
        # predict runs every step even when no frame is recorded: for the
        # recurrent policy it is what advances the hidden state.
        act, next_state = model.predict(obs, state=state, episode_start=episode_start,
                                        deterministic=True)
        if record:
            cam, action, prob = grad_cam(model.policy, obs, state, episode_start)
            current = obs[0, :, :, -3:]
            canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
            canvas.paste(Image.fromarray(base_env.render()), (0, 0))
            view = Image.fromarray(current).resize((PANEL_IMG, PANEL_IMG), Image.NEAREST)
            heat = Image.fromarray(overlay(current, cam)).resize((PANEL_IMG, PANEL_IMG), Image.NEAREST)
            draw = ImageDraw.Draw(canvas)
            x = 960 + PAD
            draw.text((x, 2), "CNN input 84x84", fill=DIM)
            canvas.paste(view, (x, CAP))
            draw.text((x, CAP + PANEL_IMG + 2), "Grad-CAM: where it looks", fill=DIM)
            canvas.paste(heat, (x, 2 * CAP + PANEL_IMG))
            draw.text((x, 2 * CAP + 2 * PANEL_IMG + 4),
                      "%s  p=%.2f" % (ACTIONS[action], prob), fill=FG)
            draw.text((x, 2 * CAP + 2 * PANEL_IMG + 18),
                      "%s seed %d" % (args.difficulty, args.seed), fill=DIM)
            frames.append(canvas)
            act = np.array([action])
        obs, _, dones, infos = venv.step(act)
        state, episode_start = next_state, np.zeros((1,), dtype=bool)
        done = bool(dones[0])
        info = infos[0]
        steps += 1
    venv.close()

    size = (int(canvas_w * args.scale), int(canvas_h * args.scale))
    imgs = [f.resize(size, Image.LANCZOS) for f in frames]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=int(args.every * 16.7), loop=0, optimize=True)
    print("seed=%d difficulty=%s success=%s x=%d steps=%d frames=%d -> %s" %
          (args.seed, args.difficulty, info.get("success"), info.get("x", -1),
           steps, len(imgs), out))


if __name__ == "__main__":
    main()
