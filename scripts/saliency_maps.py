"""Grad-CAM saliency maps for the pixel-phase CNN policy.

For sampled moments of one episode, renders three panels side by side:
the human Kenney frame, the clean 84x84 observation the CNN receives,
and a Grad-CAM heatmap over that observation showing where the network
looked when choosing its action. Same technique as a CIFAR-10 Grad-CAM:
gradients of the chosen action logit w.r.t. the last conv layer's
activations weight those activation maps into a coarse attention map.

Run:  python scripts/saliency_maps.py [--model models/pixels_best/best_model.zip]
      [--seed 10000] [--difficulty easy] [--out assets/saliency_maps.png]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from neuron_platformer_rl.envs.platformer_env import ACTIONS, NeuronPlatformerEnv

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCALE = 4          # obs panels are upscaled 84 -> 336 for readability


def grad_cam(policy, obs_batch) -> tuple[np.ndarray, int, float]:
    """Heatmap (84x84 in [0,1]), chosen action id and its probability."""
    acts: dict[str, torch.Tensor] = {}

    def save_activation(module, inputs, output):
        output.retain_grad()   # non-leaf: .grad is only kept on request
        acts["a"] = output

    conv = policy.features_extractor.cnn[4]   # last Conv2d of the Nature CNN
    hook = conv.register_forward_hook(save_activation)
    try:
        t, _ = policy.obs_to_tensor(obs_batch)
        features = policy.extract_features(t)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        logits = policy.action_net(latent_pi)
        action = int(logits.argmax(dim=1))
        prob = float(torch.softmax(logits.detach(), dim=1)[0, action])
        policy.zero_grad(set_to_none=True)
        logits[0, action].backward()
    finally:
        hook.remove()
    a = acts["a"]                              # (1, 64, 7, 7)
    w = a.grad.mean(dim=(2, 3), keepdim=True)  # channel weights: mean gradient
    cam = torch.relu((w * a).sum(dim=1))[0]
    cam = cam / cam.max() if cam.max() > 0 else cam
    cam84 = torch.nn.functional.interpolate(cam[None, None], size=(84, 84),
                                            mode="bilinear", align_corners=False)
    return cam84[0, 0].detach().cpu().numpy(), action, prob


def overlay(frame: np.ndarray, cam: np.ndarray) -> np.ndarray:
    # Blend strength follows the heat: cold regions stay the plain frame,
    # hot regions turn into the colormap. A constant-alpha blend tints the
    # whole image and hides where the attention actually is.
    heat = matplotlib.colormaps["inferno"](cam)[..., :3] * 255
    k = (0.75 * cam)[..., None]
    return ((1 - k) * frame + k * heat).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "pixels_best" / "best_model.zip"))
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--out", default=str(ROOT / "assets" / "saliency_maps.png"))
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--every", type=int, default=25, help="sample every Nth step")
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")   # gradients for one frame: CPU is enough
    make = lambda: NeuronPlatformerEnv(render_mode="rgb_array", difficulty=args.difficulty,
                                       seed=args.seed, observation_mode="rgb")
    venv = VecFrameStack(DummyVecEnv([make]), 4)
    base_env = venv.unwrapped.envs[0]

    obs = venv.reset()
    samples = []
    done, steps = False, 0
    while not done:
        if steps % args.every == 0:
            cam, action, prob = grad_cam(model.policy, obs)
            current = obs[0, :, :, -3:]        # newest frame of the stack
            samples.append(dict(step=steps, human=base_env.render(), obs=current,
                                cam=overlay(current, cam), action=action, prob=prob))
            act = np.array([action])
        else:
            act, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(act)
        done = bool(dones[0])
        steps += 1
    info = infos[0]

    keep = [samples[i] for i in
            np.linspace(0, len(samples) - 1, min(args.panels, len(samples))).astype(int)]

    # Compose the grid: one row per sampled moment, three panels per row.
    obs_w = 84 * SCALE
    hum_h = obs_w
    hum_w = int(960 / 540 * hum_h)
    pad, cap = 6, 22
    row_h = hum_h + cap + pad
    canvas = Image.new("RGB", (hum_w + 2 * (obs_w + pad), len(keep) * row_h),
                       (24, 28, 40))
    draw = ImageDraw.Draw(canvas)
    for r, s in enumerate(keep):
        y = r * row_h
        panels = [Image.fromarray(s["human"]).resize((hum_w, hum_h), Image.LANCZOS),
                  Image.fromarray(s["obs"]).resize((obs_w, obs_w), Image.NEAREST),
                  Image.fromarray(s["cam"]).resize((obs_w, obs_w), Image.NEAREST)]
        x = 0
        for img in panels:
            canvas.paste(img, (x, y))
            x += img.width + pad
        draw.text((4, y + hum_h + 4),
                  "step %d   action %s (p=%.2f)" % (s["step"], ACTIONS[s["action"]], s["prob"]),
                  fill=(226, 230, 240))
    labels = ["game frame", "agent's 84x84 view", "Grad-CAM: where the CNN looks"]
    print("episode: seed=%d success=%s x=%d, %d rows -> %s" %
          (args.seed, info.get("success"), info.get("x"), len(keep), args.out))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print("panel columns:", " | ".join(labels))
    venv.close()


if __name__ == "__main__":
    main()
