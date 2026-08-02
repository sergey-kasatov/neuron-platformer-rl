"""Grad-CAM saliency maps for the pixel-phase CNN policy.

For sampled moments of one episode, renders three panels side by side:
the human Kenney frame, the clean 84x84 observation the CNN receives,
and a Grad-CAM heatmap over that observation showing where the network
looked when choosing its action. Same technique as a CIFAR-10 Grad-CAM:
gradients of the chosen action logit w.r.t. the last conv layer's
activations weight those activation maps into a coarse attention map.

Run:  python scripts/saliency_maps.py [--model models/pixels_best/best_model.zip]
      [--seed 10000] [--difficulty easy] [--out assets/saliency_maps.png]
      [--recurrent]   # v4 LSTM model
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
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from neuron_platformer_rl.envs.platformer_env import ACTIONS, NeuronPlatformerEnv

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCALE = 4          # obs panels are upscaled 84 -> 336 for readability


def grad_cam(policy, obs_batch, state=None, episode_start=None) -> tuple[np.ndarray, int, float]:
    """Heatmap (84x84 in [0,1]), chosen action id and its probability.

    A recurrent policy (v4) needs its hidden state passed in: the action
    logit then depends on the LSTM as well as on the frame, and the gradient
    flows back through the LSTM into the same conv layer. Both branches stop
    at the RAW action logits on purpose - Categorical would hand back
    normalized log-probabilities, whose gradient is not the class score the
    Grad-CAM recipe asks for.
    """
    acts: dict[str, torch.Tensor] = {}

    def save_activation(module, inputs, output):
        output.retain_grad()   # non-leaf: .grad is only kept on request
        acts["a"] = output

    # Same object as features_extractor while the extractor is shared, which
    # it is for both policies here; naming the actor's one keeps the
    # recurrent path honest if that ever changes.
    extractor = getattr(policy, "pi_features_extractor", policy.features_extractor)
    conv = extractor.cnn[4]                   # last Conv2d of the Nature CNN
    hook = conv.register_forward_hook(save_activation)
    try:
        t, _ = policy.obs_to_tensor(obs_batch)
        # The branch follows the POLICY, not the caller's state: on the first
        # step of an episode a recurrent policy has no state yet, and it starts
        # from zeros exactly like sb3-contrib's own predict does.
        if not hasattr(policy, "lstm_actor"):
            features = policy.extract_features(t)
            latent_pi = policy.mlp_extractor.forward_actor(features)
        else:
            if state is None:
                zeros = np.zeros(policy.lstm_hidden_state_shape, dtype=np.float32)
                state, episode_start = (zeros, zeros), np.ones((1,), dtype=bool)
            lstm_state = tuple(torch.as_tensor(s, dtype=torch.float32,
                                               device=policy.device) for s in state)
            starts = torch.as_tensor(episode_start, dtype=torch.float32,
                                     device=policy.device)
            features = policy.extract_features(t, policy.pi_features_extractor)
            latent_pi, _ = policy._process_sequence(features, lstm_state, starts,
                                                    policy.lstm_actor)
            latent_pi = policy.mlp_extractor.forward_actor(latent_pi)
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
    ap.add_argument("--recurrent", action="store_true",
                    help="v4 LSTM model: single frames, hidden state threaded")
    args = ap.parse_args()

    # Gradients for one frame: CPU is enough.
    model = (RecurrentPPO if args.recurrent else PPO).load(args.model, device="cpu")
    make = lambda: NeuronPlatformerEnv(render_mode="rgb_array", difficulty=args.difficulty,
                                       seed=args.seed, observation_mode="rgb")
    venv = DummyVecEnv([make])
    if not args.recurrent:
        venv = VecFrameStack(venv, 4)          # the LSTM replaces the stack
    base_env = venv.unwrapped.envs[0]

    obs = venv.reset()
    samples = []
    done, steps = False, 0
    state, episode_start = None, np.ones((1,), dtype=bool)
    while not done:
        # predict runs every step even when nothing is sampled: for the
        # recurrent policy it is what advances the hidden state.
        act, next_state = model.predict(obs, state=state, episode_start=episode_start,
                                        deterministic=True)
        if steps % args.every == 0:
            cam, action, prob = grad_cam(model.policy, obs, state, episode_start)
            current = obs[0, :, :, -3:]        # newest frame of the stack, or the only one
            samples.append(dict(step=steps, human=base_env.render(), obs=current,
                                cam=overlay(current, cam), action=action, prob=prob))
            act = np.array([action])
        obs, _, dones, infos = venv.step(act)
        state, episode_start = next_state, np.zeros((1,), dtype=bool)
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
