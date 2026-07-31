"""Live policy introspection panel: what the agent sees, thinks and does.

Renders, next to the game frame, the 19 input features, the actor's hidden
activations, the action probability distribution and the value estimate for
the current observation. A full weight-graph visualisation only stays
readable for toy networks; for a 19-64-64-6 MLP the standard approach is
exactly this: feature bars, activation strips and output probabilities.

Assumes an SB3 ActorCriticPolicy MLP with the default Tanh activations and
a flatten features extractor (identity for a Box(19,) observation).
"""
from __future__ import annotations

import numpy as np
import pygame
import torch

FEATURES = [
    "x", "y", "vx", "vy", "ground", "flag_dx", "coins", "kills",
    "enem_dx", "enem_dy", "enemy", "coin_dx", "coin_dy", "coin",
    "edge", "gap_w", "gap_dy", "time", "bias",
]
ACTIONS = ["NOOP", "LEFT", "RIGHT", "JUMP", "R+JMP", "L+JMP"]

BG = (24, 28, 40)
FG = (226, 230, 240)
DIM = (120, 128, 150)
GRID = (44, 50, 68)
POS = (64, 200, 190)     # teal: positive values
NEG = (235, 140, 70)     # orange: negative values
HI = (255, 214, 60)      # chosen action


def _blend(base, col, k):
    k = max(0.15, min(1.0, k))
    return tuple(int(b + (c - b) * k) for b, c in zip(base, col))


class BrainPanel:
    def __init__(self, model, width: int = 380, height: int = 540):
        self.model = model
        self.w, self.h = width, height
        if not pygame.font.get_init():
            pygame.font.init()
        self.f_big = pygame.font.SysFont("Consolas", 20, bold=True)
        self.f_small = pygame.font.SysFont("Consolas", 13)
        self.surf = pygame.Surface((width, height))

    @torch.no_grad()
    def introspect(self, obs) -> dict:
        p = self.model.policy
        t, _ = p.obs_to_tensor(np.asarray(obs))
        x = t.float()
        acts = []
        for layer in p.mlp_extractor.policy_net:
            x = layer(x)
            if isinstance(layer, torch.nn.Tanh):
                acts.append(x.squeeze(0).cpu().numpy())
        probs = torch.softmax(p.action_net(x), dim=-1).squeeze(0).cpu().numpy()
        return {
            "probs": probs,
            "acts": acts,
            "value": float(p.predict_values(t).squeeze()),
            "action": int(np.argmax(probs)),
        }

    def draw(self, obs) -> pygame.Surface:
        d = self.introspect(obs)
        s = self.surf
        s.fill(BG)
        y = 8
        s.blit(self.f_big.render("POLICY MONITOR", True, FG), (14, y))
        y += 32

        # input features as signed bars around a zero line
        s.blit(self.f_small.render("INPUT: 19 features [-1..1]", True, DIM), (14, y))
        y += 18
        bar_x, bar_w = 104, self.w - 104 - 16
        mid = bar_x + bar_w // 2
        for name, v in zip(FEATURES, np.asarray(obs, dtype=float)):
            s.blit(self.f_small.render(name, True, FG), (14, y - 1))
            pygame.draw.rect(s, GRID, (bar_x, y, bar_w, 11))
            ln = int(min(1.0, abs(v)) * (bar_w // 2))
            if v >= 0:
                pygame.draw.rect(s, POS, (mid, y, ln, 11))
            else:
                pygame.draw.rect(s, NEG, (mid - ln, y, ln, 11))
            pygame.draw.line(s, DIM, (mid, y), (mid, y + 10))
            y += 15

        # hidden activations, one strip per layer
        y += 6
        s.blit(self.f_small.render("ACTOR HIDDEN: 2 x 64 tanh", True, DIM), (14, y))
        y += 17
        cw = (self.w - 28) / 64.0
        for a in d["acts"]:
            for i, v in enumerate(a):
                col = _blend(BG, POS if v >= 0 else NEG, abs(float(v)))
                s.fill(col, (14 + int(i * cw), y, max(1, int(cw) - 1), 12))
            y += 16

        # action distribution
        y += 8
        s.blit(self.f_small.render("ACTION probabilities", True, DIM), (14, y))
        y += 16
        aw = (self.w - 28) // 6
        bh = 56
        for i, (nm, pv) in enumerate(zip(ACTIONS, d["probs"])):
            x0 = 14 + i * aw
            col = HI if i == d["action"] else POS
            pygame.draw.rect(s, GRID, (x0, y, aw - 6, bh))
            hh = int(float(pv) * bh)
            pygame.draw.rect(s, col, (x0, y + bh - hh, aw - 6, hh))
            s.blit(self.f_small.render(nm, True, HI if i == d["action"] else DIM),
                   (x0, y + bh + 4))
        y += bh + 22

        # value estimate
        v = d["value"]
        s.blit(self.f_small.render("VALUE V(s) = %+.1f" % v, True, FG), (14, y))
        y += 18
        vmax = 200.0
        vw = int(min(1.0, abs(v) / vmax) * (self.w - 28))
        pygame.draw.rect(s, GRID, (14, y, self.w - 28, 10))
        pygame.draw.rect(s, POS if v >= 0 else NEG, (14, y, vw, 10))
        return s
