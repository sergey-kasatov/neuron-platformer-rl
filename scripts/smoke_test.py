"""Functional smoke test for the environment: 18 checks, exit 1 on failure.

Exercises the real env end to end: physics, jumping, the reward contract
(new-ground only, no back-and-forth farming), enemy stomp vs side hit,
portal, falling off, and both observation modes.

Run headless:  SDL_VIDEODRIVER=dummy python scripts/smoke_test.py
TODO: port into tests/ as pytest cases.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print("  %-44s %s %s" % (name, "PASS" if cond else "FAIL", detail))


# --- state mode: physics ----------------------------------------------------
env = NeuronPlatformerEnv(render_mode="rgb_array", difficulty="demo", seed=42,
                          observation_mode="state", debug=False)
obs, info = env.reset(seed=42)
check("state obs shape (19,)", obs.shape == (19,), obs.shape)
check("action space Discrete(6)", env.action_space.n == 6)

# settle on ground (spawn sits above the floor; truncated gravity needs ~15 frames)
for _ in range(200):
    env.step(0)
    if env.on_ground:
        break
check("gravity settles player on ground", env.on_ground, "y=%d" % env.player.y)
y0 = env.player.y

# jump: y must go up, then land back
env.step(3)
ymin = env.player.y
airborne = 0
for _ in range(120):
    env.step(0)
    ymin = min(ymin, env.player.y)
    if not env.on_ground:
        airborne += 1
    if env.on_ground:
        break
check("jump lifts the player", ymin < y0 - 60, "peak dy=%d px" % (y0 - ymin))
check("player lands back", env.on_ground, "airborne %d frames" % airborne)

# no double jump: jumping mid-air must not reset vy
env.step(3)
env.step(0)
vy_mid = env.vy
env.step(3)
check("no double jump in mid-air", env.vy >= vy_mid, "vy %.1f -> %.1f" % (vy_mid, env.vy))
for _ in range(120):
    env.step(0)
    if env.on_ground:
        break

# run right / left, and the reward contract
x0 = env.player.x
r_new_ground = 0.0
for _ in range(20):
    _, r_new_ground, *_ = env.step(2)   # breaking new ground: max_x advances
x_right = env.player.x
check("new-ground progress rewarded", r_new_ground > 0, "reward=%.3f" % r_new_ground)
for _ in range(10):
    env.step(1)
check("RIGHT moves player right", x_right > x0, "x %d -> %d" % (x0, x_right))
check("LEFT moves player left", env.player.x < x_right, "x %d -> %d" % (x_right, env.player.x))

farm = 0.0
for _ in range(30):
    _, r1, *_ = env.step(1); farm += r1
    _, r2, *_ = env.step(2); farm += r2
check("no reward farm on old ground", farm < 0, "60-step pace total=%.2f" % farm)

# enemy patrol
env2 = NeuronPlatformerEnv(render_mode="rgb_array", difficulty="demo", seed=42,
                           observation_mode="state")
env2.reset(seed=42)
e = env2.level.enemies[0]
ex = e.x
for _ in range(30):
    env2.step(0)
check("enemy patrols", e.x != ex, "x %.0f -> %.0f" % (ex, e.x))

# stomp: teleport just above a live enemy, falling; sync max_x so the
# exploration bonus cannot pollute the asserted reward
e = env2.level.enemies[0]
env2.player.x = int(e.x)
env2.player.bottom = e.y - 2
env2.max_x = env2.player.x
env2.vy = 6.0
env2.on_ground = False
_, r_stomp, term, _, _ = env2.step(0)
check("stomp kills enemy, big reward", (not e.alive) and r_stomp > 4 and not term,
      "reward=%.1f alive=%s" % (r_stomp, e.alive))
check("stomp bounces player up", env2.vy < 0, "vy=%.1f" % env2.vy)

# side collision: stand ON the platform (a mid-air teleport overlapping the
# platform solid would trigger the x-collision resolver) and walk into it
env3 = NeuronPlatformerEnv(render_mode="rgb_array", difficulty="demo", seed=42,
                           observation_mode="state")
env3.reset(seed=42)
e3 = env3.level.enemies[0]
env3.player.x = int(e3.x - 60)
env3.player.bottom = e3.y + e3.h
env3.max_x = env3.player.x
env3.vy = 0.0
term_hit, r_hit = False, 0.0
for _ in range(40):
    _, r_hit, term_hit, _, _ = env3.step(2)
    if term_hit:
        break
check("side hit terminates with penalty", term_hit and r_hit < -5, "reward=%.1f" % r_hit)

# portal ends the episode with success
env4 = NeuronPlatformerEnv(render_mode="rgb_array", difficulty="demo", seed=42,
                           observation_mode="state")
env4.reset(seed=42)
p = env4.level.portal
env4.player.x = p.x + 5
env4.player.y = p.y + 10
_, r_win, term_win, _, info_win = env4.step(0)
check("portal terminates with success", term_win and info_win["success"] and r_win > 70,
      "reward=%.1f" % r_win)

# falling off the world terminates
env5 = NeuronPlatformerEnv(render_mode="rgb_array", difficulty="demo", seed=42,
                           observation_mode="state")
env5.reset(seed=42)
env5.player.y = env5.height + 200
_, r_fall, term_fall, _, _ = env5.step(0)
check("falling off terminates with penalty", term_fall and r_fall < -5, "reward=%.1f" % r_fall)

# --- rgb mode: what the CNN will see ----------------------------------------
envr = NeuronPlatformerEnv(render_mode="rgb_array", difficulty="demo", seed=42,
                           observation_mode="rgb")
obs_r, _ = envr.reset(seed=42)
check("rgb obs shape (84,84,3)", obs_r.shape == (84, 84, 3), obs_r.shape)
check("rgb obs non-empty", obs_r.std() > 10, "std=%.1f" % obs_r.std())

for e in (env, env2, env3, env4, env5, envr):
    e.close()

failed = [r for r in results if not r[1]]
print("\n%d checks, %d failed" % (len(results), len(failed)))
sys.exit(1 if failed else 0)
