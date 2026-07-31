"""Pytest port of scripts/smoke_test.py: the same 18 functional checks.

Every test builds a fresh env (demo difficulty, seed 42) so tests stay
independent; scripts/smoke_test.py remains as the dependency-free runner.

Run:  .venv/Scripts/python.exe -m pytest
"""
import pytest

from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv


@pytest.fixture
def env():
    e = NeuronPlatformerEnv(render_mode=None, difficulty="demo", seed=42,
                            observation_mode="state")
    e.reset(seed=42)
    yield e
    e.close()


def settle(e, max_frames=200):
    """Idle until gravity puts the player on the ground (spawn floats above it)."""
    for _ in range(max_frames):
        e.step(0)
        if e.on_ground:
            return
    raise AssertionError("player never reached the ground")


# --- state mode: spaces -----------------------------------------------------

def test_state_obs_shape(env):
    obs, _ = env.reset(seed=42)
    assert obs.shape == (19,)


def test_action_space(env):
    assert env.action_space.n == 6


# --- physics ----------------------------------------------------------------

def test_gravity_settles_player(env):
    settle(env)
    assert env.on_ground


def test_jump_lifts_and_lands(env):
    settle(env)
    y0 = env.player.y
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
    assert ymin < y0 - 60, "jump peak only %d px" % (y0 - ymin)
    assert env.on_ground, "player never landed (airborne %d frames)" % airborne


def test_no_double_jump(env):
    settle(env)
    env.step(3)
    env.step(0)
    vy_mid = env.vy
    env.step(3)   # mid-air jump press must not reset upward velocity
    assert env.vy >= vy_mid


def test_run_right_and_left(env):
    settle(env)
    x0 = env.player.x
    for _ in range(20):
        env.step(2)
    x_right = env.player.x
    assert x_right > x0
    for _ in range(10):
        env.step(1)
    assert env.player.x < x_right


# --- reward contract --------------------------------------------------------

def test_new_ground_progress_rewarded(env):
    settle(env)
    r = 0.0
    for _ in range(20):
        _, r, *_ = env.step(2)   # breaking new ground: max_x advances
    assert r > 0


def test_no_reward_farm_on_old_ground(env):
    settle(env)
    for _ in range(20):
        env.step(2)
    for _ in range(10):
        env.step(1)
    farm = 0.0
    for _ in range(30):          # pacing below max_x must never pay
        _, r1, *_ = env.step(1)
        _, r2, *_ = env.step(2)
        farm += r1 + r2
    assert farm < 0


# --- enemies ----------------------------------------------------------------

def test_enemy_patrols(env):
    e = env.level.enemies[0]
    ex = e.x
    for _ in range(30):
        env.step(0)
    assert e.x != ex


def test_stomp_kills_enemy_and_bounces(env):
    # Teleport just above a live enemy, falling; sync max_x so the
    # exploration bonus cannot pollute the asserted reward.
    e = env.level.enemies[0]
    env.player.x = int(e.x)
    env.player.bottom = e.y - 2
    env.max_x = env.player.x
    env.vy = 6.0
    env.on_ground = False
    _, r, terminated, _, _ = env.step(0)
    assert not e.alive
    assert r > 4
    assert not terminated
    assert env.vy < 0, "stomp must bounce the player up"


def test_side_hit_terminates_with_penalty(env):
    # Stand ON the platform (a mid-air teleport overlapping the platform
    # solid would trigger the x-collision resolver) and walk into the enemy.
    e = env.level.enemies[0]
    env.player.x = int(e.x - 60)
    env.player.bottom = e.y + e.h
    env.max_x = env.player.x
    env.vy = 0.0
    terminated, r = False, 0.0
    for _ in range(40):
        _, r, terminated, _, _ = env.step(2)
        if terminated:
            break
    assert terminated
    assert r < -5


# --- episode ends -----------------------------------------------------------

def test_portal_terminates_with_success(env):
    p = env.level.portal
    env.player.x = p.x + 5
    env.player.y = p.y + 10
    _, r, terminated, _, info = env.step(0)
    assert terminated
    assert info["success"]
    assert r > 70


def test_falling_off_terminates_with_penalty(env):
    env.player.y = env.height + 200
    _, r, terminated, _, _ = env.step(0)
    assert terminated
    assert r < -5


# --- rgb mode: what the CNN sees --------------------------------------------

def test_rgb_observation():
    envr = NeuronPlatformerEnv(render_mode=None, difficulty="demo", seed=42,
                               observation_mode="rgb")
    obs, _ = envr.reset(seed=42)
    envr.close()
    assert obs.shape == (84, 84, 3)
    assert obs.std() > 10, "rgb obs looks empty (std=%.1f)" % obs.std()
