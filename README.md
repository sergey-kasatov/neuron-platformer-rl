# Neuron Platformer RL

A procedurally generated 2D platformer environment for Reinforcement Learning and Computer Vision experiments.

## Project Goals

- Build a custom Gymnasium-compatible platformer environment.
- Train RL agents on procedurally generated levels.
- Avoid memorization by using random seeds and controlled difficulty.
- Compare human play vs AI play.
- Add visual debug overlays for portfolio demonstrations.
- Later: train agents directly from RGB pixels and add object-detection style visualization.

## Features

- Pygame platformer engine
- Gymnasium API: `reset()`, `step()`, `render()`
- Procedural, provably solvable level generator: every gap is capped by the
  jump physics envelope, and `scripts/audit_solvability.py` re-checks the
  guarantee across hundreds of seeds
- Mario-style level patterns: ground runs with pits, floating platform
  sections, bonus ledges, coin arcs, patrolling enemies, decorations
- Difficulty modes: easy, medium, hard, demo
- Fixed demo seed for public presentation
- PPO training with Stable-Baselines3
- State-vector observation for fast first training
- RGB observation mode prepared for computer vision experiments
- Dual renderer: Kenney sprite art for humans and replays, a clean
  flat-colour frame for the agent's 84x84 observation
- Debug overlay with enemies, portal, reward, seed, and metrics

## Install

```bash
pip install -r requirements.txt
```

## Human Play

```bash
python human_play.py --difficulty demo --seed 42 --debug
```

Controls:

- Left / A
- Right / D
- Space / Up / W for jump

## Test Environment

```bash
python test_env.py
```

## Audit Level Solvability

```bash
python scripts/audit_solvability.py
```

Simulates the hardest possible jump for every main-path platform transition
across 300 seeds per difficulty and fails if any level is impossible.

## Random Agent

```bash
python -m neuron_platformer_rl.agents.random_agent
```

## Train PPO

```bash
python -m neuron_platformer_rl.agents.train_ppo
```

The model is saved to:

```text
models/ppo_neuron_platformer_v1_state.zip
```

## Watch Trained Agent

```bash
python -m neuron_platformer_rl.agents.play_agent
```

## Evaluate on Unseen Seeds

```bash
python -m neuron_platformer_rl.evaluation.evaluate_agent
```

## Assets

Sprites are from the Kenney "New Platformer Pack" (https://kenney.nl), CC0
licensed. Only the sprites the game uses are vendored in `assets/kenney/`
together with the pack's `License.txt`.

## Roadmap

- v1.1: better metrics and CSV logging
- v1.2: replay saving
- v1.3: human vs AI comparison mode
- v1.4: RGB CNN PPO training
- v1.5: vision debug panel and detection-style overlays
- v2.0: portfolio dashboard
