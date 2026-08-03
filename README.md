# Neuron Platformer RL

A procedurally generated 2D platformer environment for Reinforcement Learning and Computer Vision experiments.

![Trained PPO agent on an unseen level, with a live policy monitor](assets/demo_agent_brain.gif)

The agent above plays a level it never saw during training. The panel shows,
in real time, the 19 input features, the actor's hidden activations, the
action distribution and the value estimate behind every decision.

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
- Pixel observation mode: the CNN agent trains on stacked clean 84x84
  frames and beats the state-vector baseline (80% vs 50% on unseen seeds)
- Dual renderer: Kenney sprite art for humans and replays, a clean
  flat-colour frame drawn directly at 84x84 for the agent (the direct
  low-res draw took the env from 129 to ~4,800 steps/s)
- Grad-CAM saliency maps showing where the CNN looks before each action
- Debug overlay with enemies, portal, reward, seed, and metrics
- Live policy monitor: input features, hidden activations, action
  probabilities and V(s) rendered beside the game (roadmap v1.5)
- Functional test suite: `scripts/smoke_test.py` (18 checks, no
  dependencies) and the same checks as pytest cases in `tests/`

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

Functional checks (physics, reward contract, enemies, both observation
modes), headless:

```bash
python -m pytest
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

Defaults evaluate the v1 state model. For the pixel model:

```bash
python -m neuron_platformer_rl.evaluation.evaluate_agent --model models/ppo_neuron_platformer_v2_pixels.zip --obs rgb
```

The v4 LSTM model needs `--recurrent`, which drops the frame stack and
threads the hidden state through `predict`. Add `--device cpu` to keep
the GPU free while something else is training:

```bash
python -m neuron_platformer_rl.evaluation.evaluate_agent --model models/ppo_neuron_platformer_v4_lstm.zip --obs rgb --recurrent
```

## Record a Demo GIF

```bash
python scripts/record_agent_gif.py --seed 10000
```

Runs the trained model on the given seed and writes
`assets/demo_agent_brain.gif` with the policy monitor panel attached.

## Training Results (v1 state model)

PPO (SB3), 2M steps across 8 parallel envs, CPU, about 30 minutes.
Evaluated on 30 procedurally generated levels the agent never saw:

| Metric | Value |
|--------|-------|
| Success rate (reached the flag) | 50% |
| Average distance | 2434 px |
| Average coins | 9.7 |

Getting here required fixing four classic RL failure modes, all documented
in code comments: pit-blind observations (terrain features added to the
state vector), entropy collapse (ent_coef), a reward farm from paying for
raw rightward movement (progress is now rewarded only beyond max_x), and
an over-harsh death penalty that taught the agent to freeze at the first
pit edge instead of jumping.

The whole build history, one panel per training run:

![Training history: seven runs, four fixes](assets/training_history.png)

Y-scales are per-panel on purpose: the reward function changed between
runs, so episode rewards are not comparable across panels. The only number
that is comparable is the success rate on the same 30 unseen seeds. The
interactive version of every curve is in tensorboard:

```bash
tensorboard --logdir logs/tensorboard
```

## Training Results (v2 pixel model, roadmap v1.4)

The computer-vision phase: no hand-crafted features, the agent reads the
game from pixels alone. PPO with a Nature-CNN policy on 4 stacked clean
84x84 RGB frames (colour kept on purpose: entity classes are
colour-coded), 10M steps across 24 parallel envs, RTX 3070 laptop GPU,
about 3.5 hours:

```bash
python -m neuron_platformer_rl.agents.train_ppo_pixels
```

Every 250k steps the policy was evaluated on the same 30 held-out seeds
as v1; the committed `models/ppo_neuron_platformer_v2_pixels.zip` is the
best-on-eval snapshot, and its numbers reproduce exactly with the
evaluation command above:

| Metric (30 unseen seeds, deterministic) | v1 state | v2 pixels |
|--------|-------|--------|
| Success rate (reached the flag) | 50% | **80%** |
| Average distance | 2434 px | 2943 px |
| Average coins | 9.7 | 15.1 |

![Pixel training curve](assets/pixel_training_curve.png)

The pixel agent crosses the state-vector baseline around 3M steps and
keeps climbing. Rewards in this figure are comparable to each other (one
reward function throughout the run) but not to the v1 panels above.

Two things made this run cheap. The observation renderer draws the flat
frame directly at 84x84 instead of downscaling a full 960x540 frame
(129 to ~4,800 env steps/s). And with an env step at ~0.2 ms, Windows
subprocess vec-envs lost to a plain `DummyVecEnv`: pipe round trips cost
more than the envs themselves (measured ~350 fps subproc x8 vs ~1,000
fps dummy x24 end to end).

## Where the CNN Looks (Grad-CAM)

```bash
python scripts/saliency_maps.py --model models/ppo_neuron_platformer_v2_pixels.zip --seed 10003
```

Gradients of the chosen action's logit weight the last conv layer's
activations into an attention heatmap - the same technique as Grad-CAM
on an image classifier, applied to a policy network:

![Grad-CAM saliency maps](assets/saliency_maps.png)

Columns: human-facing game frame, the actual 84x84 observation, and the
Grad-CAM overlay. The trained network attends to exactly the
task-relevant pixels: the platform edge and the gap ahead of a jump, the
enemy when close, and the portal with its coin cluster in the final
approach.

`--recurrent` runs the same analysis on the v4 LSTM policy. Its forward
path differs - the action logit depends on the hidden state as well as on
the frame - so the hidden state is threaded in and the gradient reaches
the conv layer through the LSTM.

## Training Results (v3 curriculum, roadmap v1.6)

Continuing from the v2 weights, a difficulty curriculum walks demo ->
medium -> hard with mixed env pools (16 envs on the new tier + 8 on the
previous one), so earlier tiers keep being rehearsed while a harder one
is learned:

```bash
python -m neuron_platformer_rl.agents.train_ppo_curriculum
```

9M steps, ~2.5 h on the same setup. Wider levels get bigger step budgets
(easy/demo 1400, medium 2000, hard 2400): a hard level is up to 5440 px
wide, and with run speed truncating to 4 px/step the flat 1400 budget
could not even cross one.

| Success rate (30 held-out seeds each) | v2 pixels | v3 curriculum |
|---|---|---|
| easy | 80% | 77% |
| demo | 80% | **93%** |
| medium | 33% | **57%** |
| hard | 23% | **53%** |

![Curriculum training curve](assets/curriculum_training_curve.png)

The committed `models/ppo_neuron_platformer_v3_curriculum.zip` is the
run's final model and also its best-on-eval hard snapshot; the table
reproduces via `evaluate_agent --obs rgb --difficulty <tier>`. Easy
slips by a single episode while every other tier climbs - the mixed
pools kept old skills alive while new ones were learned.

The pixel agent clearing a hard level it never saw, with its actual
84x84 input and a live Grad-CAM beside the game:

![Pixel agent on a hard unseen level with live Grad-CAM](assets/demo_pixel_agent.gif)

```bash
python scripts/record_pixel_agent_gif.py --difficulty hard --seed 10000
```

## Training Results (v4 recurrence experiment, roadmap v1.7)

v2 and v3 read velocity out of a stack of four frames. This run removes
the stack and asks an LSTM to carry that context instead, so the policy
sees exactly one 84x84 frame per step. Hyperparameters come from an
Optuna study (median-pruned, sqlite-resumable):

```bash
python -m neuron_platformer_rl.agents.tune_lstm_optuna
python -m neuron_platformer_rl.agents.train_ppo_lstm
```

LSTM weights cannot start from the CNN-only v2/v3, so this is 8M steps
from scratch and the ladder opens one tier lower: easy -> demo at 2.5M ->
medium + demo at 4M -> hard + medium at 6M, about 7 h. RecurrentPPO
measures ~250 fps here against ~1,000 for plain PPO, so those 8M
recurrent steps cost roughly the wall clock of v2's 10M reactive ones.
The budgets differ by design and the table has to be read that way.

| Success rate (30 held-out seeds each) | v2 pixels (10M, easy only) | v3 curriculum (19M cumulative) | v4 LSTM (8M from scratch) |
|---|---|---|---|
| easy | 80% | 77% | 70% |
| demo | 80% | **93%** | 60% |
| medium | 33% | **57%** | 37% |
| hard | 23% | **53%** | 30% |

The curve below covers the whole experiment, including the budget-doubling
extension described in the next section, so it runs to 15.5M rather than 8M:

![LSTM training curve](assets/lstm_training_curve.png)

Recurrence does replace the frame stack: an LSTM on single raw frames
learns this game. It also scores above the v2 column on the two harder
tiers (37 vs 33, 30 vs 23), but neither gap survives the noise analysis
below, and v2 only ever trained on easy, so its medium and hard figures
are pure transfer while v4 trained those tiers directly. The like-for-like
comparison is v4 against v3, both curricula ending on hard + medium, and
there the recurrent agent reaches roughly 55 to 90 percent of v3's
per-tier scores on less than half the steps and with no warm start.

### Doubling the budget: 8M to 16M steps

The 8M run ended with both eval curves at their maximum, which looked like
a trend that had not finished. It was not one. The run was resumed from its
final save and trained to 16M inside the same hard + medium stage, so the
two halves are directly comparable. Pooling the last four evaluations of
each half gives 120 held-out episodes per cell:

| pooled over 120 episodes | 6.5 to 8M | 14 to 15.5M | delta | z |
|---|---|---|---|---|
| medium success | 24.2% +/- 3.9 | 36.7% +/- 4.4 | +12.5 | 2.12 |
| hard success | 12.5% +/- 3.0 | 25.8% +/- 4.0 | +13.3 | 2.66 |
| medium reward | 161.9 | 191.3 | +29.4 | 1.96 |
| hard reward | 156.1 | 205.3 | +49.2 | 2.97 |

Four tests, so a Bonferroni correction asks for |z| > 2.50: hard clears it
on both metrics, medium does not. Meanwhile the peak barely moved. Running
the same held-out matrix on the 16M best-on-eval snapshot gives 36.7 percent
medium (identical to 8M) and 33.3 percent hard (one more episode out of 30),
so the extra budget bought typical performance rather than peak performance:
the agent became reliably as good as it used to be on its best days, without
getting better on its best days. The published table above stays the 8M
snapshot, because the 16M one does not dominate it.

It also cost something. The mixed env pools rehearse only the ADJACENT tier -
the final stage runs hard x16 + medium x8 - so nothing has trained or
rehearsed easy since 2.5M, and over those 8M extra steps easy decayed from
70 to 50 percent (demo 60 to 57). That is the same mechanism the curriculum
section credits for protecting demo, seen from the other side: mixed pools
slow forgetting where they reach and do nothing where they do not.

### What a 30-episode evaluation can and cannot tell you

This is the most transferable result in the project, and it applies to every
table above. A deterministic evaluation over 30 held-out seeds has a binomial
standard error of about 7.3 points at a true success rate near 0.2, so its
95 percent interval spans roughly plus or minus 14 points. The extension
demonstrates it directly: with the training side flat, individual medium
evaluations read anywhere from 6.7 to 43.3 percent and hard from 0.0 to
33.3 percent.

Two consequences:

- The v4-vs-v2 gaps quoted above (37 vs 33 on medium, 30 vs 23 on hard) sit
  well inside that interval. They are not evidence of anything.
- Every published column is a best-on-eval snapshot, which is the top of a
  noisy band rather than typical behaviour. That is true of v2 and v3 too, so
  the columns are at least biased the same way, but any single one should be
  read as an upper estimate.

The tables themselves are exact and reproduce bit for bit, because a fixed
model on fixed seeds is deterministic. What is noisy is reading one such
number as the architecture's ability. The fix is more seeds: 200 held-out
seeds would cut the standard error to 2.8 points, and re-measuring v2, v3
and v4 together is the obvious next step.

One negative result worth recording: a shared actor-critic LSTM, used to
buy throughput during tuning, collapsed in the first full run -
explained variance around 0, approx_kl about 2e-4 and clip fraction 0, so
the critic never fit and the updates were effectively dead. The
sb3-contrib default separate critic LSTM costs roughly 30 percent
throughput and trains normally (explained variance 0.85 to 0.95).

The recurrent agent clearing a hard level it has never seen, on one frame
of memory-free input at a time, with the Grad-CAM computed through the
LSTM. It is one of the 9 hard seeds out of 30 that this model clears, and
the widest of them at 5,956 px:

![v4 LSTM agent clearing a hard unseen level](assets/demo_v4_lstm.gif)

```bash
python scripts/record_pixel_agent_gif.py --recurrent --model models/ppo_neuron_platformer_v4_lstm.zip --difficulty hard --seed 10027
```

The committed `models/ppo_neuron_platformer_v4_lstm.zip` is the
best-on-eval hard snapshot at 8M, which also beats the final-step model
on three tiers of four (70/60/37/30 against 67/63/30/20 - a deterministic
30-episode evaluation is sensitive to one more gradient update). The
table reproduces via:

```bash
python -m neuron_platformer_rl.evaluation.evaluate_agent --model models/ppo_neuron_platformer_v4_lstm.zip --obs rgb --recurrent --difficulty medium
```

## Assets

Sprites are from the Kenney "New Platformer Pack" (https://kenney.nl), CC0
licensed. Only the sprites the game uses are vendored in `assets/kenney/`
together with the pack's `License.txt`.

## Roadmap

- v1.1: better metrics and CSV logging
- v1.2: replay saving
- v1.3: human vs AI comparison mode
- v1.4: RGB CNN PPO training - done (80% on unseen seeds + Grad-CAM)
- v1.5: vision debug panel and detection-style overlays - done (policy monitor)
- v1.6: curriculum to medium/hard - done (57% medium / 53% hard from raw pixels)
- v1.7: recurrence ceiling experiment - done (LSTM on single frames, 37% medium
  / 30% hard on 8M from scratch; doubling the budget to 16M raised typical
  performance but not the peak)
- v2.0: portfolio dashboard
