from __future__ import annotations
import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from neuron_platformer_rl.generators.level_generator import LevelGenerator, Level
from neuron_platformer_rl.utils.assets import load_assets

ACTIONS = {0: "NOOP", 1: "LEFT", 2: "RIGHT", 3: "JUMP", 4: "RIGHT_JUMP", 5: "LEFT_JUMP"}

# Episode step budget per difficulty. A hard level is up to 5440 px wide and
# the run speed truncates to 4 px/step, so the flat 1400-step budget of the
# easy phase cannot even cross the level; wider maps get more time. easy and
# demo stay at exactly 1400 so all published v1/v2 numbers keep reproducing.
TIME_BUDGET = {"easy": 1400, "demo": 1400, "medium": 2000, "hard": 2400}

class NeuronPlatformerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, difficulty: str = "easy", seed: int | None = None,
                 observation_mode: str = "rgb", debug: bool = False):
        super().__init__()
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.fixed_seed = seed
        self.observation_mode = observation_mode
        self.debug = debug
        self.width, self.height = 960, 540
        self.generator = LevelGenerator(self.width, self.height)
        self.action_space = spaces.Discrete(6)
        if observation_mode == "state":
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(19,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
        self.screen = None
        self.clock = None
        self._clean = None      # offscreen surface for the agent's observation
        self._font = None
        self.facing = 1
        self.reset()

    def set_difficulty(self, difficulty: str):
        """Curriculum hook: takes effect on the next reset. Reached through
        vec_env.env_method, which forwards through the Monitor wrapper."""
        self.difficulty = difficulty

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        lvl_seed = self.fixed_seed if self.fixed_seed is not None else seed
        self.level: Level = self.generator.generate(lvl_seed, self.difficulty)
        # Main forward path (bonus ledges that overlap a run are excluded):
        # used by the terrain features in the state observation.
        chain = []
        for p in sorted(self.level.platforms, key=lambda q: q.x):
            if chain and p.x < chain[-1].x + chain[-1].w:
                continue
            chain.append(p)
        self._chain = chain
        self.player = pygame.Rect(self.level.spawn_x, self.level.spawn_y, 32, 44)
        self.vx = 0.0; self.vy = 0.0; self.on_ground = False
        self.camera_x = 0
        self.steps = 0; self.episode_reward = 0.0; self.max_x = self.player.x
        self.deaths = 0; self.enemy_kills = 0; self.chips_collected = 0; self.success = False
        return self._obs(), self._info()

    def step(self, action: int):
        self.steps += 1
        reward = -0.01
        move = 0
        jump = False
        if action in (1,5): move = -1
        if action in (2,4): move = 1
        if action in (3,4,5): jump = True
        if move: self.facing = move
        self.vx = move * 4.2
        if jump and self.on_ground:
            self.vy = -12.2
            self.on_ground = False
        self.vy = min(15, self.vy + 0.55)
        self.player.x += int(self.vx)
        self._collide_x()
        self.player.y += int(self.vy)
        self.on_ground = False
        self._collide_y()
        self._update_enemies()

        # Reward only NEW forward progress (beyond max_x). Paying for raw
        # rightward movement let PPO farm reward by pacing back and forth on
        # the spawn platform: 100 average reward at x=400 with 0% success.
        # max_x is monotonic, so this term cannot be harvested twice.
        if self.player.x > self.max_x:
            reward += (self.player.x - self.max_x) * 0.05
            self.max_x = self.player.x

        for chip in self.level.chips:
            if not chip.collected and self.player.collidepoint(chip.x, chip.y):
                chip.collected = True
                self.chips_collected += 1
                reward += 2.0

        terminated = False
        for enemy in self.level.enemies:
            if not enemy.alive:
                continue
            er = pygame.Rect(enemy.x, enemy.y, enemy.w, enemy.h)
            if self.player.colliderect(er):
                if self.vy > 0 and self.player.bottom - er.top < 18:
                    enemy.alive = False
                    self.enemy_kills += 1
                    self.vy = -7.5
                    reward += 5.0
                else:
                    # Death penalty is mild on purpose. At -50 the safest
                    # policy PPO could find was to stop at the first pit edge
                    # and wait out the episode (0% success, x stuck at the
                    # spawn platform). Dying while trying must stay cheaper
                    # than never trying.
                    reward -= 10.0
                    terminated = True
                    self.deaths += 1

        portal_rect = pygame.Rect(self.level.portal.x, self.level.portal.y, self.level.portal.w, self.level.portal.h)
        if self.player.colliderect(portal_rect):
            reward += 80.0
            self.success = True
            terminated = True

        if self.player.y > self.height + 120:
            reward -= 10.0   # mild: see the enemy-death comment above
            self.deaths += 1
            terminated = True
        truncated = self.steps >= TIME_BUDGET.get(self.difficulty, 1400)
        self.episode_reward += reward
        return self._obs(), reward, terminated, truncated, self._info(action, reward)

    def _collide_x(self):
        for p in self.level.platforms:
            r = pygame.Rect(p.x, p.y, p.w, p.h)
            if self.player.colliderect(r):
                if self.vx > 0: self.player.right = r.left
                elif self.vx < 0: self.player.left = r.right

    def _collide_y(self):
        for p in self.level.platforms:
            r = pygame.Rect(p.x, p.y, p.w, p.h)
            if self.player.colliderect(r):
                if self.vy > 0:
                    self.player.bottom = r.top; self.vy = 0; self.on_ground = True
                elif self.vy < 0:
                    self.player.top = r.bottom; self.vy = 0

    def _update_enemies(self):
        for e in self.level.enemies:
            if not e.alive: continue
            e.x += e.direction * e.speed
            current = next((p for p in self.level.platforms if p.x <= e.x <= p.x+p.w and abs((p.y-32)-e.y)<4), None)
            if current and (e.x < current.x+20 or e.x+e.w > current.x+current.w-20):
                e.direction *= -1

    def _obs(self):
        if self.observation_mode == "state":
            portal = self.level.portal
            nearest_enemy = min([e for e in self.level.enemies if e.alive], key=lambda e: abs(e.x-self.player.x), default=None)
            nearest_chip = min([c for c in self.level.chips if not c.collected], key=lambda c: abs(c.x-self.player.x), default=None)
            vals = [
                self.player.x/self.level.width, self.player.y/self.height, self.vx/10, self.vy/16,
                float(self.on_ground), (portal.x-self.player.x)/self.level.width,
                self.chips_collected/max(1,len(self.level.chips)), self.enemy_kills/max(1,len(self.level.enemies)),
            ]
            if nearest_enemy: vals += [(nearest_enemy.x-self.player.x)/500, (nearest_enemy.y-self.player.y)/300, 1.0]
            else: vals += [0,0,0]
            if nearest_chip: vals += [(nearest_chip.x-self.player.x)/500, (nearest_chip.y-self.player.y)/300, 1.0]
            else: vals += [0,0,0]
            # Terrain ahead. Without these the policy is blind to pits: a
            # 500k-step PPO run collapsed to deterministic running and scored
            # 0% on unseen seeds, because nothing in the observation says
            # where the current platform ends or how wide the next gap is.
            px = self.player.x
            cur = nxt = None
            for q in self._chain:
                if q.x - 4 <= px <= q.x + q.w + 4: cur = q
                elif q.x > px: nxt = q; break
            edge = (cur.x + cur.w - px) / 400 if cur else 0.0
            if nxt is not None:
                start = cur.x + cur.w if cur else px
                vals += [edge, (nxt.x - start) / 300, (nxt.y - self.player.bottom) / 250]
            else:
                vals += [edge, 0.0, 0.0]
            vals += [self.steps/TIME_BUDGET.get(self.difficulty, 1400), 1.0]
            return np.clip(np.array(vals, dtype=np.float32), -1, 1)
        # The agent trains on a deliberately clean frame (flat colours, no
        # parallax or animation): visual noise slows pixel-based RL without
        # adding information. The pretty Kenney frame stays for humans/GIFs.
        return self._render_clean()

    def _info(self, action=None, reward=0.0):
        # is_success duplicates success under the exact key SB3's EvalCallback
        # reads, so evaluations log a success rate, not just mean reward.
        return dict(seed=self.level.seed, difficulty=self.level.difficulty, action=ACTIONS.get(action,"RESET"),
                    reward=reward, episode_reward=self.episode_reward, success=self.success, is_success=self.success,
                    chips_collected=self.chips_collected, enemy_kills=self.enemy_kills,
                    deaths=self.deaths, distance_to_goal=max(0, self.level.portal.x-self.player.x), x=self.player.x)

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width,self.height)) if self.render_mode=="human" else pygame.Surface((self.width,self.height))
            self.clock = pygame.time.Clock()
        surf = self.screen
        self.camera_x = max(0, min(self.player.x - 240, self.level.width-self.width))
        self._draw_world(surf)
        if self.debug:
            self._draw_debug(surf)
        if self.render_mode == "human":
            pygame.display.flip(); self.clock.tick(60)
        return pygame.surfarray.array3d(surf).swapaxes(0,1)

    def _draw_world(self, surf):
        """Human-facing renderer: Kenney sprites, parallax, animation, HUD."""
        A = load_assets()
        cam = int(self.camera_x)
        bg = A["bg"]
        off = -int(cam * 0.3) % 540
        for bx in range(off - 540, self.width + 540, 540):
            surf.blit(bg, (bx, 0))
        for p in self.level.platforms:
            x = p.x - cam
            if x > self.width or x + p.w < 0:
                continue
            cols = p.w // 32
            if p.y >= 500:
                # ground run: grass top row, dirt filled to the screen bottom
                for i in range(cols):
                    tile = A["g_tl"] if i == 0 else A["g_tr"] if i == cols-1 else A["g_top"]
                    surf.blit(tile, (x + i*32, p.y))
                for ty in range(p.y + 32, self.height + 32, 32):
                    for i in range(cols):
                        tile = A["g_l"] if i == 0 else A["g_r"] if i == cols-1 else A["g_c"]
                        surf.blit(tile, (x + i*32, ty))
            else:
                # floating platform: thin grass strip
                for i in range(cols):
                    tile = A["h_l"] if i == 0 else A["h_r"] if i == cols-1 else A["h_m"]
                    surf.blit(tile, (x + i*32, p.y))
        for d in getattr(self.level, "decor", []):
            img = A.get(d.kind)
            if img:
                surf.blit(img, (d.x - cam - img.get_width()//2, d.y - img.get_height()))
        # spin: mostly the front frame with a brief side flash reads as a
        # rotation in motion and avoids sliver-only coins on still frames
        coin = A["coin"][1 if (self.steps // 6) % 4 == 3 else 0]
        for c in self.level.chips:
            if not c.collected:
                surf.blit(coin, (int(c.x - cam) - 12, int(c.y) - 12))
        prt = self.level.portal; px = prt.x - cam
        pygame.draw.rect(surf, (90, 94, 106), (px + 4, prt.y - 8, 6, prt.h + 8))
        surf.blit(A["flag"][(self.steps // 12) % 2], (px + 10, prt.y - 8))
        for e in self.level.enemies:
            if e.alive:
                img = A["slime"][(self.steps // 10) % 2]
                if e.direction > 0:
                    img = pygame.transform.flip(img, True, False)
                surf.blit(img, (int(e.x) - cam, e.y))
        if not self.on_ground:
            img = A["player"]["jump"]
        elif self.vx:
            img = A["player"]["walk_a" if (self.steps // 6) % 2 else "walk_b"]
        else:
            img = A["player"]["idle"]
        if self.facing < 0:
            img = pygame.transform.flip(img, True, False)
        surf.blit(img, (self.player.centerx - cam - 24, self.player.bottom - 48))
        surf.blit(A["hud_coin"], (12, 10))
        if self._font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font = pygame.font.SysFont("Arial", 22, bold=True)
        surf.blit(self._font.render(str(self.chips_collected), True, (40, 40, 60)), (46, 12))

    def _render_clean(self):
        """Observation renderer: flat colours, no parallax, no animation.
        One colour per entity class keeps the 84x84 CNN input unambiguous.
        Drawn directly at 84x84: the first version drew the full 960x540
        frame and smoothscaled it down, capping the env at ~130 steps/s.
        Both edges of every rect are mapped to obs space independently
        (width = right - left), so gap widths stay pixel-accurate and
        standing entities keep exact contact with platform tops."""
        if self._clean is None:
            pygame.init()
            self._clean = pygame.Surface((84, 84))
        surf = self._clean
        sx, sy = 84 / self.width, 84 / self.height
        cam = max(0, min(self.player.x - 240, self.level.width - self.width))

        def rect(colour, x0, y0, x1, y1):
            rx, ry = round((x0 - cam) * sx), round(y0 * sy)
            rw = max(1, round((x1 - cam) * sx) - rx)
            rh = max(1, round(y1 * sy) - ry)
            pygame.draw.rect(surf, colour, (rx, ry, rw, rh))

        surf.fill((94, 174, 235))
        for p in self.level.platforms:
            x = p.x - cam
            if -p.w < x < self.width:
                bottom = self.height if p.y >= 500 else p.y + p.h
                rect((150, 92, 42), p.x, p.y, p.x + p.w, bottom)
                rect((80, 180, 70), p.x, p.y, p.x + p.w, p.y + 10)
        for c in self.level.chips:
            if not c.collected:
                rect((255, 220, 60), c.x - c.r, c.y - c.r, c.x + c.r, c.y + c.r)
        for e in self.level.enemies:
            if e.alive:
                rect((205, 55, 65), e.x, e.y, e.x + e.w, e.y + e.h)
        prt = self.level.portal
        rect((25, 210, 255), prt.x, prt.y, prt.x + prt.w, prt.y + prt.h)
        rect((28, 60, 205), self.player.x, self.player.y,
             self.player.x + self.player.w, self.player.y + self.player.h)
        return pygame.surfarray.array3d(surf).swapaxes(0, 1).astype(np.uint8)

    def _draw_debug(self, surf):
        font = pygame.font.SysFont("Arial", 16)
        for e in self.level.enemies:
            if e.alive:
                pygame.draw.rect(surf, (255,0,0), (e.x-self.camera_x,e.y,e.w,e.h), 2)
                surf.blit(font.render("ENEMY", True, (255,0,0)), (e.x-self.camera_x,e.y-18))
        for c in self.level.chips:
            if not c.collected:
                pygame.draw.circle(surf, (255,255,0), (int(c.x-self.camera_x), int(c.y)), c.r+3, 2)
        pygame.draw.rect(surf, (0,255,255), (self.level.portal.x-self.camera_x,self.level.portal.y,self.level.portal.w,self.level.portal.h), 2)
        info = f"seed={self.level.seed} diff={self.difficulty} reward={self.episode_reward:.1f} x={self.player.x} chips={self.chips_collected} kills={self.enemy_kills}"
        surf.blit(font.render(info, True, (0,0,0)), (10,10))

    def close(self):
        if self.screen is not None:
            pygame.quit(); self.screen = None
