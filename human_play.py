import argparse
import pygame

from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="demo", choices=["easy", "medium", "hard", "demo"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pygame.init()

    env = NeuronPlatformerEnv(
        render_mode="human",
        difficulty=args.difficulty,
        seed=args.seed,
        debug=args.debug,
    )

    obs, info = env.reset(seed=args.seed)

    clock = pygame.time.Clock()
    running = True

    while running:
        action = 0  # idle

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            action = 1
        elif keys[pygame.K_RIGHT]:
            action = 2

        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = 3

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            obs, info = env.reset(seed=args.seed)

        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()