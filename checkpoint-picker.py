import argparse
import json
import os

import pygame


parser = argparse.ArgumentParser()
parser.add_argument("--track", default="track1.png")
parser.add_argument("--output", default=None)
args = parser.parse_args()
output_file = args.output or f"{os.path.splitext(args.track)[0]}_checkpoints.json"

pygame.init()
track = pygame.image.load(args.track)
track = pygame.transform.scale(track, (1200, 800))
screen = pygame.display.set_mode((1200, 800))
pygame.display.set_caption(
    f"Click checkpoints in order | Z = undo | S = save to {output_file}"
)

checkpoints = []
running = True

while running:
    screen.blit(track, (0, 0))

    for i, (x, y) in enumerate(checkpoints):
        pygame.draw.circle(screen, (255, 0, 0), (x, y), 6)
        pygame.draw.circle(screen, (255, 255, 255), (x, y), 6, 1)

    if len(checkpoints) > 1:
        pygame.draw.lines(screen, (255, 255, 0), False, checkpoints, 2)

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            checkpoints.append(list(event.pos))
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z and checkpoints:
                checkpoints.pop()
            elif event.key == pygame.K_s:
                with open(output_file, "w") as f:
                    json.dump(checkpoints, f)
                print(f"Saved {len(checkpoints)} checkpoints to {output_file}")

pygame.quit()