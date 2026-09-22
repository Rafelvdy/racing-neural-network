import pygame
import json

pygame.init()
track = pygame.image.load("track1.png")
track = pygame.transform.scale(track, (1200, 800))
screen = pygame.display.set_mode((1200, 800))
pygame.display.set_caption("Click checkpoints in order | Z = undo | S = save")

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
                with open("checkpoints.json", "w") as f:
                    json.dump(checkpoints, f)
                print(f"Saved {len(checkpoints)} checkpoints to checkpoints.json")

pygame.quit()