import pygame
from utils import car, find_start_position, next_generation

running = True
screen = pygame.display.set_mode((1200, 800))
pygame.display.set_caption("Racing neural network")

track = pygame.image.load("track1.png")
track = pygame.transform.scale(track, (1200, 800))

start_pos = find_start_position(track)
cars = [car(start_pos, 5, 4) for _ in range(10)]

pygame.font.init()
font = pygame.font.SysFont(None, 36)


def draw(screen, images, cars, track):
    for img, pos in images:
        screen.blit(img, pos)

    alive_count = 0
    for c in cars:
        c.draw(screen, track)
        if c.alive and not c.lap_complete:
            alive_count += 1

    text = f"Alive: {alive_count}/{len(cars)}"
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(text_surface, (10, 10))
    pygame.display.update()


FPS = 60
clock = pygame.time.Clock()
images = [(track, (0, 0))]
generation = 1

while running:
    clock.tick(FPS)
    draw(screen, images, cars, track)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for c in cars:
        c.drive_with_network(track)

    all_done = all((not c.alive) or c.lap_complete for c in cars)
    if all_done:
        best_fitness = max(c.fitness() for c in cars)
        print(f"Generation {generation} done. Best fitness: {best_fitness:.1f}")
        cars = next_generation(cars, start_pos)
        generation += 1

pygame.quit()