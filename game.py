import pygame
from utils import car, find_start_position, next_generation, save_weights, load_checkpoints

running = True
screen = pygame.display.set_mode((1200, 800))
pygame.display.set_caption("Racing neural network")

track = pygame.image.load("track1.png")
track = pygame.transform.scale(track, (1200, 800))

start_pos = find_start_position(track)
checkpoints = load_checkpoints("checkpoints.json")

cars = [car(start_pos, 5, 4, checkpoints=checkpoints) for _ in range(10)]

pygame.font.init()
font = pygame.font.SysFont(None, 36)


def draw(screen, images, cars, track, checkpoints):
    for img, pos in images:
        screen.blit(img, pos)

    for x, y in checkpoints:
        pygame.draw.circle(screen, (255, 165, 0), (x, y), 6, 1)

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

GENERATION_TIMEOUT_MS = 35000
generation_start_time = pygame.time.get_ticks()

BASE_MUTATION_STRENGTH = 0.4
MAX_MUTATION_STRENGTH = 1.5
STAGNATION_LIMIT = 30

best_fitness_ever = float("-inf")
generations_without_improvement = 0
mutation_strength = BASE_MUTATION_STRENGTH

while running:
    clock.tick(FPS)
    draw(screen, images, cars, track, checkpoints)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for c in cars:
        c.drive_with_network(track)

    elapsed = pygame.time.get_ticks() - generation_start_time
    all_done = all((not c.alive) or c.lap_complete for c in cars) or elapsed > GENERATION_TIMEOUT_MS

    if all_done:
        best_car = max(cars, key=lambda c: c.fitness())
        current_best = best_car.fitness()

        if current_best > best_fitness_ever:
            best_fitness_ever = current_best
            generations_without_improvement = 0
            mutation_strength = BASE_MUTATION_STRENGTH
        else:
            generations_without_improvement += 1

        if generations_without_improvement > STAGNATION_LIMIT:
            mutation_strength = min(mutation_strength * 1.2, MAX_MUTATION_STRENGTH)

        print(f"Generation {generation} done. Best fitness: {current_best:.1f}, "
              f"checkpoints passed: {best_car.checkpoints_passed}/{len(checkpoints)}, "
              f"stagnant for: {generations_without_improvement}, "
              f"mutation strength: {mutation_strength:.2f}")

        save_weights(best_car, "best_car.json")

        cars = next_generation(cars, start_pos, checkpoints=checkpoints,
                                mutation_strength=mutation_strength)
        generation += 1
        generation_start_time = pygame.time.get_ticks()

pygame.quit()