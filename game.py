import argparse
import os

import pygame

from utils import (CHECKPOINT_GATE_HALF_WIDTH, car, checkpoint_gate,
                   find_start_position, load_archive, load_checkpoints,
                   load_weights, next_generation, save_archive, save_weights,
                   update_checkpoint_archive)


SCREEN_SIZE = (1200, 800)
FPS = 60
GENERATION_TIMEOUT_MS = 35000
POPULATION_SIZE = 40
BASE_MUTATION_STRENGTH = 0.4
MAX_MUTATION_STRENGTH = 1.5
STAGNATION_LIMIT = 30
BASE_MUTATION_RATE = 0.15
MAX_MUTATION_RATE = 0.5
IMMIGRANTS_ON_STAGNATION = 6


def parse_args():
    parser = argparse.ArgumentParser(description="Train or test a racing network.")
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--track", default="track1.png")
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--population", type=int, default=POPULATION_SIZE)
    parser.add_argument("--timeout", type=int, default=GENERATION_TIMEOUT_MS)
    parser.add_argument("--laps", type=int, default=1)
    return parser.parse_args()


def default_checkpoint_file(track_file):
    track_name = os.path.splitext(os.path.basename(track_file))[0]
    candidate = f"{track_name}_checkpoints.json"
    if os.path.exists(candidate):
        return candidate
    return "checkpoints.json" if track_name == "track1" else candidate


def default_weights_file(track_file):
    track_name = os.path.splitext(os.path.basename(track_file))[0]
    return "best_car.json" if track_name == "track1" else f"best_{track_name}.json"


def default_archive_file(track_file):
    track_name = os.path.splitext(os.path.basename(track_file))[0]
    return ("checkpoint_archive.json" if track_name == "track1"
            else f"{track_name}_checkpoint_archive.json")


def load_track(track_file):
    track = pygame.image.load(track_file)
    return pygame.transform.scale(track, SCREEN_SIZE)


def draw(screen, cars, track, checkpoints, status):
    screen.blit(track, (0, 0))
    for index, (x, y) in enumerate(checkpoints):
        gate_start, gate_end = checkpoint_gate(
            checkpoints,
            index,
            CHECKPOINT_GATE_HALF_WIDTH,
        )
        pygame.draw.line(screen, (255, 165, 0), gate_start, gate_end, 2)
        pygame.draw.circle(screen, (255, 165, 0), (x, y), 4, 1)

    alive_count = 0
    for current_car in cars:
        current_car.draw(screen, track)
        if current_car.alive and not current_car.lap_complete:
            alive_count += 1

    text = f"{status} | Alive: {alive_count}/{len(cars)}"
    text_surface = pygame.font.SysFont(None, 30).render(
        text, True, (255, 255, 255)
    )
    screen.blit(text_surface, (10, 10))
    pygame.display.update()


def run_test(screen, track, checkpoints, weights_file, timeout_ms, target_laps):
    start_pos = find_start_position(track)
    test_car = car(start_pos, 5, 4, checkpoints=checkpoints,
                   target_laps=target_laps)
    load_weights(test_car, weights_file)
    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks()
    running = True
    elapsed = 0

    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if running and test_car.alive and not test_car.lap_complete:
            test_car.drive_with_network(track)

        elapsed = pygame.time.get_ticks() - start_time
        draw(screen, [test_car], track, checkpoints, "TEST")

        if test_car.lap_complete or not test_car.alive or elapsed > timeout_ms:
            running = False

    if test_car.lap_complete:
        print(f"Test complete. Laps: {test_car.lap_count}. "
              f"Last lap time: {test_car.lap_time:.2f}s")
    elif not test_car.alive:
        reason = test_car.death_reason or "stopped"
        print(f"Test ended: car {reason} after {elapsed / 1000:.2f}s. "
              f"Checkpoints: {test_car.checkpoints_passed}/{len(checkpoints)}")
    else:
        print(f"Test timed out after {elapsed / 1000:.2f}s. "
              f"Checkpoints: {test_car.checkpoints_passed}/{len(checkpoints)}")


def run_training(screen, track, checkpoints, weights_file, archive_file,
                 population_size, timeout_ms):
    start_pos = find_start_position(track)
    cars = [car(start_pos, 5, 4, checkpoints=checkpoints)
            for _ in range(population_size)]
    loaded_existing_weights = False
    try:
        load_weights(cars[0], weights_file)
        loaded_existing_weights = True
    except FileNotFoundError:
        pass

    clock = pygame.time.Clock()
    generation = 1
    generation_start_time = pygame.time.get_ticks()
    best_fitness_ever = float("-inf")
    generations_without_improvement = 0
    mutation_strength = BASE_MUTATION_STRENGTH
    mutation_rate = BASE_MUTATION_RATE
    checkpoint_archive = load_archive(archive_file)
    running = True

    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for current_car in cars:
            current_car.drive_with_network(track)

        elapsed = pygame.time.get_ticks() - generation_start_time
        all_done = (all((not current_car.alive) or current_car.lap_complete
                        for current_car in cars)
                    or elapsed > timeout_ms)
        draw(screen, cars, track, checkpoints, f"TRAIN G{generation}")

        if not all_done:
            continue

        normal_start_cars = [current_car for current_car in cars
                     if not current_car.is_recovery]
        best_car = max(normal_start_cars, key=lambda current_car: current_car.fitness())
        current_best = best_car.fitness()
        existing_best = cars[0].fitness() if loaded_existing_weights else float("-inf")
        update_checkpoint_archive(cars, checkpoint_archive)

        improved_saved_model = current_best > best_fitness_ever
        if loaded_existing_weights and generation == 1:
            improved_saved_model = current_best > existing_best

        if improved_saved_model:
            best_fitness_ever = current_best
            generations_without_improvement = 0
            mutation_strength = BASE_MUTATION_STRENGTH
            mutation_rate = BASE_MUTATION_RATE
            save_weights(best_car, weights_file)
        else:
            best_fitness_ever = max(best_fitness_ever, existing_best)
            generations_without_improvement += 1

        if generations_without_improvement > STAGNATION_LIMIT:
            mutation_strength = min(mutation_strength * 1.2,
                                    MAX_MUTATION_STRENGTH)
            mutation_rate = min(mutation_rate + 0.02, MAX_MUTATION_RATE)

        furthest_checkpoint = min(
            max([int(index) for index in checkpoint_archive] or [0]),
            max(0, len(checkpoints) - 1),
        )
        is_stagnant = generations_without_improvement > STAGNATION_LIMIT
        print(f"Generation {generation} done. Best fitness: {current_best:.1f}, "
              f"checkpoints: {best_car.checkpoints_passed}/{len(checkpoints)}, "
              f"stagnant: {generations_without_improvement}, "
              f"mutation: {mutation_strength:.2f}/{mutation_rate:.2f}, "
              f"recovery checkpoint: {furthest_checkpoint}")

        save_archive(checkpoint_archive, archive_file)
        cars = next_generation(
            cars,
            start_pos,
            checkpoints=checkpoints,
            mutation_strength=mutation_strength,
            mutation_rate=mutation_rate,
            archive=checkpoint_archive,
            recovery_checkpoint_index=furthest_checkpoint,
            immigrant_count=(IMMIGRANTS_ON_STAGNATION if is_stagnant else 1),
        )
        generation += 1
        generation_start_time = pygame.time.get_ticks()


def main():
    args = parse_args()
    checkpoints_file = args.checkpoints or default_checkpoint_file(args.track)
    weights_file = args.weights or default_weights_file(args.track)
    archive_file = args.archive or default_archive_file(args.track)

    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption(f"Racing neural network - {args.mode}")
    track = load_track(args.track)
    checkpoints = load_checkpoints(checkpoints_file)

    try:
        if args.mode == "test":
            run_test(screen, track, checkpoints, weights_file, args.timeout,
                     args.laps)
        else:
            run_training(screen, track, checkpoints, weights_file, archive_file,
                         args.population, args.timeout)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
