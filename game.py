import argparse
import os
import random

import pygame

from utils import (CHECKPOINT_GATE_HALF_WIDTH, apply_weights, car,
                   checkpoint_gate, crossover, find_start_position,
                   load_archive, load_checkpoints, load_weights, mutate,
                   next_generation, save_archive, save_weights,
                   update_checkpoint_archive, weights_from_car)


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
AVERAGE_SCORE_WEIGHT = 0.3
WORST_TRACK_SCORE_WEIGHT = 0.7
RECOVERY_TRIAL_COUNT = 8
RECOVERY_START_AFTER_STAGNATION = 5
RECOVERY_TIMEOUT_MS = 15000


def parse_args():
    parser = argparse.ArgumentParser(description="Train or test a racing network.")
    parser.add_argument("--mode", choices=("train", "test", "multi-train"),
                        default="train")
    parser.add_argument("--track", default="track1.png")
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--population", type=int, default=POPULATION_SIZE)
    parser.add_argument("--timeout", type=int, default=GENERATION_TIMEOUT_MS)
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--tracks", default="track1.png,track2.png,track3.png")
    parser.add_argument("--checkpoint-files", default=None)
    parser.add_argument("--seed-weights", default=None)
    parser.add_argument("--render-every", type=int, default=3,
                        help="Render every N multi-training frames")
    parser.add_argument("--show-sensors", action="store_true",
                        help="Draw sensor rays and endpoints")
    parser.add_argument("--show-checkpoints", action="store_true",
                        help="Draw checkpoint gates and markers")
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


def draw_scene(surface, cars, track, checkpoints, show_sensors=False,
               show_checkpoints=False):
    surface.blit(track, (0, 0))
    if show_checkpoints:
        for index, (x, y) in enumerate(checkpoints):
            gate_start, gate_end = checkpoint_gate(
                checkpoints,
                index,
                CHECKPOINT_GATE_HALF_WIDTH,
            )
            pygame.draw.line(surface, (255, 165, 0), gate_start, gate_end, 2)
            pygame.draw.circle(surface, (255, 165, 0), (x, y), 4, 1)

    alive_count = 0
    for current_car in cars:
        current_car.draw(surface, track, show_sensors=show_sensors)
        if current_car.alive and not current_car.lap_complete:
            alive_count += 1
    return alive_count


def draw(screen, cars, track, checkpoints, status, show_sensors=False,
         show_checkpoints=False):
    alive_count = draw_scene(screen, cars, track, checkpoints, show_sensors,
                             show_checkpoints)
    text = f"{status} | Alive: {alive_count}/{len(cars)}"
    text_surface = pygame.font.SysFont(None, 30).render(
        text, True, (255, 255, 255)
    )
    screen.blit(text_surface, (10, 10))
    pygame.display.update()


def draw_bar(surface, label, value, x, y, width=220):
    pygame.draw.rect(surface, (55, 55, 55), (x, y, width, 16))
    midpoint = x + width // 2
    pygame.draw.line(surface, (130, 130, 130), (midpoint, y),
                     (midpoint, y + 16), 1)
    magnitude = int(max(-1.0, min(1.0, value)) * width / 2)
    colour = (70, 190, 110) if value >= 0 else (220, 90, 80)
    if magnitude >= 0:
        pygame.draw.rect(surface, colour, (midpoint, y, magnitude, 16))
    else:
        pygame.draw.rect(surface, colour, (midpoint + magnitude, y,
                                           -magnitude, 16))
    text = pygame.font.SysFont(None, 22).render(
        f"{label}: {value:+.2f}", True, (235, 235, 235)
    )
    surface.blit(text, (x, y - 22))


def draw_neural_observer(surface, observer, generation):
    surface.fill((30, 30, 30))
    title = pygame.font.SysFont(None, 28).render(
        f"NEURAL OBSERVER | Generation {generation}",
        True,
        (255, 255, 255),
    )
    surface.blit(title, (20, 18))
    status = pygame.font.SysFont(None, 22).render(
        f"Best Track 1 observer | checkpoints: {observer.checkpoints_passed} | "
        f"alive: {observer.alive}",
        True,
        (190, 190, 190),
    )
    surface.blit(status, (20, 55))

    font = pygame.font.SysFont(None, 22)
    surface.blit(font.render("Sensors", True, (255, 190, 80)), (20, 100))
    for index, value in enumerate(observer.sensor_values):
        draw_bar(surface, f"S{index + 1}", value, 25, 135 + index * 30,
                 width=250)

    surface.blit(font.render("Hidden neurons", True, (100, 190, 255)), (315, 100))
    for index, value in enumerate(observer.hidden_values):
        draw_bar(surface, f"H{index + 1}", value, 320, 135 + index * 45,
                 width=250)

    surface.blit(font.render("Outputs and action", True, (120, 230, 140)), (20, 290))
    draw_bar(surface, "Steering", observer.output_values[0], 25, 330,
             width=250)
    draw_bar(surface, "Throttle", observer.output_values[1], 320, 330,
             width=250)
    action = ("RIGHT" if observer.output_values[0] > 0.1 else
              "LEFT" if observer.output_values[0] < -0.1 else "STRAIGHT")
    throttle = "ACCELERATE" if observer.output_values[1] > 0 else "SLOW"
    surface.blit(font.render(f"Decision: {action} + {throttle}", True,
                             (255, 255, 255)), (25, 380))


def draw_multi_grid(screen, cars_by_track, tracks, checkpoints_by_track,
                    generation, show_sensors=False, show_checkpoints=False):
    panel_size = (SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2)
    panel_positions = [(0, 0), (panel_size[0], 0),
                       (0, panel_size[1]), (panel_size[0], panel_size[1])]
    screen.fill((20, 20, 20))

    for track_index, (cars, track, checkpoints) in enumerate(
            zip(cars_by_track, tracks, checkpoints_by_track)):
        scene = pygame.Surface(SCREEN_SIZE)
        alive_count = draw_scene(scene, cars, track, checkpoints, show_sensors,
                     show_checkpoints)
        panel = pygame.transform.smoothscale(scene, panel_size)
        position = panel_positions[track_index]
        screen.blit(panel, position)
        label = pygame.font.SysFont(None, 24).render(
            f"Track {track_index + 1} | Alive: {alive_count}/{len(cars)}",
            True,
            (255, 255, 255),
        )
        screen.blit(label, (position[0] + 8, position[1] + 8))

    info_surface = pygame.Surface(panel_size)
    observer = max(cars_by_track[0], key=lambda current_car: current_car.fitness())
    draw_neural_observer(info_surface, observer, generation)
    screen.blit(info_surface, panel_positions[3])
    pygame.display.update()


def run_test(screen, track, checkpoints, weights_file, timeout_ms, target_laps,
             show_sensors=False, show_checkpoints=False):
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
            draw(screen, [test_car], track, checkpoints, "TEST", show_sensors,
                 show_checkpoints)

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


def normalized_track_score(current_car, checkpoints, timeout_ms):
    checkpoint_total = max(1, len(checkpoints))
    progress = min(1.0, current_car.checkpoints_passed / checkpoint_total)
    if current_car.lap_complete:
        time_bonus = max(0.0, 1.0 - current_car.lap_time / (timeout_ms / 1000))
        return 1.0 + time_bonus * 0.1
    if current_car.closest_distance_to_next == float("inf"):
        approach_bonus = 0.0
    else:
        approach_bonus = max(0.0, 1.0 - current_car.closest_distance_to_next / 200)
    return progress + approach_bonus * 0.01


def run_recovery_trials(tracks, checkpoints_by_track, start_positions,
                        genome_cars, cars_by_track, weakest_track_index,
                        timeout_ms, mutation_strength, mutation_rate):
    normal_cars = cars_by_track[weakest_track_index]
    checkpoint_count = len(checkpoints_by_track[weakest_track_index])
    furthest_checkpoint = max(
        (current_car.checkpoints_passed for current_car in normal_cars),
        default=0,
    )
    recovery_start = min(furthest_checkpoint, max(0, checkpoint_count - 1))
    if recovery_start == 0:
        return None, 0

    ranked_indices = sorted(
        range(len(normal_cars)),
        key=lambda index: normalized_track_score(
            normal_cars[index],
            checkpoints_by_track[weakest_track_index],
            timeout_ms,
        ),
        reverse=True,
    )
    trial_cars = []
    for index in ranked_indices[:RECOVERY_TRIAL_COUNT]:
        trial_car = car(
            start_positions[weakest_track_index],
            5,
            4,
            checkpoints=checkpoints_by_track[weakest_track_index],
            start_checkpoint_index=recovery_start,
        )
        genome = genome_cars[index]
        weights = mutate(
            (
                genome.hidden_weights,
                genome.hidden_biases,
                genome.output_weights,
                genome.output_biases,
            ),
            rate=mutation_rate,
            strength=mutation_strength,
        )
        trial_car.hidden_weights, trial_car.hidden_biases, \
            trial_car.output_weights, trial_car.output_biases = weights
        trial_cars.append(trial_car)

    recovery_start_time = pygame.time.get_ticks()
    recovery_timeout = min(timeout_ms, RECOVERY_TIMEOUT_MS)
    while trial_cars and pygame.time.get_ticks() - recovery_start_time < recovery_timeout:
        for trial_car in trial_cars:
            if trial_car.alive and not trial_car.lap_complete:
                trial_car.drive_with_network(tracks[weakest_track_index])
        if all((not trial_car.alive) or trial_car.lap_complete
               for trial_car in trial_cars):
            break

    best_trial = max(
        trial_cars,
        key=lambda current_car: normalized_track_score(
            current_car,
            checkpoints_by_track[weakest_track_index],
            recovery_timeout,
        ),
    )
    best_score = normalized_track_score(
        best_trial,
        checkpoints_by_track[weakest_track_index],
        recovery_timeout,
    )
    return weights_from_car(best_trial), best_score


def run_multi_training(screen, tracks, checkpoints_by_track, weights_file,
                       population_size, timeout_ms, seed_weight_files=None,
                       render_every=3, show_sensors=False,
                       show_checkpoints=False):
    start_positions = [find_start_position(track) for track in tracks]
    genome_cars = [car(start_positions[0], 5, 4, checkpoints=checkpoints_by_track[0])
                   for _ in range(population_size)]
    loaded_existing_weights = False
    try:
        load_weights(genome_cars[0], weights_file)
        loaded_existing_weights = True
    except FileNotFoundError:
        pass

    if not loaded_existing_weights and seed_weight_files:
        for genome_car, seed_file in zip(genome_cars[1:], seed_weight_files):
            try:
                load_weights(genome_car, seed_file)
            except FileNotFoundError:
                continue

    clock = pygame.time.Clock()
    generation = 1
    best_score_ever = float("-inf")
    generations_without_improvement = 0
    mutation_strength = BASE_MUTATION_STRENGTH
    mutation_rate = BASE_MUTATION_RATE
    running = True
    frame_number = 0

    while running:
        cars_by_track = []
        for track_index, track in enumerate(tracks):
            track_cars = []
            for genome_car in genome_cars:
                track_car = car(
                    start_positions[track_index],
                    5,
                    4,
                    checkpoints=checkpoints_by_track[track_index],
                )
                apply_weights(track_car, weights_from_car(genome_car))
                track_cars.append(track_car)
            cars_by_track.append(track_cars)

        generation_start_time = pygame.time.get_ticks()
        track_scores = [0.0] * population_size
        per_track_scores = [[] for _ in tracks]
        track_done = [False] * len(tracks)

        while not all(track_done):
            clock.tick(FPS)
            frame_number += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not running:
                break

            for track_index, track_cars in enumerate(cars_by_track):
                if track_done[track_index]:
                    continue
                for current_car in track_cars:
                    if current_car.alive and not current_car.lap_complete:
                        current_car.drive_with_network(tracks[track_index])
                track_done[track_index] = (
                    all((not current_car.alive) or current_car.lap_complete
                        for current_car in track_cars)
                    or pygame.time.get_ticks() - generation_start_time > timeout_ms
                )

            if frame_number % max(1, render_every) == 0:
                draw_multi_grid(
                    screen,
                    cars_by_track,
                    tracks,
                    checkpoints_by_track,
                    generation,
                    show_sensors,
                    show_checkpoints,
                )

        if not running:
            break

        for genome_index in range(population_size):
            scores = [normalized_track_score(
                cars_by_track[track_index][genome_index],
                checkpoints_by_track[track_index],
                timeout_ms,
            ) for track_index in range(len(tracks))]
            average_score = sum(scores) / len(scores)
            track_scores[genome_index] = (
                AVERAGE_SCORE_WEIGHT * average_score
                + WORST_TRACK_SCORE_WEIGHT * min(scores)
            )
            for track_index, score in enumerate(scores):
                per_track_scores[track_index].append(score)

        best_index = max(range(population_size), key=track_scores.__getitem__)
        current_best = track_scores[best_index]
        best_track_scores = [scores[best_index] for scores in per_track_scores]
        hardest_track_index = min(
            range(len(best_track_scores)),
            key=best_track_scores.__getitem__,
        )
        recovery_weights = None
        recovery_score = None
        if generations_without_improvement >= RECOVERY_START_AFTER_STAGNATION:
            recovery_weights, recovery_score = run_recovery_trials(
                tracks,
                checkpoints_by_track,
                start_positions,
                genome_cars,
                cars_by_track,
                hardest_track_index,
                timeout_ms,
                mutation_strength,
                mutation_rate,
            )
        existing_best = track_scores[0] if loaded_existing_weights else float("-inf")
        if current_best > best_score_ever and (not loaded_existing_weights
                                               or generation > 1
                                               or current_best > existing_best):
            best_score_ever = current_best
            generations_without_improvement = 0
            mutation_strength = BASE_MUTATION_STRENGTH
            mutation_rate = BASE_MUTATION_RATE
            save_weights(genome_cars[best_index], weights_file)
        else:
            best_score_ever = max(best_score_ever, existing_best)
            generations_without_improvement += 1

        if generations_without_improvement > STAGNATION_LIMIT:
            mutation_strength = min(mutation_strength * 1.2,
                                    MAX_MUTATION_STRENGTH)
            mutation_rate = min(mutation_rate + 0.02, MAX_MUTATION_RATE)

        ranked_indices = sorted(range(population_size),
                                key=track_scores.__getitem__, reverse=True)
        survivors = [genome_cars[index] for index in ranked_indices[:8]]
        next_genomes = [genome_cars[best_index]]
        if recovery_weights is not None:
            recovery_candidate = car(
                start_positions[0],
                5,
                4,
                checkpoints=checkpoints_by_track[0],
            )
            apply_weights(recovery_candidate, recovery_weights)
            next_genomes.append(recovery_candidate)
        immigrant_count = IMMIGRANTS_ON_STAGNATION if (
            generations_without_improvement > STAGNATION_LIMIT) else 1

        while len(next_genomes) < population_size - immigrant_count:
            parent_a, parent_b = random.sample(survivors, 2)
            child = car(start_positions[0], 5, 4,
                        checkpoints=checkpoints_by_track[0])
            weights = mutate(crossover(parent_a, parent_b), rate=mutation_rate,
                             strength=mutation_strength)
            child.hidden_weights, child.hidden_biases, child.output_weights, child.output_biases = weights
            next_genomes.append(child)

        while len(next_genomes) < population_size:
            next_genomes.append(car(start_positions[0], 5, 4,
                                    checkpoints=checkpoints_by_track[0]))

        genome_cars = next_genomes
        loaded_existing_weights = False
        print(f"Multi-generation {generation} done. Combined score: "
              f"{current_best:.4f}, track scores: "
              f"{[round(score, 3) for score in best_track_scores]}, "
              f"hardest track: {hardest_track_index + 1}, "
              f"recovery: {recovery_score if recovery_score is not None else '-'}, "
              f"stagnant: {generations_without_improvement}, "
              f"mutation: {mutation_strength:.2f}/{mutation_rate:.2f}")
        generation += 1


def main():
    args = parse_args()
    checkpoints_file = args.checkpoints or default_checkpoint_file(args.track)
    weights_file = args.weights or default_weights_file(args.track)
    archive_file = args.archive or default_archive_file(args.track)

    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption(f"Racing neural network - {args.mode}")

    try:
        if args.mode == "test":
            track = load_track(args.track)
            checkpoints = load_checkpoints(checkpoints_file)
            run_test(screen, track, checkpoints, weights_file, args.timeout,
                     args.laps, args.show_sensors, args.show_checkpoints)
        elif args.mode == "multi-train":
            track_files = [track.strip() for track in args.tracks.split(",")
                           if track.strip()]
            if args.checkpoint_files:
                checkpoint_files = [checkpoint.strip() for checkpoint in
                                    args.checkpoint_files.split(",")
                                    if checkpoint.strip()]
            else:
                checkpoint_files = [default_checkpoint_file(track)
                                    for track in track_files]
            if len(track_files) != len(checkpoint_files):
                raise ValueError("Each multi-track image needs one checkpoint file")
            tracks = [load_track(track_file) for track_file in track_files]
            checkpoints_by_track = [load_checkpoints(checkpoint_file)
                                    for checkpoint_file in checkpoint_files]
            run_multi_training(screen, tracks, checkpoints_by_track,
                               args.weights or "shared_best.json",
                         args.population, args.timeout,
                         [default_weights_file(track_file)
                          for track_file in track_files]
                         if args.seed_weights is None
                         else [seed.strip() for seed in
                             args.seed_weights.split(",")
                             if seed.strip()],
                         args.render_every, args.show_sensors,
                         args.show_checkpoints)
        else:
            track = load_track(args.track)
            checkpoints = load_checkpoints(checkpoints_file)
            run_training(screen, track, checkpoints, weights_file, archive_file,
                         args.population, args.timeout)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
