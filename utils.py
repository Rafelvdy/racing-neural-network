import pygame
import math
import random
import json
from copy import deepcopy
from network import random_weights, random_biases, forward

GRASS = (34, 177, 76)
TOLERANCE = 25
START_COLOUR = (255, 255, 255)
START_TOLERANCE = 20
SENSOR_ANGLES = [-60, -30, 0, 30, 60]
SENSOR_MAX_RANGE = 200
CHECKPOINT_GATE_HALF_WIDTH = 64
RECOVERY_START_RADIUS = 18
STUCK_CHECK_INTERVAL_MS = 2500
STUCK_MIN_TRAVEL_DISTANCE = 120
STUCK_MAX_NET_DISPLACEMENT = 40
STUCK_STATIONARY_TIMEOUT_MS = 3000


class car(pygame.sprite.Sprite):
    def __init__(self, location, max_vel, rotation_vel, checkpoints=None,
                 start_checkpoint_index=0, target_laps=1):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("car.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.center = location
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 270
        self.x, self.y = location
        self.acceleration = 0.1
        self.half_w = self.rect.width / 2
        self.half_h = self.rect.height / 2
        self.alive = True
        self.left_start = False
        self.lap_complete = False
        self.target_laps = target_laps
        self.lap_count = 0
        self.start_time = pygame.time.get_ticks()
        self.lap_start_time = self.start_time
        self.lap_time = None
        self.crash_time = None
        self.death_reason = None
        self.start_location = location
        self.max_distance_from_start = 0
        self.closest_distance_to_next = float("inf")
        self.checkpoints = checkpoints or []
        self.next_checkpoint_index = start_checkpoint_index
        self.checkpoints_passed = start_checkpoint_index
        self.is_recovery = start_checkpoint_index > 0
        self.stuck_check_time = pygame.time.get_ticks()
        self.stuck_check_x = self.x
        self.stuck_check_y = self.y
        self.distance_since_stuck_check = 0
        self.last_motion_time = self.stuck_check_time
        self.previous_x = self.x
        self.previous_y = self.y

        if start_checkpoint_index > 0 and self.checkpoints:
            checkpoint_x, checkpoint_y = self.checkpoints[start_checkpoint_index - 1]
            self.x = checkpoint_x + random.uniform(-RECOVERY_START_RADIUS, RECOVERY_START_RADIUS)
            self.y = checkpoint_y + random.uniform(-RECOVERY_START_RADIUS, RECOVERY_START_RADIUS)

        if self.checkpoints:
            target_index = min(start_checkpoint_index, len(self.checkpoints) - 1)
            self.angle = angle_towards(self.x, self.y, *self.checkpoints[target_index])

        self.stuck_check_x = self.x
        self.stuck_check_y = self.y
        self.hidden_weights = random_weights(3, 5)
        self.hidden_biases = random_biases(3)
        self.output_weights = random_weights(2, 3)
        self.output_biases = random_biases(2)

    def get_corners(self, x=None, y=None):
        cx = self.x if x is None else x
        cy = self.y if y is None else y
        return get_rotated_corners(cx, cy, self.half_w, self.half_h, -self.angle)

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, screen, track):
        blit_rotate_center(screen, self.image, (self.x, self.y), self.angle)
        colour = (255, 0, 255) if self.alive else (100, 100, 100)
        for corner_x, corner_y in self.get_corners():
            pygame.draw.circle(screen, colour, (int(corner_x), int(corner_y)), 3)
        for distance, (end_x, end_y) in self.get_sensor_readings(track):
            pygame.draw.line(screen, (0, 200, 255), (self.x, self.y), (end_x, end_y), 1)
            pygame.draw.circle(screen, (0, 200, 255), (int(end_x), int(end_y)), 3)

    def check_crash(self, track):
        if self.alive and any(is_grass(track, x, y) for x, y in self.get_corners()):
            self.alive = False
            self.vel = 0
            self.crash_time = (pygame.time.get_ticks() - self.start_time) / 1000
            self.death_reason = "crash"

    def check_stuck(self):
        if not self.alive:
            return
        now = pygame.time.get_ticks()
        if now - self.last_motion_time >= STUCK_STATIONARY_TIMEOUT_MS:
            self.alive = False
            self.vel = 0
            self.death_reason = "stationary"
            self.crash_time = (now - self.start_time) / 1000
            return
        if now - self.stuck_check_time < STUCK_CHECK_INTERVAL_MS:
            return
        net_displacement = math.hypot(self.x - self.stuck_check_x, self.y - self.stuck_check_y)
        if self.distance_since_stuck_check >= STUCK_MIN_TRAVEL_DISTANCE and net_displacement <= STUCK_MAX_NET_DISPLACEMENT:
            self.alive = False
            self.vel = 0
            self.death_reason = "stuck"
            self.crash_time = (now - self.start_time) / 1000
        self.stuck_check_time = now
        self.stuck_check_x, self.stuck_check_y = self.x, self.y
        self.distance_since_stuck_check = 0

    def check_checkpoints(self):
        if not self.checkpoints or self.next_checkpoint_index >= len(self.checkpoints):
            return
        furthest = None
        for index in range(self.next_checkpoint_index, len(self.checkpoints)):
            if self.crossed_checkpoint_gate(index):
                furthest = index
        target_x, target_y = self.checkpoints[self.next_checkpoint_index]
        distance = math.hypot(self.x - target_x, self.y - target_y)
        if distance < self.closest_distance_to_next:
            self.closest_distance_to_next = distance
        if furthest is not None:
            self.checkpoints_passed = furthest + 1
            self.next_checkpoint_index = furthest + 1
            self.closest_distance_to_next = float("inf")
            self.stuck_check_time = pygame.time.get_ticks()
            self.stuck_check_x, self.stuck_check_y = self.x, self.y
            self.distance_since_stuck_check = 0

    def crossed_checkpoint_gate(self, index):
        start, end = checkpoint_gate(self.checkpoints, index, CHECKPOINT_GATE_HALF_WIDTH)
        return segments_intersect((self.previous_x, self.previous_y), (self.x, self.y), start, end) and self.movement_towards_checkpoint(index)

    def movement_towards_checkpoint(self, index):
        target_x, target_y = self.checkpoints[index]
        tx, ty = checkpoint_tangent(self.checkpoints, index)
        before = (self.previous_x - target_x) * tx + (self.previous_y - target_y) * ty
        after = (self.x - target_x) * tx + (self.y - target_y) * ty
        return before < 0 <= after

    def move_forward(self, track):
        if not self.alive:
            return
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()
        self.check_crash(track)
        self.check_lap(track)
        self.check_checkpoints()
        self.check_stuck()

    def move(self):
        previous_x, previous_y = self.x, self.y
        self.previous_x, self.previous_y = previous_x, previous_y
        radians = math.radians(self.angle)
        self.x += math.sin(radians) * self.vel
        self.y += math.cos(radians) * self.vel
        moved = math.hypot(self.x - previous_x, self.y - previous_y)
        self.distance_since_stuck_check += moved
        if moved > 0.01:
            self.last_motion_time = pygame.time.get_ticks()
        self.max_distance_from_start = max(self.max_distance_from_start, math.hypot(self.x - self.start_location[0], self.y - self.start_location[1]))

    def reduce_speed(self, track):
        if not self.alive:
            return
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()
        self.check_crash(track)
        self.check_lap(track)
        self.check_checkpoints()
        self.check_stuck()

    def break_car(self):
        if self.alive:
            self.vel = max(self.vel - self.acceleration * 2, 0)

    def reset(self, location):
        self.x, self.y = location
        self.vel = 0
        self.alive = True
        self.left_start = False
        self.lap_complete = False
        self.lap_count = 0
        self.lap_time = None
        self.crash_time = None
        self.death_reason = None
        self.max_distance_from_start = 0
        self.next_checkpoint_index = 0
        self.checkpoints_passed = 0
        self.start_time = pygame.time.get_ticks()
        self.lap_start_time = self.start_time
        self.closest_distance_to_next = float("inf")
        self.angle = angle_towards(self.x, self.y, *self.checkpoints[0]) if self.checkpoints else 270
        self.stuck_check_time = self.start_time
        self.stuck_check_x, self.stuck_check_y = self.x, self.y
        self.distance_since_stuck_check = 0
        self.last_motion_time = self.start_time
        self.previous_x, self.previous_y = self.x, self.y

    def check_lap(self, track):
        if not self.alive or self.lap_complete:
            return
        on_start = is_start(track, self.x, self.y)
        if not on_start and not self.left_start:
            self.left_start = True
        elif on_start and self.left_start:
            if self.checkpoints and self.checkpoints_passed < len(self.checkpoints):
                return
            now = pygame.time.get_ticks()
            self.lap_count += 1
            self.lap_time = (now - self.lap_start_time) / 1000
            if self.lap_count >= self.target_laps:
                self.lap_complete = True
            else:
                self.left_start = False
                self.next_checkpoint_index = 0
                self.checkpoints_passed = 0
                self.closest_distance_to_next = float("inf")
                self.lap_start_time = now

    def get_forward_vector(self):
        corners = self.get_corners()
        dx = (corners[2][0] + corners[3][0]) / 2 - self.x
        dy = (corners[2][1] + corners[3][1]) / 2 - self.y
        length = math.hypot(dx, dy)
        return dx / length, dy / length

    def get_sensor_readings(self, track):
        fx, fy = self.get_forward_vector()
        base_angle = math.degrees(math.atan2(fy, fx))
        return [cast_ray(track, self.x, self.y, base_angle + offset, SENSOR_MAX_RANGE) for offset in SENSOR_ANGLES]

    def get_sensor_distances(self, track):
        return [distance / SENSOR_MAX_RANGE for distance, _ in self.get_sensor_readings(track)]

    def drive_with_network(self, track):
        steering, throttle = forward(self.get_sensor_distances(track), self.hidden_weights, self.hidden_biases, self.output_weights, self.output_biases)
        if steering > 0.1:
            self.rotate(right=True)
        elif steering < -0.1:
            self.rotate(left=True)
        if throttle > 0:
            self.move_forward(track)
        else:
            self.reduce_speed(track)

    def fitness(self):
        checkpoint_score = self.checkpoints_passed * 10_000
        distance_score = 0 if self.closest_distance_to_next == float("inf") else max(0, 1_000 - self.closest_distance_to_next)
        if self.lap_complete:
            return 1_000_000 - self.lap_time * 100
        return checkpoint_score + distance_score + self.max_distance_from_start * 0.2


def blit_rotate_center(screen, image, center, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    screen.blit(rotated_image, rotated_image.get_rect(center=center).topleft)


def angle_towards(start_x, start_y, target_x, target_y):
    return math.degrees(math.atan2(target_x - start_x, target_y - start_y))


def checkpoint_tangent(checkpoints, index):
    if len(checkpoints) == 1:
        return 0, 1
    start = checkpoints[max(0, index - 1)]
    end = checkpoints[min(len(checkpoints) - 1, index + 1)]
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length else (0, 1)


def checkpoint_gate(checkpoints, index, half_width):
    cx, cy = checkpoints[index]
    tx, ty = checkpoint_tangent(checkpoints, index)
    nx, ny = -ty, tx
    return ((cx - nx * half_width, cy - ny * half_width), (cx + nx * half_width, cy + ny * half_width))


def segments_intersect(a, b, c, d):
    def orientation(p, q, r):
        value = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        return 0 if abs(value) < 1e-9 else (1 if value > 0 else 2)
    def on_segment(p, q, r):
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    return ((o1 != o2 and o3 != o4) or
            (o1 == 0 and on_segment(a, c, b)) or
            (o2 == 0 and on_segment(a, d, b)) or
            (o3 == 0 and on_segment(c, a, d)) or
            (o4 == 0 and on_segment(c, b, d)))


def find_start_position(surface, target=(255, 255, 255), tolerance=20):
    positions = []
    for x in range(0, surface.get_width(), 2):
        for y in range(0, surface.get_height(), 2):
            if all(abs(value - wanted) <= tolerance for value, wanted in zip(surface.get_at((x, y))[:3], target)):
                positions.append((x, y))
    if not positions:
        raise ValueError("No start-line pixels found")
    return sum(x for x, _ in positions) // len(positions), sum(y for _, y in positions) // len(positions)


def is_grass(surface, x, y):
    if x < 0 or y < 0 or x >= surface.get_width() or y >= surface.get_height():
        return True
    return all(abs(value - wanted) <= TOLERANCE for value, wanted in zip(surface.get_at((int(x), int(y)))[:3], GRASS))


def get_rotated_corners(cx, cy, half_w, half_h, angle_degrees):
    radians = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    return [(cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a)
            for lx, ly in [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]]


def is_start(surface, x, y):
    if x < 0 or y < 0 or x >= surface.get_width() or y >= surface.get_height():
        return False
    return all(abs(value - wanted) <= START_TOLERANCE for value, wanted in zip(surface.get_at((int(x), int(y)))[:3], START_COLOUR))


def cast_ray(track, cx, cy, angle_degrees, max_range=200, step=4):
    radians = math.radians(angle_degrees)
    dx, dy = math.cos(radians), math.sin(radians)
    distance = 0
    while distance < max_range:
        px, py = cx + dx * distance, cy + dy * distance
        if is_grass(track, px, py):
            return distance, (px, py)
        distance += step
    return distance, (cx + dx * distance, cy + dy * distance)


def crossover(parent_a, parent_b):
    def mix_matrix(a, b):
        return [[random.choice([x, y]) for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]
    def mix_vector(a, b):
        return [random.choice([x, y]) for x, y in zip(a, b)]
    return (mix_matrix(parent_a.hidden_weights, parent_b.hidden_weights), mix_vector(parent_a.hidden_biases, parent_b.hidden_biases), mix_matrix(parent_a.output_weights, parent_b.output_weights), mix_vector(parent_a.output_biases, parent_b.output_biases))


def mutate(weights_bundle, rate=0.15, strength=0.4):
    hidden_weights, hidden_biases, output_weights, output_biases = weights_bundle
    def matrix(values):
        return [[value + random.uniform(-strength, strength) if random.random() < rate else value for value in row] for row in values]
    def vector(values):
        return [value + random.uniform(-strength, strength) if random.random() < rate else value for value in values]
    return matrix(hidden_weights), vector(hidden_biases), matrix(output_weights), vector(output_biases)


def weights_from_car(car_obj):
    return {"hidden_weights": deepcopy(car_obj.hidden_weights), "hidden_biases": deepcopy(car_obj.hidden_biases), "output_weights": deepcopy(car_obj.output_weights), "output_biases": deepcopy(car_obj.output_biases)}


def apply_weights(car_obj, weights):
    car_obj.hidden_weights = deepcopy(weights["hidden_weights"])
    car_obj.hidden_biases = deepcopy(weights["hidden_biases"])
    car_obj.output_weights = deepcopy(weights["output_weights"])
    car_obj.output_biases = deepcopy(weights["output_biases"])


def update_checkpoint_archive(cars, archive):
    for current_car in cars:
        index = current_car.checkpoints_passed
        if index == 0:
            continue
        saved = archive.get(str(index))
        if saved is None or (not current_car.is_recovery and current_car.fitness() > saved["fitness"]):
            archive[str(index)] = {"fitness": current_car.fitness(), "weights": weights_from_car(current_car)}


def next_generation(cars, start_pos, checkpoints=None, mutation_strength=0.4, mutation_rate=0.15, archive=None, recovery_checkpoint_index=0, recovery_fraction=0.25, immigrant_count=0):
    normal_cars = [current_car for current_car in cars if not current_car.is_recovery]
    ranked = sorted(normal_cars or cars, key=lambda current_car: current_car.fitness(), reverse=True)
    survivors = ranked[:min(8, len(ranked))]
    new_cars = []
    elite = car(start_pos, 5, 4, checkpoints=checkpoints)
    apply_weights(elite, weights_from_car(survivors[0]))
    new_cars.append(elite)
    recovery_count = int(len(cars) * recovery_fraction) if recovery_checkpoint_index else 0
    if recovery_count and archive and str(recovery_checkpoint_index) in archive:
        recovery_elite = car(start_pos, 5, 4, checkpoints=checkpoints, start_checkpoint_index=recovery_checkpoint_index)
        apply_weights(recovery_elite, archive[str(recovery_checkpoint_index)]["weights"])
        new_cars.append(recovery_elite)
        recovery_count -= 1
    while len(new_cars) < len(cars) - immigrant_count:
        parent_a, parent_b = random.sample(survivors, 2)
        start_index = recovery_checkpoint_index if recovery_count > 0 else 0
        if recovery_count > 0:
            recovery_count -= 1
        child = car(start_pos, 5, 4, checkpoints=checkpoints, start_checkpoint_index=start_index)
        child.hidden_weights, child.hidden_biases, child.output_weights, child.output_biases = mutate(crossover(parent_a, parent_b), rate=mutation_rate, strength=mutation_strength)
        new_cars.append(child)
    while len(new_cars) < len(cars):
        new_cars.append(car(start_pos, 5, 4, checkpoints=checkpoints))
    return new_cars


def save_weights(car_obj, filename="best_car.json"):
    with open(filename, "w") as f:
        json.dump(weights_from_car(car_obj), f)


def load_weights(car_obj, filename="best_car.json"):
    with open(filename) as f:
        weights = json.load(f)
    apply_weights(car_obj, weights)


def save_archive(archive, filename="checkpoint_archive.json"):
    with open(filename, "w") as f:
        json.dump(archive, f)


def load_archive(filename="checkpoint_archive.json"):
    try:
        with open(filename) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def load_checkpoints(filename="checkpoints.json"):
    try:
        with open(filename) as f:
            return json.load(f)
    except FileNotFoundError:
        return []
