import pygame
import math
from network import random_weights, random_biases, forward
import random

GRASS = (34, 177, 76)
TOLERANCE = 25
START_COLOUR = (255, 255, 255)
START_TOLERANCE = 20
SENSOR_ANGLES = [-60, -30, 0, 30, 60]
SENSOR_MAX_RANGE = 200

class car(pygame.sprite.Sprite):
    def __init__(self, location, max_vel, rotation_vel):
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
        self.start_time = pygame.time.get_ticks()
        self.lap_time = None
        self.crash_time = None
        self.distance_travelled = 0

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

        for corner_x, corner_y in self.get_corners():
            colour = (255, 0, 255) if self.alive else (100, 100, 100)
            pygame.draw.circle(screen, colour, (int(corner_x), int(corner_y)), 3)

        for distance, (end_x, end_y) in self.get_sensor_readings(track):
            pygame.draw.line(screen, (0, 200, 255), (self.x, self.y), (end_x, end_y), 1)
            pygame.draw.circle(screen, (0, 200, 255), (int(end_x), int(end_y)), 3)

    def check_crash(self, track):
        if not self.alive:
            return
        corners = self.get_corners()
        if any(is_grass(track, cx, cy) for cx, cy in corners):
            self.alive = False
            self.vel = 0
            self.crash_time = (pygame.time.get_ticks() - self.start_time) / 1000

    def move_forward(self, track):
        if not self.alive:
            return
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()
        self.check_crash(track)
        self.check_lap(track)

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel
        self.distance_travelled += math.hypot(horizontal, vertical)
        self.x += horizontal
        self.y += vertical

    def reduce_speed(self, track):
        if not self.alive:
            return
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()
        self.check_crash(track)

    def break_car(self):
        if not self.alive:
            return
        self.vel = max(self.vel - self.acceleration * 2, 0)

    def reset(self, location):
        self.x, self.y = location
        self.vel = 0
        self.angle = 270
        self.alive = True
        self.left_start = False
        self.lap_complete = False
        self.lap_time = None
        self.crash_time = None
        self.start_time = pygame.time.get_ticks()

    def check_lap(self, track):
        if not self.alive or self.lap_complete:
            return
        on_start = is_start(track, self.x, self.y)

        if not on_start and not self.left_start:
            self.left_start = True

        elif on_start and self.left_start:
            self.lap_complete = True
            self.lap_time = (pygame.time.get_ticks() - self.start_time) / 1000

    def get_forward_vector(self):
        corners = self.get_corners()
        front_x = (corners[2][0] + corners[3][0]) / 2
        front_y = (corners[2][1] + corners[3][1]) / 2
        dx = front_x - self.x
        dy = front_y - self.y
        length = math.hypot(dx, dy)
        return dx / length, dy / length

    def get_sensor_readings(self, track):
        fx, fy = self.get_forward_vector()
        base_angle = math.degrees(math.atan2(fy, fx))

        readings = []
        for offset in SENSOR_ANGLES:
            angle = base_angle + offset
            distance, endpoint = cast_ray(track, self.x, self.y, angle, SENSOR_MAX_RANGE)
            readings.append((distance, endpoint))
        return readings
    
    def get_sensor_distances(self, track):
        readings = self.get_sensor_readings(track)
        return [distance / SENSOR_MAX_RANGE for distance, _ in readings]

    def drive_with_network(self, track):
        inputs = self.get_sensor_distances(track)
        steering, throttle = forward(inputs, self.hidden_weights, self.hidden_biases,
                                    self.output_weights, self.output_biases)

        if steering > 0.1:
            self.rotate(right=True)
        elif steering < -0.1:
            self.rotate(left=True)

        if throttle > 0:
            self.move_forward(track)
        else:
            self.reduce_speed(track)

    def fitness(self):
        if self.lap_complete:
            return 100000 + self.distance_travelled - (self.lap_time * 100)
        return self.distance_travelled
    
        
def blit_rotate_center(screen, image, center, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=center)
    screen.blit(rotated_image, new_rect.topleft)

def find_start_position(surface, target=(255, 255, 255), tolerance=20):
    xs, ys = [], []
    width, height = surface.get_size()
    for x in range(0, width, 2):
        for y in range(0, height, 2):
            r, g, b, *_ = surface.get_at((x, y))
            if all(abs(c - t) <= tolerance for c, t in zip((r, g, b), target)):
                xs.append(x)
                ys.append(y)
    if not xs:
        raise ValueError("No start-line pixels found")
    return sum(xs) // len(xs), sum(ys) // len(ys)

def is_grass(surface, x, y):
    if x < 0 or y < 0 or x >= surface.get_width() or y >= surface.get_height():
        return True
    r, g, b, *_ = surface.get_at((int(x), int(y)))
    return all(abs(c - t) <= TOLERANCE for c, t in zip((r, g, b), GRASS))

def get_rotated_corners(cx, cy, half_w, half_h, angle_degrees):
    radians = math.radians(angle_degrees)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)

    local_corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]

    world_corners = []
    for lx, ly in local_corners:
        rotated_x = lx * cos_a - ly * sin_a
        rotated_y = lx * sin_a + ly * cos_a
        world_corners.append((cx + rotated_x, cy + rotated_y))

    return world_corners

def is_start(surface, x, y):
    if x < 0 or y < 0 or x >= surface.get_width() or y >= surface.get_height():
        return False
    r, g, b, *_ = surface.get_at((int(x), int(y)))
    return all(abs(c - t) <= START_TOLERANCE for c, t in zip((r, g, b), START_COLOUR))

def cast_ray(track, cx, cy, angle_degrees, max_range=200, step=4):
    radians = math.radians(angle_degrees)
    dx = math.cos(radians)
    dy = math.sin(radians)

    distance = 0
    while distance < max_range:
        px = cx + dx * distance
        py = cy + dy * distance
        if is_grass(track, px, py):
            break
        distance += step

    return distance, (px, py)

def crossover(parent_a, parent_b):
    def mix_matrix(ma, mb):
        return [[random.choice([va, vb]) for va, vb in zip(ra, rb)] for ra, rb in zip(ma, mb)]
    def mix_vector(va, vb):
        return [random.choice([a, b]) for a, b in zip(va, vb)]

    return (
        mix_matrix(parent_a.hidden_weights, parent_b.hidden_weights),
        mix_vector(parent_a.hidden_biases, parent_b.hidden_biases),
        mix_matrix(parent_a.output_weights, parent_b.output_weights),
        mix_vector(parent_a.output_biases, parent_b.output_biases),
    )


def mutate(weights_bundle, rate=0.15, strength=0.4):
    hidden_weights, hidden_biases, output_weights, output_biases = weights_bundle

    def mutate_matrix(m):
        return [[v + random.uniform(-strength, strength) if random.random() < rate else v
                 for v in row] for row in m]
    def mutate_vector(v):
        return [x + random.uniform(-strength, strength) if random.random() < rate else x
                for x in v]

    return (
        mutate_matrix(hidden_weights),
        mutate_vector(hidden_biases),
        mutate_matrix(output_weights),
        mutate_vector(output_biases),
    )


def next_generation(cars, start_pos):
    ranked = sorted(cars, key=lambda c: c.fitness(), reverse=True)
    survivors = ranked[:4]

    new_cars = []
    best = survivors[0]
    elite = car(start_pos, 5, 4)
    elite.hidden_weights = best.hidden_weights
    elite.hidden_biases = best.hidden_biases
    elite.output_weights = best.output_weights
    elite.output_biases = best.output_biases
    new_cars.append(elite)

    while len(new_cars) < len(cars):
        parent_a, parent_b = random.sample(survivors, 2)
        mutated = mutate(crossover(parent_a, parent_b))

        child = car(start_pos, 5, 4)
        child.hidden_weights, child.hidden_biases, child.output_weights, child.output_biases = mutated
        new_cars.append(child)

    return new_cars