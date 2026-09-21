import pygame
import math

GRASS = (34, 177, 76)
TOLERANCE = 25

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

    def get_corners(self, x=None, y=None):
        cx = self.x if x is None else x
        cy = self.y if y is None else y
        return get_rotated_corners(cx, cy, self.half_w, self.half_h, -self.angle)

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, screen):
        blit_rotate_center(screen, self.image, (self.x, self.y), self.angle)
        for corner_x, corner_y in self.get_corners():           # debug dots
            colour = (255, 0, 255) if self.alive else (100, 100, 100)
            pygame.draw.circle(screen, colour, (int(corner_x), int(corner_y)), 3)

    def check_crash(self, track):
        if not self.alive:
            return
        corners = self.get_corners()
        if any(is_grass(track, cx, cy) for cx, cy in corners):
            self.alive = False
            self.vel = 0

    def move_forward(self, track):
        if not self.alive:
            return
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()
        self.check_crash(track)

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel
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