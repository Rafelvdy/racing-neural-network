import pygame

class car(pygame.sprite.Sprite):
    def __init__(self, location, max_vel, rotation_vel):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("car.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.center = location
        self.max_vel - max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel


def blit_rotate_center(screen, image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topLeft = top_left).center)
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