import pygame

class car(pygame.sprite.Sprite):
    def __init__(self, location):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("car.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.center = location


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