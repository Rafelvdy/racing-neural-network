import pygame
from utils import car, find_start_position


running = True
screen = pygame.display.set_mode((1200,800))
pygame.display.set_caption("Racing neural network")

track = pygame.image.load("track1.png")
track = pygame.transform.scale(track, (1200,800))

start_pos = find_start_position(track)
car1 = car(start_pos)

FPS = 60
clock = pygame.time.Clock()

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  

    screen.blit(track, (0,0))
    screen.blit(car1.image,car1.rect)
    pygame.display.update()