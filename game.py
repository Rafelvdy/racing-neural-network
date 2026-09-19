import pygame
from utils import car, find_start_position


running = True
screen = pygame.display.set_mode((1200,800))
pygame.display.set_caption("Racing neural network")

track = pygame.image.load("track1.png")
track = pygame.transform.scale(track, (1200,800))

start_pos = find_start_position(track)
car1 = car(start_pos, 5,4)

def draw(screen, images, agent_car):
    for img, pos in images:
        screen.blit(img, pos)

    agent_car.draw(screen)
    pygame.display.update()

FPS = 60
clock = pygame.time.Clock()
images = [(track, (0,0))]


while running:
    clock.tick(FPS)

    draw(screen, images, car1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  


    keys = pygame.key.get_pressed()

    if keys[pygame.k_a]:
        car1.rotate(Left=True)
    elif keys[pygame.k_d]:
        car1.rotate(Right=True)
    
    