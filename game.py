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
    moved = False

    if car1.alive:
        if keys[pygame.K_a]:
            car1.rotate(left=True)
        if keys[pygame.K_d]:
            car1.rotate(right=True)
        if keys[pygame.K_w]:
            moved = True
            car1.move_forward(track)

        if keys[pygame.K_s]:
            car1.break_car()
        
        if not moved:
            car1.reduce_speed(track)

        if keys[pygame.K_r]:
            car1.reset(start_pos)

  