import pygame
from utils import car, find_start_position

running = True
screen = pygame.display.set_mode((1200,800))
pygame.display.set_caption("Racing neural network")

track = pygame.image.load("track1.png")
track = pygame.transform.scale(track, (1200,800))

start_pos = find_start_position(track)
car1 = car(start_pos, 5,4)

pygame.font.init()
font = pygame.font.SysFont(None, 36)

def draw(screen, images, agent_car, track):
    for img, pos in images:
        screen.blit(img, pos)

    agent_car.draw(screen, track)

    if agent_car.lap_complete:
        text = f"Lap complete: {agent_car.lap_time:.2f}s"
        colour = (0, 255, 0)
    elif not agent_car.alive:
        text = f"Crashed at {agent_car.crash_time:.2f}s"
        colour = (255, 0, 0)
    else:
        elapsed = (pygame.time.get_ticks() - agent_car.start_time) / 1000
        text = f"Time: {elapsed:.2f}s"
        colour = (255, 255, 255)

    text_surface = font.render(text, True, colour)
    screen.blit(text_surface, (10, 10))

    pygame.display.update()

FPS = 60
clock = pygame.time.Clock()
images = [(track, (0,0))]


while running:
    clock.tick(FPS)
    draw(screen, images, car1, track)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    car1.drive_with_network(track)

    if keys := pygame.key.get_pressed():
        if keys[pygame.K_r]:
            car1.reset(start_pos)

  