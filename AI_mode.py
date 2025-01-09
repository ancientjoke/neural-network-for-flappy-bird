import pygame
import neat
import os
import pickle
import time

from Bird import Bird
from Pipe import Pipe
from Base import Base

pygame.font.init()
pygame.display.set_caption("Flappy Bird")

WIN_WIDTH = 400
WIN_HEIGHT = 700

BG_IMG = pygame.transform.scale_by(surface=pygame.image.load(os.path.join("images", "bg.png")), factor=1.5)

STAT_FONT = pygame.font.SysFont("comicsans", 25)
GAMEOVER_FONT = pygame.font.SysFont("comicsans", 50)
MENU_FONT = pygame.font.SysFont("comicsans", 20)

def draw_performance_overlay(win, fps, bird_height, pipe_distance, bird_velocity, neural_output):
    overlay_height = WIN_HEIGHT // 3
    overlay_surface = pygame.Surface((WIN_WIDTH, overlay_height))
    overlay_surface.fill((0, 0, 0))
    overlay_surface.set_alpha(180)
    
    win.blit(overlay_surface, (0, WIN_HEIGHT - overlay_height))
    
    y_pos = WIN_HEIGHT - overlay_height + 10
    spacing = 25
    
    metrics = [
        f"FPS: {fps:.1f}",
        f"Bird Height: {bird_height:.1f}",
        f"Distance to Pipe: {pipe_distance:.1f}",
        f"Bird Velocity: {bird_velocity:.2f}",
        f"Neural Output: {neural_output:.3f}",
    ]
    
    for metric in metrics:
        text = MENU_FONT.render(metric, True, (255, 255, 255))
        win.blit(text, (20, y_pos))
        y_pos += spacing

def draw_window(win, bird, pipes, base, score, gameover, time_alive, show_overlay, fps, neural_output):
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Time: " + str(round(time_alive/30, 1)), 1, (255, 255, 255))
    win.blit(text, (10, 10))

    base.draw(win)
    bird.draw(win)

    if gameover:
        text = GAMEOVER_FONT.render("GAMEOVER ", 1, (0, 0, 0))
        win.blit(text, (WIN_WIDTH - (WIN_WIDTH / 8) - text.get_width(), WIN_HEIGHT / 2 - 50))
        
    if show_overlay:
        pipe_distance = abs(bird.x - pipes[0].x) if pipes else 0
        draw_performance_overlay(win, fps, bird.y, pipe_distance, bird.vel, neural_output)
    
    pygame.display.update()

def main():
    with open("winner_genome.pkl", "rb") as f:
        winner = pickle.load(f)

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, "config-feedforward.txt")

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    score = 0
    time_alive = 0
    gameover = False
    show_overlay = False
    bird = Bird(150, 300)
    last_time = time.time()
    fps = 0
    neural_output = 0

    base = Base(600)
    pipes = [Pipe(500)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    run = True

    while run:
        current_time = time.time()
        fps = 1.0 / (current_time - last_time)
        last_time = current_time
        
        clock.tick(30)
        time_alive += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    main()
                if event.key == pygame.K_TAB:
                    show_overlay = not show_overlay

        pipe_ind = 0
        if bird:
            if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            break

        bird.move()

        output = net.activate((bird.y,
                               abs(bird.y - pipes[pipe_ind].height),
                               abs(bird.y - pipes[pipe_ind].bottom)))
        
        neural_output = output[0]

        if output[0] > 0.5:
            bird.jump()

        add_pipe = False
        removed_pipes = []
        for pipe in pipes:
            if pipe.collide(bird):
                gameover = True
            
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                removed_pipes.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            pipes.append(Pipe(450))

        for rp in removed_pipes:
            pipes.remove(rp)

        if bird.y + bird.img.get_height() >= 600 or bird.y < 0:
            gameover = True

        base.move()
        print(f"\rScore: {score} | Time Alive: {round(time_alive/30, 1)}s", end="")
        draw_window(win, bird, pipes, base, score, gameover, time_alive, show_overlay, fps, neural_output)


if __name__ == "__main__":
    main()
