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
GEN = 0

def draw_window(win, birds, pipes, base, score, gameover, gen, alive, max_time):
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(text, (10, 10))

    text = STAT_FONT.render("Alive: " + str(alive), 1, (255, 255, 255))
    win.blit(text, (10, 50))
    
    text = STAT_FONT.render("Time: " + str(round(max_time/30, 1)), 1, (255, 255, 255))
    win.blit(text, (10, 90))

    base.draw(win)

    for bird in birds:
        bird.draw(win)

    if gameover:
        text = GAMEOVER_FONT.render("GAMEOVER ", 1, (0, 0, 0))
        win.blit(text, (WIN_WIDTH - (WIN_WIDTH / 8) - text.get_width(), WIN_HEIGHT / 2 - 50))
    pygame.display.update()

def genome_evaluation(genomes, config):
    global GEN
    GEN += 1
    score = 0
    gameover = False
    network_list = []
    genome_list = []
    birds = []
    survival_times = []
    max_survival_time = 0

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        network_list.append(net)
        birds.append(Bird(150, 300))
        g.fitness = 0
        genome_list.append(g)
        survival_times.append(0)

    base = Base(600)
    pipes = [Pipe(500)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    run = True

    while run:
        clock.tick(120)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            break

        for x, bird in enumerate(birds):
            survival_times[x] += 1
            max_survival_time = max(max(survival_times), max_survival_time)
            
            bird.move()
            genome_list[x].fitness += 0.1
            
            if score < 10:
                survival_bonus = 0.05 * (1 - score/10)
                genome_list[x].fitness += survival_bonus
            
            optimal_height = WIN_HEIGHT/2
            height_diff = abs(bird.y - optimal_height)
            height_fitness = 0.01 * (1 - height_diff/optimal_height)
            genome_list[x].fitness += height_fitness

            output = network_list[x].activate((bird.y,
                                           abs(bird.y - pipes[pipe_ind].height),
                                           abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        removed_pipes = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    collision_penalty = 2 if score < 5 else 1
                    genome_list[x].fitness -= collision_penalty
                    birds.pop(x)
                    network_list.pop(x)
                    genome_list.pop(x)
                    survival_times.pop(x)

                if bird.y + bird.img.get_height() >= 600 or bird.y < 0:
                    boundary_penalty = 2 if score < 5 else 1
                    genome_list[x].fitness -= boundary_penalty
                    birds.pop(x)
                    network_list.pop(x)
                    genome_list.pop(x)
                    survival_times.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                removed_pipes.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            pipe_reward = 5 + (score // 10)
            for g in genome_list:
                g.fitness += pipe_reward
            pipes.append(Pipe(450))

        for rp in removed_pipes:
            pipes.remove(rp)

        if score > 100:
            break
            
        base.move()
        print(f"\rGeneration: {GEN} | Score: {score} | Alive: {len(birds)} | Time Alive: {round(max_survival_time/30, 1)}s", end="")
        draw_window(win, birds, pipes, base, score, gameover, GEN, len(birds), max_survival_time)

def run_config(configuration_file_path, save_file_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, configuration_file_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(genome_evaluation, 50)

    with open(save_file_path, 'wb') as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    save_path = os.path.join(local_dir, "winner_genome.pkl") 
    run_config(config_path, save_path)
