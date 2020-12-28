import math
import time
import os
import random
import neat
import pygame
pygame.font.init()
WIN_WIDTH = 500
WIN_HEIGHT = 800
GEN = 0
Rate = 30
pygame.display.set_caption("Flappy Bird?")

scriptDir = os.path.dirname(__file__)
imgs = os.path.join(scriptDir, 'imgs')

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join(imgs, "bird1.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join(imgs, "bird2.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join(imgs, "bird3.png")))]

PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join(imgs, "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join(imgs, "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join(imgs, "bg.png")))
STAT_FONT = pygame.font.SysFont("comicsans", 50)

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5  # animation speed

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count +=1

        d = self.vel*self.tick_count + 1.5*self.tick_count**2  # calculates speed going up or down

        if d >= 16:
            d = 16

        if d < 0:
            d -=2 # =- instead asdasodhoasgdys for a while

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:  # if going up do max rotation
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:  # if going down slowly get to 90 degrees
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count +=1

        if(self.img_count < self.ANIMATION_TIME):
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:  # if less than 10
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4+1:
            self.img = self.IMGS[0]
            self.img_count = 0

            if self.tilt <= -80:  # falling?dont flap
                self.img = self.IMGS[1]
                self.img_count = self.ANIMATION_TIME*2  # starts back at animation frame of 10

        rotated_image = pygame.transform.rotate(self.img, self.tilt)  # rot from center not topleft
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    GAP = 300 # why is this here?
    VEL = 10

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 180 #160

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False # for counting score later
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50,450)  #set height
        self.top = self.height - self.PIPE_TOP.get_height()  #bottom height plus gap so it can place top
        self.bottom = self.height + self.gap

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP,(self.x, self.top)) # drawing pipes
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird): # pixel perfect? - yes
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x , self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y)) # how far away pipe is from bird so it compares the right pixel

        b_point = bird_mask.overlap(bottom_mask, bottom_offset) # actual collision checking bot and top pipes
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True

        return False

class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self): # just cycles images behind one another
        self.x1 -= self.VEL # for moving illusion ^
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score, gen, alive, fitness, pipe_ind):
    global Rate
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    for bird in birds:
        bird.draw(win)
    base.draw(win)
    if alive < 5 and fitness > 16:
        for bird in birds:
            try: # pipe ind isnt always on and might crash on first run
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255)) # white
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10)) # always in screen

    text = STAT_FONT.render("Fitness: " + str(round(fitness)), 1, (255, 255, 255)) # white
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 50)) # always in screen

    text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255)) # white
    win.blit(text, (10 ,10)) # always in screen

    text = STAT_FONT.render("Rate: " + str(Rate), 1, (255, 255, 255)) # white
    win.blit(text, (10 ,50)) # always in screen

    text = STAT_FONT.render("Alive: " + str(alive), 1, (255, 255, 255)) # white
    win.blit(text, (10 ,90)) # always in screen




    pygame.display.update()


def main(genomes, config):
    global GEN, Rate
    GEN += 1 # every gen this 'game' loop is run
    nets = []
    ge = []
    birds = [] # Bird object array for many bird at once ^ etc
    MFitness = 0
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0
    run = True

    while run:
        clock.tick(Rate)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    Rate+=5
                if event.key == pygame.K_DOWN:
                    Rate-=5

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width(): # check pipe to input values - if passed
                pipe_ind = 1 # if passed look at the next pipe
        else:
            run = False # no birds left? quit the game / next generation
            break
            
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1 # you lasted a lil ,get a treat - 1 fitness per sec

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom))) # input values to network

            if output[0] > 0: # only one output neuron else we would check more
                bird.jump()


        base.move()
        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds): # for bird in birds but we can get position in list
                if pipe.collide(bird):
                    ge[x].fitness -=1 # so birds dont ram into pipe to get higher fitness
                    birds.pop(x) # remove and return item at index
                    nets.pop(x)
                    ge.pop(x)
                if not pipe.passed and pipe.x < bird.x: # if passed pipe
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0: # if outside screen
                    rem.append(pipe) # say goodbye - to the trashmobile

            pipe.move()
        if add_pipe: # if you made it trough pipe you get a treat
            score += 1
            for g in ge:
                g.fitness +=5 # make sure ramming is not rewarded - no ramming into pipes for higher fitness
            pipes.append(Pipe(600)) # add new pipe yes

        for r in rem:
            pipes.remove(r) # clear recycle bin
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0: # if it hits the floor # no going over no
                birds.pop(x) # remove and return item at index
                nets.pop(x)
                ge.pop(x)
        for g in ge: # just tracking max fitness
            if g.fitness > MFitness:
                MFitness = g.fitness

        Alive = len(birds)
        draw_window(win, birds, pipes, base, score, GEN, Alive, MFitness, pipe_ind)




def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True)) # h
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 5000) # , fitness, 50 - max generations
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
