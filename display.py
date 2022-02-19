import pygame
from pygame.locals import *
import numpy as np
from defines import *
from utilities import Pos


class Display:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.tile_size = 80

        self.window_width = grid_size[0] * self.tile_size
        self.window_height = grid_size[1] * self.tile_size

        pygame.init()
        self.display = pygame.display.set_mode((self.window_width, self.window_height))

    def draw_grid(self):
        for row in range(self.grid_size[1]):
            for col in range(self.grid_size[0]):
                pygame.draw.rect(self.display, BLACK, [col * self.tile_size, row * self.tile_size, self.tile_size, self.tile_size],1)

    def update(self, environment, agent_pos: Pos):
        self.display.fill(WHITE)
        self.draw_grid()

        for row in range(self.grid_size[1]):
            for col in range(self.grid_size[0]):
                el = environment[row][col]
                if el == WALL:
                    pygame.draw.rect(self.display, GREY, [col * self.tile_size, row * self.tile_size, self.tile_size, self.tile_size])
                elif el == BOX:
                    pygame.draw.rect(self.display, ORANGE, [col * self.tile_size + self.tile_size * 0.1, row * self.tile_size + self.tile_size * 0.1, self.tile_size * 0.8, self.tile_size * 0.8])
                elif el == GOAL:
                    pygame.draw.rect(self.display, RED, [col * self.tile_size + self.tile_size * 0.1, row * self.tile_size + self.tile_size * 0.1, self.tile_size * 0.8, self.tile_size * 0.8])
                elif el == GOAL_FILLED:
                    pygame.draw.rect(self.display, GREEN, [col * self.tile_size + self.tile_size * 0.1, row * self.tile_size + self.tile_size * 0.1, self.tile_size * 0.8, self.tile_size * 0.8])

        col = agent_pos.x
        row = agent_pos.y
        pygame.draw.circle(self.display, PURPLE, (col * self.tile_size + 0.5 * self.tile_size, row * self.tile_size + 0.5 * self.tile_size), self.tile_size * 0.3)

        pygame.display.update()