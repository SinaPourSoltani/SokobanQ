import sys
import numpy as np
from copy import deepcopy
import math
from time import sleep
from dataclasses import dataclass
import argparse

import pygame
from pygame.locals import *
from display import Display

from defines import *
from utilities import Utilities as utils
from utilities import Pos

Q_THRESHOLD = 0.001

SLOW = 15
FAST = 1000

discount_rate = 0.95
epsilon = 0.15
alpha = 0.2

pos2index = index2pos = {}
boxes_positions2state = boxes_state2positions = {}


def setup_lookups(rows, cols, env, num_spaces, num_boxes, corners):
    global pos2index, index2pos
    pos2index, index2pos = utils.create_space_and_index_conversion_dictionaries(rows, cols, env)
    global boxes_positions2state, boxes_state2positions
    boxes_positions2state, boxes_state2positions = \
        utils.create_boxes_combinatorics_conversion_dictionaries(num_spaces, num_boxes, index2pos, corners)


@dataclass
class AgentState:
    state: int

    def as_position(self):
        return index2pos[self.state]

    def is_outside_environment(self):
        return True if self.state is None else False

    def set_state(self, agent_pos):
        self.state = pos2index.get(agent_pos, None)


@dataclass
class BoxesState:
    state: int

    def as_positions(self):
        return boxes_state2positions[self.state]

    def set_state(self, box_positions):
        box_positions = tuple(sorted(box_positions))
        self.state = boxes_positions2state.get(box_positions, None)


@dataclass
class State:
    boxes: BoxesState
    agent: AgentState


class SokobanQ:
    def __init__(self, map_file_path):
        self.map_file_path = map_file_path
        self.rows = 0
        self.cols = 0

        self.environment = []
        self.Q_prev = []
        self.Q = []
        self.moves = [Pos(-1, 0), Pos(0, -1), Pos(1, 0), Pos(0, 1)]

        self.num_boxes, self.num_spaces = self.read_map()
        self.corners = self.detect_corners()
        print(self.environment)

        # Setup hash tables
        setup_lookups(self.rows, self.cols, self.environment, self.num_spaces, self.num_boxes, self.corners)

        # Initial states
        self.initial_state, self.goal_state = self.set_initial_states(including_goals=True)
        self.goal_positions = self.goal_state.as_positions()
        self.state = self.initial_state
        self.goal_reached = False

        self.reset_Q_table()

        # Display
        self.TICKS = FAST
        self.display = Display((self.cols, self.rows))
        self.display.update(self.environment, self.state.agent.as_position())
        self.start_game()

    def read_map(self):
        map_file = open(self.map_file_path, 'r')
        lines = map_file.readlines()
        map_rows = [line.replace('\n', '') for line in lines]
        self.rows = len(map_rows)
        self.cols = len(map_rows[0])
        self.environment = np.chararray((self.rows, self.cols), unicode=True)
        for y, row in enumerate(map_rows):
            for x, el in enumerate(row):
                self.environment[y][x] = el

        num_boxes = num_spaces = 0
        for y, row in enumerate(map_rows):
            for x, el in enumerate(row):
                if self.environment[y][x] == BOX:
                    num_boxes += 1
                if self.environment[y][x] != WALL:
                    num_spaces += 1
        return num_boxes, num_spaces

    def detect_corners(self):
        corners = []
        for y in range(self.rows):
            for x in range(self.cols):
                if self.environment[y][x] != WALL and self.environment[y][x] != GOAL:
                    moves = []
                    num_surrounding_walls = 0
                    env_pos = Pos(x,y)
                    for move in self.moves:
                        surrounding_space = env_pos + move
                        if self.environment[surrounding_space.y][surrounding_space.x] == WALL:
                            moves.append(move)
                            num_surrounding_walls += 1
                    if num_surrounding_walls >= 2:
                        for i, m in enumerate(moves):
                            if(self.moves.index(m) == self.moves.index(moves[i-1]) + 1 or self.moves.index(m) == self.moves.index(moves[i-1]) - 1):
                                corners.append(env_pos)
                                break
                            elif(self.moves.index(m) == 3 and self.moves.index(moves[i-1]) == 0 or self.moves.index(m) == 0 and self.moves.index(moves[i-1]) == 3):
                                corners.append(env_pos)
                                break
        return corners

    def update_environment(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.environment[row][col] != WALL:
                    self.environment[row][col] = PASSAGE

        for box_pos in self.state.boxes.as_positions():
            self.environment[box_pos.y][box_pos.x] = BOX

        for gs in self.goal_positions:
            self.environment[gs.y][gs.x] = GOAL_FILLED if self.environment[gs.y][gs.x] == BOX else GOAL

    def set_initial_states(self, including_goals: bool):
        agent_state = 0
        boxes_positions = ()
        goals_positions = ()
        for row in range(self.rows):
            for col in range(self.cols):
                if self.environment[row][col] == BOX:
                    boxes_positions += Pos(col, row),
                elif including_goals and self.environment[row][col] == GOAL:
                    goals_positions += Pos(col, row),
                elif self.environment[row][col] == AGENT:
                    agent_state = pos2index[Pos(col, row)]
                    self.environment[row][col] = PASSAGE
        if including_goals:
            return State(BoxesState(boxes_positions2state[boxes_positions]), AgentState(agent_state)), \
                   BoxesState(boxes_positions2state[goals_positions])
        else:
            return State(BoxesState(boxes_positions2state[boxes_positions]), AgentState(agent_state))

    def reset_Q_table(self):
        num_box_states = len(boxes_state2positions.keys())
        num_agent_states = self.num_spaces
        num_actions = len(possible_actions)
        assert num_box_states == math.comb(self.num_spaces, self.num_boxes)

        # Q[box_states][agent_state][action]
        self.Q_prev = np.zeros((num_box_states,num_agent_states,num_actions))
        self.Q = np.random.rand(num_box_states, num_agent_states, num_actions)
        self.Q *= 0.01

    def set_prev_Q_table(self):
        self.Q_prev = deepcopy(self.Q)

    def get_biggest_Q_val_difference(self):
        return np.amax(np.maximum(self.Q_prev,self.Q))

    def get_reward(self, s: State, a, steps):
        reward = -0.1

        next_s = self.get_next_state(s, a)
        if next_s == s: # next state is None (invalid) therefore previous state is returned
            reward -= 10
            #print("R",reward)
            return reward

        if next_s.boxes.state < 0:  # box is pushed to a corner
            reward -= 10
            return reward

        box_positions = s.boxes.as_positions()
        next_box_positions = next_s.boxes.as_positions()

        boxes_on_goal = 0
        next_boxes_on_goal = 0

        box_goal_distance = 0
        next_box_goal_distance = 0

        for box in box_positions:
            for goal in self.goal_positions:
                if box == goal:
                    boxes_on_goal += 1
                else:
                    distance = abs(box.x - goal.x) + abs(box.y - goal.y)
                    box_goal_distance += distance
            box_goal_distance /= len(self.goal_positions)

        for next_box in next_box_positions:
            for goal in self.goal_positions:
                if next_box == goal:
                    next_boxes_on_goal += 1
                else:
                    distance = abs(next_box.x - goal.x) + abs(next_box.y - goal.y)
                    next_box_goal_distance += distance
            next_box_goal_distance /= len(self.goal_positions)

        reward += (next_boxes_on_goal - boxes_on_goal) * 100 * 100 * 1/steps
        reward += -(next_box_goal_distance - box_goal_distance) * 10

        if next_boxes_on_goal == len(self.goal_positions):
            print("WON")
            self.goal_reached = True
            reward += 1000

        #print("R",reward)
        return reward

    def get_best_action(self, s: State):
        best_action = LEFT
        best_q_val = self.Q[s.boxes.state][s.agent.state][best_action]
        for a in possible_actions:
            if self.Q[s.boxes.state][s.agent.state][a] > best_q_val:
                best_q_val = self.Q[s.boxes.state][s.agent.state][a]
                best_action = a
        return best_action

    def get_next_action(self, s: State):
        best_action = self.get_best_action(s)

        probability = np.random.uniform(0, 1)

        if probability < (1 - epsilon):
            return best_action
        else:
            return np.random.choice(possible_actions)

    def get_next_state(self, s: State, a):
        next_s = deepcopy(s)

        next_agent_pos = next_s.agent.as_position() + self.moves[a]
        next_s.agent.set_state(next_agent_pos)

        if next_s.agent.state is None:
            return deepcopy(s)

        boxes_positions = next_s.boxes.as_positions()
        new_boxes_positions = ()
        for box_pos in boxes_positions:
            if next_agent_pos == box_pos:
                new_box_pos = box_pos + self.moves[a]
                new_boxes_positions += new_box_pos,
            else:
                new_boxes_positions += box_pos,

        next_s.boxes.set_state(new_boxes_positions)

        if next_s.boxes.state is None:
            return deepcopy(s)

        return deepcopy(next_s)

    def start_game(self, human_player=False):
        if human_player:
            while True:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            self.state = self.get_next_state(self.state, UP)
                        if event.key == pygame.K_DOWN:
                            self.state = self.get_next_state(self.state, DOWN)
                        if event.key == pygame.K_LEFT:
                            self.state = self.get_next_state(self.state, LEFT)
                        if event.key == pygame.K_RIGHT:
                            self.state = self.get_next_state(self.state, RIGHT)
                # updates
                self.update_environment()
                self.display.update(self.environment, self.state.agent.as_position())
                pygame.time.Clock().tick(self.TICKS)
        else:
            biggest_q_diff = 1
            optimal_steps = 1000
            episode = 0
            steps = 0
            cumulative_reward = 0

            self.state = deepcopy(self.initial_state)
            global epsilon
            tmp_epsilon = epsilon
            while biggest_q_diff > Q_THRESHOLD:
                episode += 1
                if episode % 10 == 0:
                    tmp_epsilon = epsilon
                    epsilon = 0
                else:
                    epsilon = tmp_epsilon
                    epsilon -= 0.001
                print("Episode:", episode, "Steps:", steps, "ε:", epsilon, "𝝨R", cumulative_reward)
                steps = 0
                cumulative_reward = 0
                while not self.state.agent.is_outside_environment() and steps < 50 and not self.goal_reached:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                self.TICKS = SLOW
                        if event.type == pygame.KEYUP:
                            if event.key == pygame.K_SPACE:
                                self.TICKS = FAST
                    steps += 1
                    a = self.get_next_action(self.state)
                    next_s = self.get_next_state(self.state, a)

                    if next_s.agent.is_outside_environment():
                        next_s = deepcopy(self.state)

                    reward = self.get_reward(self.state, a, steps)
                    cumulative_reward += reward
                    #print(self.Q[self.state.boxes.state][self.state.agent.state][a])
                    self.Q[self.state.boxes.state][self.state.agent.state][a] += alpha * (reward + discount_rate * self.Q[next_s.boxes.state][next_s.agent.state][self.get_best_action(next_s)] - self.Q[self.state.boxes.state][self.state.agent.state][a])
                    #print(self.Q[self.state.boxes.state][self.state.agent.state][a])
                    self.state = deepcopy(next_s)

                    self.update_environment()
                    self.display.update(self.environment, self.state.agent.as_position())
                    pygame.time.Clock().tick(self.TICKS)

                if steps < optimal_steps:
                    optimal_steps = steps

                self.state = deepcopy(self.initial_state)
                self.goal_reached = False

                biggest_q_diff = self.get_biggest_Q_val_difference()
                self.set_prev_Q_table()


def parse_args(args):
    defualt_map_path = "maps/map.txt"
    parser = argparse.ArgumentParser(description="Start a Q-learning agent on a Sokoban map.")
    parser.add_argument('-m','--map',type=str,default=defualt_map_path, help=f"set the path to the file containing the map. Default: {defualt_map_path}.")

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    sokoban = SokobanQ(map_file_path=args.map)
