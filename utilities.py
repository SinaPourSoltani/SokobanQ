import sys
from defines import *
from dataclasses import dataclass


COLS = 0
ROWS = 0

@dataclass
class Pos:
    x: int
    y: int

    def __str__(self):
        return Utilities.double_digit_stringify_int(self.x) + Utilities.double_digit_stringify_int(self.y)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __add__(self, other):
        return Pos(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Pos(self.x - other.x, self.y - other.y)

    def __lt__(self, other):
        return (self.y * COLS + self.x) < (other.y * COLS + other.x)

    def __gt__(self, other):
        return (self.y * COLS + self.x) > (other.y * COLS + other.x)


class Utilities:
    @staticmethod
    def double_digit_stringify_int(num):
        return '0' + str(num) if num < 10 else str(num)

    @staticmethod
    def get_sorted_indeces(indeces):
        box_indeces = [int(indeces[i:i + 2]) for i in range(0, len(indeces), 2)]
        box_indeces.sort()
        mapped = map(Utilities.double_digit_stringify_int, box_indeces)
        return "".join(mapped)

    @staticmethod
    def create_space_and_index_conversion_dictionaries(rows, cols, environment):
        value = 0
        pos2index = {}
        global COLS, ROWS
        COLS = cols
        ROWS = rows
        for row in range(rows):
            for col in range(cols):
                if environment[row][col] != WALL:
                    key = Pos(col, row)
                    pos2index[key] = value
                    value += 1

        index2pos = {v: k for k, v in pos2index.items()}
        return pos2index, index2pos

    @staticmethod
    def combinations(n, k, min_n=0, accumulator=None):
        if accumulator is None:
            accumulator = []
        if k == 0:
            return [accumulator]
        else:
            return [l for x in range(min_n, n)
                    for l in Utilities.combinations(n, k - 1, x + 1, accumulator + [x + 1])]

    @staticmethod
    def create_boxes_combinatorics_conversion_dictionaries(num_spaces, num_boxes, index2pos, corners):
        boxes_index2states = {}
        value = 0
        combinations = Utilities.combinations(num_spaces, num_boxes)
        for combi in combinations:
            is_deadlock = False
            key = ()
            for index in combi:
                pos = index2pos[index-1]
                key += pos,
            key = tuple(sorted(key))
            value += 1
            for corner in corners:
                for k in key:
                    if corner == k:
                        is_deadlock = True
            dict_value = -1 * value if is_deadlock else value
            boxes_index2states[key] = dict_value
        boxes_states2index = {v: k for k, v in boxes_index2states.items()}
        return boxes_index2states, boxes_states2index