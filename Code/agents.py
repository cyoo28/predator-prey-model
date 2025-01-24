import numpy as np
from environment import *


class Prey:
    def __init__(self, position):
        self.position = position

    # Prey follows fixed policy of randomly choosing a movement
    # Generates a potential move for the prey to execute
    def move(self):
        actions = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]] # Stay, Right, Left, Up, Down
        choice = np.random.randint(0,4)
        action = actions[choice]
        new_x = self.position[0] + action[0]
        new_y = self.position[1] + action[1]
        new_position = [new_x, new_y]
        new_position = self.restrict(new_position)
        return new_position

    # Returns prey position on the board
    def get_position(self):
        return self.position

    # Moves prey to new position on the board
    def set_position(self, new_position):
        self.position = new_position
        return self.position

    # Forces stay action if the prey tries to exceed the boundaries of the board
    def restrict(self, position):
        x = position[0]
        y = position[1]
        position = [max(min(x,6),0),max(min(y,6),0)]
        return position


class Predator:

    def __init__(self, position):
        self.position = position

    # Predator follows fixed policy
    # Generates a potential move for the predator to execute that will get it closer to the nearest prey
    def fixed_move(self, prey_positions):
        predator_position = self.get_position()
        closest_index = np.argmin(np.linalg.norm(np.array(predator_position) - np.array(prey_positions), axis=1))
        closest_prey_position = prey_positions[closest_index]

        pp = np.array(closest_prey_position) - np.array(predator_position)

        actions = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        min_idx = np.argmin(np.linalg.norm(pp-actions, axis=1))

        return min_idx

    # Predator follows epsilon greedy policy
    # Generates a potential move for the predator to execute based on choice given by policy
    def move(self, action):
        new_position = self.position + action
        new_position = self.restrict(new_position)
        return new_position

    # Returns predator position on the board
    def get_position(self):
        return self.position

    # Moves predator to new position on the board
    def set_position(self, new_position):
        self.position = new_position
        return self.position

    # Forces stay action if the predator tries to exceed the boundaries of the board
    def restrict(self, position):
        x = position[0]
        y = position[1]
        position = [max(min(x,6),0),max(min(y,6),0)]
        return position

    # Returns euclidean distance between predator and the nearest prey
    def get_local_state(self, prey_positions):
        predator_position = self.get_position()
        closest_index = np.argmin(np.linalg.norm(np.array(predator_position)-np.array(prey_positions),axis=1))
        closest_prey_position = prey_positions[closest_index]
        return predator_position, closest_prey_position
