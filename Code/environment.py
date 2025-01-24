import gym
from gym import spaces
import pygame
import numpy as np
from itertools import product
from random import sample
from agents import *


def choice_to_action(choices):
    c2a_dict = {0: [0, 0],
                1: [1, 0],
                2: [-1, 0],
                3: [0, 1],
                4: [0,-1],
                5: [1, 1],
                6: [1, -1],
                7: [-1, 1],
                8: [-1, -1]}
    actions = []
    for choice in choices:
        actions.append(c2a_dict[choice])
    return actions


class PredatorPreyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, args):
        self.done = False
        self.terminal = False
        self.successes = 0

        # Defining relevant parameters
        self.size = args['gridsize']
        self.prey_count = args['prey_count'] # For this there are 4 initial prey
        self.predator_count = args['predator_count'] # For this there are 4 initial predators
        # All agents start alive
        self.dead_prey = [False, False, False, False]
        self.dead_predator = [False, False, False, False]

        self.state_size = (self.prey_count + self.predator_count) * 2
        self.action_size = 8

        self.prey_active_list: list[Prey] = []
        self.predator_active_list: list[Predator] = []
        # Initializing all n (8) agents to unique random positions on the board
        self.positions = sample(list(product(range(7), repeat=2)), k=self.prey_count + self.predator_count)
        for prey_id in range(self.prey_count):
            self.prey_active_list.append(Prey(self.positions[prey_id]))
        for predator_id in range(self.predator_count):
            self.predator_active_list.append(Predator(self.positions[self.prey_count + predator_id]))

        #self.window_size = 512

        # Defining 9 possible actions for predator learning agents
        self.action_space = spaces.Discrete(8)

        # Defining observation space for the states of all n (8) agents
        self.observation_space = spaces.Dict(
            {
                "predators": spaces.Box(0, self.size - 1, shape=(self.predator_count, 2), dtype=int),
                "preys": spaces.Box(0, self.size - 1, shape=(self.prey_count, 2), dtype=int)
            }
        )

    # Function to get the states of all active agents
    def get_state(self):
        predator_states = []
        prey_states = []
        for predator in self.predator_active_list:
            predator_states.append(predator.get_position())
        for prey in self.prey_active_list:
            prey_states.append(prey.get_position())

        return {"predators": predator_states,
                "prey": prey_states
                }

    # Initialize the environment
    def reset(self):
        self.done = False
        self.terminal = False
        self.successes = 0

        # Reset status of all agents
        self.dead_prey = [False, False, False, False]
        self.dead_predator = [False, False, False, False]
        self.prey_active_list: list[Prey] = []
        self.predator_active_list: list[Predator] = []

        # Restarting board to initial state defined in initialization
        for prey_id in range(self.prey_count):
            self.prey_active_list.append(Prey(self.positions[prey_id]))
        for predator_id in range(self.predator_count):
            self.predator_active_list.append(Predator(self.positions[self.prey_count + predator_id]))

        # Getting the initial state of the board
        initial_state = self.get_state()
        return initial_state

    # Function to decide whether an attack was a success
    def check_attack(self, prey_new_moves, predator_new_moves, hunting_predators):
        dead_prey_list = []
        dead_predator_list = []
        for hunting_idx in range(len(hunting_predators)):
            if hunting_predators[hunting_idx] in hunting_predators[0:hunting_idx] + hunting_predators[hunting_idx + 1:]:
                # predators successfully attack and prey dies
                for prey_idx in range(len(prey_new_moves)):
                    if hunting_predators[hunting_idx] == prey_new_moves[prey_idx]:
                        dead_prey_list.append(prey_idx)
            else:
                # predator fails to attack and dies
                for predator_idx in range(len(predator_new_moves)):
                    if hunting_predators[hunting_idx] == predator_new_moves[predator_idx]:
                        dead_predator_list.append(predator_idx)
        return [dead_prey_list, dead_predator_list]

    # Take a step in the environment
    def step(self, predator_choices):
        # Calculating next move for prey agents
        prey_new_moves = []
        for index, prey in enumerate(self.prey_active_list):
            if not self.dead_prey[index]:
                prey_new_moves.append(prey.move())
            else:
                prey_new_moves.append(prey.get_position())
        # Calculating next move for predator agents
        predator_actions = choice_to_action(predator_choices)
        predator_new_moves = []
        for index, predator in enumerate(self.predator_active_list):
            if not self.dead_predator:
                predator_new_moves.append(predator.move(predator.move(predator_actions[index])))
            else:
                predator_new_moves.append(predator.get_position())

        # Checking if any agents moves cause a collision with another agent
        # Agents will only move if there is no collision
        for index, prey in enumerate(prey_new_moves):
            if prey not in prey_new_moves[0:index] + prey_new_moves[index + 1:] and prey not in predator_new_moves:
                self.prey_active_list[index].set_position(prey)
        hunting_predators = []
        for index, predator in enumerate(predator_new_moves):
            if predator not in predator_new_moves[0:index] + predator_new_moves[index + 1:] and predator not in prey_new_moves:
                self.predator_active_list[index].set_position(predator)
            # However, if there is a collision between a predator and prey,
            # it must be determined if the predator is actually hunting (i.e. moving)
            elif predator in prey_new_moves and predator_actions[index] != [0, 0]:
                hunting_predators.append(predator)

        # If the predator is determined to be hunting, check the result of the attack
        [dead_prey_list, dead_predator_list] = self.check_attack(prey_new_moves, predator_new_moves, hunting_predators)

        # Update life status of all agents
        # If an agent has died, remove from the world
        for idx in dead_prey_list:
            self.dead_prey[idx] = True
            self.prey_active_list[idx].set_position(-10, -10)
        for idx in dead_predator_list:
            self.dead_predator[idx] = True
            self.predator_active_list[idx].set_position(-10, -10)

        # done flag indicates if game is finished, terminal flag indicates if game is finished SUCCESSFULLY
        # i.e. all prey have been removed and there remains at least two predators

        # Check the terminal case where all prey have been removed
        if False not in self.dead_prey:
            self.done = True
            self.terminal = True
            reward = 0
        # Check the terminal case where all predators have been removed
        elif False not in self.dead_predator:
            self.done = True
            self.terminal = False
        # If not a terminal case, then penalize the euclidean distance between each predator and the closest prey
        else:
            distances = 0
            prey_positions = []
            for prey in self.prey_active_list:
                prey_positions.append(prey.get_position())
            for predator in self.predator_active_list:
                predator_position = predator.get_position()
                distances += min(np.linalg.norm(np.array(predator_position)-np.array(prey_positions), axis=1))
            reward = -distances

        # Update next state of the environment
        next_state = self.get_state()
        return next_state, reward, self.done, self.terminal

    def render(self):
        # Haven't implemented render yet
        return 1
