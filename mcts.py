from __future__ import division

import time
import math
import random


def random_policy(state):
    while not state.is_terminal():
        try:
            action = random.choice(state.get_possible_actions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.take_action(action)
    return state.get_reward()


class TreeNode:
    def __init__(self, state, parent):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.num_visits = 0
        self.total_reward = 0
        self.children = {}


def expand(node):
    actions = node.state.get_possible_actions()
    for action in actions:
        if action not in node.children:
            new_node = TreeNode(node.state.take_action(action), node)
            node.children[action] = new_node
            if len(actions) == len(node.children):
                node.is_fully_expanded = True
            return new_node

    raise Exception("Should never reach here")


def back_propagate(node, reward):
    while node is not None:
        node.num_visits += 1
        node.total_reward += reward
        node = node.parent


def get_best_child(node, exploration_value):
    best_value = float("-inf")
    best_nodes = []
    for child in node.children.values():
        node_value = child.total_reward / child.num_visits + exploration_value * math.sqrt(
            2 * math.log(node.num_visits) / child.num_visits)
        if node_value > best_value:
            best_value = node_value
            best_nodes = [child]
        elif node_value == best_value:
            best_nodes.append(child)
    return random.choice(best_nodes)


def get_action(root, best_child):
    for action, node in root.children.items():
        if node is best_child:
            return action


class MCTS:
    def __init__(self, time_limit=None, iteration_limit=None, exploration_constant=1 / math.sqrt(2),
                 rollout_policy=random_policy):
        if (time_limit is not None) and (iteration_limit is not None):
            raise ValueError("Cannot have both a time limit and an iteration limit")
        self.limit_type = 'time' if time_limit is not None else 'iterations'
        self.time_limit = time_limit
        self.search_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.rollout = rollout_policy

    def search(self, initial_state):
        self.root = TreeNode(initial_state, None)

        if self.limit_type == 'time':
            time_limit = time.time() + self.time_limit / 1000
            while time.time() < time_limit:
                self.execute_round()
        else:
            for i in range(self.search_limit):
                self.execute_round()

        best_child = get_best_child(self.root, 0)
        return get_action(self.root, best_child)

    def execute_round(self):
        node = self.select_node(self.root)
        reward = self.rollout(node.state)
        back_propagate(node, reward)

    def select_node(self, node):
        while not node.is_terminal:
            node = get_best_child(node, self.exploration_constant) if node.is_fully_expanded else expand(node)
        return node
