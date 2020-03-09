from copy import deepcopy
from mcts import MCTS
from namedlist import namedlist
from collections import namedtuple
from enum import Enum
from itertools import combinations, chain, product

Tile = namedtuple('Tile', 'x y')
Village = namedtuple('Village', 'x y')
Unit = namedlist('Unit', 'x y player level hp attack defence kill')
City = namedlist('City', 'x y capital range player level pop trained')
Resource = namedtuple('Resource', 'type x y')
ResourceType = Enum('ResourceType', 'fruit animal null')
TechType = Enum('TechType', 'hunting organization farm')
Stage = Enum('Stage', 'tech resource train move attack end')

next_stage = {
    Stage.tech: Stage.resource,
    Stage.resource: Stage.train,
    Stage.train: Stage.move,
    Stage.move: Stage.attack,
    Stage.attack: Stage.end,
    Stage.end: Stage.tech
}

tech_cost = {
    TechType.hunting: 2,
    TechType.organization: 2,
    TechType.farm: 5
}

resource_cost = {
    ResourceType.fruit: 2,
    ResourceType.animal: 2
}

max_hp = 10
unit_cost = 2

resource_tech = {
    ResourceType.fruit: TechType.organization,
    ResourceType.animal: TechType.hunting
}


def power_set(s):
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def total_tech_cost(t_combo):
    return sum(tech_cost[tech] for tech in t_combo)


def within_border(city, a):
    return abs(city.x - a.x) <= city.range and abs(city.y - a.y) < city.range


def inc_pop(city, delta):
    city.pop += delta
    if city.pop >= city.level:
        city.pop -= city.level
        city.level += 1
        return True
    return False


def get_star(city):
    return city.level + city.capital


class GameAction:
    def __init__(self, player, stage, data):
        self.player = player
        self.stage = stage
        self.data = data

    def __str__(self):
        return str((self.player, self.stage.name, self.data))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                self.player == other.player and
                self.stage == other.stage and
                self.data == other.data)

    def __hash__(self):
        if self.data is None:
            return hash((self.stage, -1, self.player))

        hash_data = 0
        if self.stage == Stage.tech:
            hash_data = frozenset(self.data)
        elif self.stage == Stage.resource:
            hash_data = frozenset(self.data)
        elif self.stage == Stage.train:
            hash_data = frozenset(map(tuple, self.data))
        elif self.stage in [Stage.move, Stage.attack]:
            hash_data = tuple((tuple(u), x, y) for u, x, y in self.data)
        return hash((self.stage, hash_data, self.player))


class GameState:
    def __init__(self):
        self.size = 5
        self.num_player = 2
        self.cur_player = 0
        self.turn_num = 0
        self.villages = [Village(0, 4), Village(2, 2)]
        self.resources = [Resource(ResourceType.fruit, 0, 1), Resource(ResourceType.animal, 3, 4)]
        self.cities = [[City(1, 1, 1, 1, 0, 1, 0, 0)],
                       [City(3, 3, 1, 1, 1, 1, 0, 0)]]
        self.visible = [{(1 + dx, 1 + dy) for dx, dy in product([-1, 1, 0], repeat=2)},
                        {(3 + dx, 3 + dy) for dx, dy in product([-1, 1, 0], repeat=2)}]
        self.units = [[], []]
        self.stars = [100, 100]
        self.tech = [{TechType.hunting}, {TechType.organization}]
        self.stage = Stage.tech
        self.actions = None

    def within_map(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def attack(self, u, v, retaliate=True):
        if v.hp <= 0:
            return False

        damage = u.attack + ((u.attack - v.defence) * u.hp if u.attack > v.defence else 0)
        v.hp -= damage
        if v.hp <= 0:
            u.kill += 1
            return True

        if retaliate:
            self.attack(v, u, False)

        return False

    def create_game_action(self, action_data):
        return GameAction(self.cur_player, self.stage, action_data)

    def get_possible_actions(self):
        if self.actions is not None:
            return self.actions

        j = self.cur_player
        stars = self.stars[j]
        actions = []

        if self.stage == Stage.tech:
            new_techs = [t for t in TechType if t not in self.tech[j] and tech_cost[t] <= stars]
            actions = [self.create_game_action(t_combo) for t_combo in power_set(new_techs) if
                       total_tech_cost(t_combo) <= stars]

        if self.stage == Stage.resource:
            avail_resources = []
            for r in self.resources:
                if resource_tech[r.type] in self.tech[j]:
                    if any(within_border(city, r) for city in self.cities[j]):
                        avail_resources.append(r)

            for r_combo in power_set(avail_resources):
                if sum(resource_cost[r.type] for r in r_combo) <= stars:
                    actions.append(self.create_game_action(r_combo))

        occupied = {(u.x, u.y): u for u in chain(*self.units)}

        if self.stage == Stage.train:
            trainable_cities = [c for c in self.cities[j] if c.trained < c.level and (c.x, c.y) not in occupied]
            actions = [self.create_game_action(c_combo) for c_combo in power_set(trainable_cities) if
                       len(c_combo) * unit_cost <= stars]

        if self.stage == Stage.move:
            movable = []
            for u in self.units[j]:
                movable.append([(u, u.x, u.y)])  # no move
                for dx, dy in product([-1, 0, 1], repeat=2):
                    vx, vy = u.x + dx, u.y + dy
                    if self.within_map(vx, vy) and (vx, vy) not in occupied:
                        movable[-1].append((u, vx, vy))
            actions = [self.create_game_action(move) for move in product(*movable)]

        if self.stage == Stage.attack:
            targets = []
            for u in self.units[j]:
                targets.append([(u, -1, -1)])  # no attack
                for dx, dy in product([-1, 0, 1], repeat=2):
                    vx, vy = u.x + dx, u.y + dy
                    if self.within_map(vx, vy) and (vx, vy) in occupied and occupied[vx, vy] != j:
                        targets[-1].append((u, vx, vy))
            actions = [self.create_game_action(attack) for attack in product(*targets)]

        if self.stage != Stage.end and not actions:
            actions = [self.create_game_action(None)]
        else:
            actions = [GameAction(self.cur_player, Stage.end, 1)]

        self.actions = actions
        return self.actions

    def take_action(self, a):
        state = deepcopy(self)
        j, stage, data = a.player, a.stage, a.data

        assert (j == state.cur_player)
        assert (stage == state.stage)
        state.actions = None

        if data is None:
            state.stage = next_stage[state.stage]
            return state

        if state.stage == Stage.tech:
            total_cost = sum(tech_cost[tech] for tech in data) + len(state.cities[j])
            if total_cost <= state.stars[j]:
                state.stars[j] -= total_cost
                state.tech[j].update(data)

        # TODO: check valid resource data
        if state.stage == Stage.resource:
            total_cost = sum(resource_cost[r.type] for r in data)
            if total_cost <= state.stars[j]:
                state.stars[j] -= total_cost
                state.resources = set(state.resources) - set(data)
                for r in data:
                    for city in state.cities[j]:
                        if any(within_border(city, r) for city in state.cities[j]):
                            inc_pop(city, 1)
                            break

        # TODO: check valid data
        if state.stage == Stage.train:
            total_cost = unit_cost * len(data)
            if total_cost <= state.stars[j]:
                state.stars[j] -= total_cost
                for c in data:
                    state.units[j].append(Unit(c.x, c.y, j, 1, max_hp, 2, 2, 0))
                    c.trained += 1

        if state.stage == Stage.move:
            for u, x, y in data:
                u.x, u.y = x, y
                for dx, dy in product([-1, 0, 1], repeat=2):
                    if state.within_map(x + dx, y + dy):
                        state.visible[j].add((x + dx, y + dy))

        occupied = {(u.x, u.y): u for u in chain(*state.units)}

        if state.stage == Stage.attack:
            for u, x, y in data:
                if x != -1:
                    v = occupied[x, y]
                    state.attack(u, v, True)

            dead = [u for u in chain(*state.units) if u.hp <= 0]
            for u in dead:
                state.units[u.player].remove(u)

        if state.stage != Stage.end:
            state.stage = next_stage[state.stage]
            return state

        # heal all units before end
        for u in filter(lambda u: u.hp < max_hp, state.units[j]):
            healed_hp = 2 if any(within_border(city, u) for city in state.cities[j]) else 4
            u.hp = min(u.hp + healed_hp, max_hp)

        next_player = (j + 1) % state.num_player
        while len(state.cities[next_player]) == 0 and next_player != j:
            state.turn_num += next_player == 0
            next_player = (next_player + 1) % state.num_player

        state.turn_num += next_player == 0
        state.cur_player = next_player
        state.stage = Stage.tech
        state.stars[j] += sum(get_star(city) for city in state.cities[j])
        return state

    def is_terminal(self):
        surviving = sum(len(city_set) > 0 for city_set in self.cities)
        return surviving <= 1 or self.turn_num >= 30

    def get_points(self, j):
        if j > self.num_player:
            return 0
        city_points = sum(c.level * 100 + c.pop * 50 + c.trained * 10 for c in self.cities[j])
        kill_points = sum(u.kill * 10 for u in self.units[j])
        territory_points = sum(any(within_border(c, Tile(x, y)) for c in self.cities[j])
                               for x, y in product(range(self.size), repeat=2))
        return city_points + kill_points + territory_points * 10

    def get_reward(self):
        surviving = sum(len(city_set) > 0 for city_set in self.cities)
        if surviving == 1:
            return len(self.cities[self.cur_player]) * 10000
        return self.get_points(self.cur_player)


import random
import numpy as np


def uniform_dist(N):
    return [1 / N] * N


import math


def gen_myopic_policy(evaluate, link=math.exp):
    def myopic_policy(state):
        actions = state.get_available_actions()
        action_prob = [link(evaluate(state.take_action(a))) for a in actions]
        total_prob = sum(action_prob)
        action_prob = [p/total_prob for p in action_prob]
        return actions, action_prob
    return myopic_policy


def random_policy(state):
    actions = state.get_available_actions()
    action_prob = uniform_dist(len(actions))
    return actions, action_prob


def random_value(state):
    return random.random()


def self_value(state):
    return state.get_reward()


def random_Q(state, action):
    return random.random()


class Agents:
    def __init__(self, values, policies):
        self.num_players = len(policies)
        self.policies = policies
        self.values = values

        for i, policy in enumerate(self.policies):
            if policy is None:
                self.policies[i] = random_policy

        for i, value in enumerate(self.values):
            if value is None:
                self.values[i] = random_value

    def get_rollout(self):
        def rollout(state):
            while not state.is_terminal():
                j = state.cur_player
                policy = self.policies[j](state)
                if sum(policy[1]) < 0.1:
                    policy[1] = uniform_dist(len(policy[1]))
                try:
                    a = np.random.choice(policy[0], policy[1])
                except IndexError:
                    raise Exception("Non-terminal state has no possible actions: " + str(state))
                state = state.take_action(a)
            return self.values[j](state)

        return rollout


def run():
    initial_state = GameState()

    for i in range(100):
        ts = MCTS(time_limit=1000)
        action = ts.search(initial_state=initial_state)
        print(action)
        initial_state = initial_state.take_action(action)

