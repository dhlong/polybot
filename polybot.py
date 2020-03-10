from copy import deepcopy
from mcts import MCTS
from namedlist import namedlist
from collections import namedtuple
from enum import Enum
from itertools import combinations, chain, product
import random
import numpy as np
import math

Tile = namedtuple('Tile', 'x y')
Village = namedtuple('Village', 'x y')
Unit = namedlist('Unit', 'x y player level hp attack defence kill')
City = namedlist('City', 'x y capital range player level pop trained')
Resource = namedtuple('Resource', 'type x y')
ResourceType = Enum('ResourceType', 'fruit animal null')

org_tech = 'null organization shields farming construction'
hunt_tech = 'hunting forestry archery mathematics'
ride_tech = 'riding'
climb_tech = 'climbing'
all_tech = ' '.join([org_tech, hunt_tech, ride_tech, climb_tech])
TechType = Enum('TechType', all_tech)
Stage = Enum('Stage', 'tech resource train move attack capture end')

next_stage = {
    Stage.tech: Stage.resource,
    Stage.resource: Stage.train,
    Stage.train: Stage.move,
    Stage.move: Stage.attack,
    Stage.attack: Stage.capture,
    Stage.capture: Stage.end,
    Stage.end: Stage.tech
}

tech_cost = {
    TechType.null: 0,
    TechType.hunting: 2,
    TechType.organization: 2,
    TechType.farming: 5,
    TechType.climbing: 5,
    TechType.shields: 6,
    TechType.construction: 7,
    TechType.riding: 5,
    TechType.forestry: 6,
    TechType.archery: 6,
    TechType.mathematics: 10
}

tech_prerequisite = {
    TechType.null: TechType.null,
    TechType.organization: TechType.null,
    TechType.hunting: TechType.null,
    TechType.riding: TechType.null,
    TechType.climbing: TechType.null,
    TechType.shields: TechType.organization,
    TechType.farming: TechType.organization,
    TechType.construction: TechType.farming,
    TechType.archery: TechType.hunting,
    TechType.forestry: TechType.hunting,
    TechType.mathematics: TechType.forestry
}

city_points = [0, 100, 40, 35, 30, 25, 20, 15, 10, 5, 0]


def tech_tier(tech):
    if tech == TechType.null:
        return 0
    return tech_tier(tech_prerequisite[tech]) + 1


resource_cost = {
    ResourceType.fruit: 2,
    ResourceType.animal: 2
}

max_hp = 10
unit_cost = 2
max_turn = 10

resource_tech = {
    ResourceType.fruit: TechType.organization,
    ResourceType.animal: TechType.hunting
}


def power_set(s):
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def total_tech_cost(t_combo):
    return sum(tech_cost[tech] for tech in t_combo)


def within_border(city, a):
    return abs(city.x - a.x) <= city.range and abs(city.y - a.y) <= city.range


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
        self.cities = [[City(1, 1, 1, 2, 0, 1, 0, 0)],
                       [City(3, 3, 1, 1, 1, 1, 0, 0)]]
        self.visible = [{(1 + dx, 1 + dy) for dx, dy in product([-1, 1, 0], repeat=2)},
                        {(3 + dx, 3 + dy) for dx, dy in product([-1, 1, 0], repeat=2)}]
        self.units = [[], []]
        self.points = [0, 0]
        self.stars = [4, 4]
        self.tech = [{TechType.null, TechType.hunting}, {TechType.null, TechType.organization}]
        self.stage = Stage.tech
        self.actions = None

    def inc_pop(self, city, delta):
        city.pop += delta
        self.points[city.player] += 5 * delta
        if city.pop >= city.level:
            city.pop -= city.level
            city.level += 1
            self.points[city.player] += city_points[city.level]
            return True
        return False

    def within_map(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def attack(self, u, v, retaliate=True):
        if u.player == v.player:
            return False

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

    def tech_upgradable(self, t, j):
        return t not in self.tech[j] and tech_prerequisite[t] in self.tech[j] and tech_cost[t] <= self.stars[j]

    def create_game_action(self, action_data):
        return GameAction(self.cur_player, self.stage, action_data)

    def display(self):
        m = [['..'] * self.size for _ in range(self.size)]
        for city in chain(*self.cities):
            m[city.x][city.y] = '{}.'.format(city.player)
        for unit in chain(*self.units):
            m[unit.x][unit.y] = '{}{}'.format(m[unit.x][unit.y][0], unit.player)
        for r in self.resources:
            m[r.x][r.y] = ('f' if r.type == ResourceType.fruit else 'a') + m[r.x][r.y][1]
        for r in self.villages:
            m[r.x][r.y] = 'v' + m[r.x][r.y][1]
        print('======')
        for i in range(self.num_player):
            tech_str = ' '.join(t.name for t in self.tech[i])
            print('Player {}: stars = {}, tech = {}'.format(i, self.stars[i], tech_str))
        print('======')
        for x in range(self.size):
            print('|'.join(m[x]))
        print('======')

    def get_possible_actions(self):
        if self.actions is not None:
            return self.actions

        j = self.cur_player
        stars = self.stars[j]
        actions = []

        if self.stage == Stage.tech:
            new_techs = [t for t in TechType if self.tech_upgradable(t, j)]
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

        if self.stage == Stage.capture:
            may_capture = [v for v in self.villages if any((v.x, v.y) == (u.x, u.y) for u in self.units[j])]
            actions = [self.create_game_action(v_combo) for v_combo in power_set(may_capture)]

        if not actions:
            actions = [self.create_game_action(None)]

        self.actions = actions
        return self.actions

    def take_action_inplace(self, a):
        j, stage, data = a.player, a.stage, a.data

        assert (j == self.cur_player and stage == self.stage)
        self.actions = None

        if data is None and stage != Stage.end:
            self.stage = next_stage[self.stage]
            return self

        if self.stage == Stage.tech:
            total_cost = sum(tech_cost[tech] for tech in data) + len(self.cities[j])
            if total_cost <= self.stars[j]:
                self.stars[j] -= total_cost
                self.tech[j].update(data)
                self.points[j] += 100 * sum(tech_tier(tech) for tech in data if tech not in self.tech[j])

        # TODO: check valid resource data
        if self.stage == Stage.resource:
            total_cost = sum(resource_cost[r.type] for r in data)
            if total_cost <= self.stars[j]:
                self.stars[j] -= total_cost
                self.resources = set(self.resources) - set(data)
                for r in data:
                    for city in self.cities[j]:
                        if any(within_border(city, r) for city in self.cities[j]):
                            self.inc_pop(city, 1)
                            break

        # TODO: check valid data
        if self.stage == Stage.train:
            total_cost = unit_cost * len(data)
            if total_cost <= self.stars[j]:
                self.stars[j] -= total_cost
                for c in data:
                    # City = namedlist('City', 'x y capital range player level pop trained')
                    # Unit = namedlist('Unit', 'x y player level hp attack defence kill')
                    self.units[j].append(Unit(c.x, c.y, j, 1, max_hp, 2, 2, 0))
                    c.trained += 1
                    self.points[j] += 5 * unit_cost

        if self.stage == Stage.move:
            for u, x, y in data:
                self.units[j].remove(u)
                u.x, u.y = x, y
                self.units[j].append(u)
                for dx, dy in product([-1, 0, 1], repeat=2):
                    x2, y2 = x + dx, y + dy
                    if self.within_map(x2, y2):
                        if (x2, y2) not in self.visible[j]:
                            self.visible[j].add((x2, y2))
                            self.points[j] += 5

        occupied = {(u.x, u.y): u for u in chain(*self.units)}

        if self.stage == Stage.attack:
            for u, x, y in data:
                if x != -1 and occupied[x, y].player != occupied[u.x, u.y].player:
                    v = occupied[x, y]
                    self.attack(u, v, True)

            dead = [u for u in chain(*self.units) if u.hp <= 0]
            for u in dead:
                self.units[u.player].remove(u)

        # City = namedlist('City', 'x y capital range player level pop trained')
        if self.stage == Stage.capture:
            for v in data:
                if any((u.x, u.y) == (v.x, v.y) for u in self.units[j]):
                    self.cities[j].append(City(v.x, v.y, 0, 1, j, 1, 0, 0))
                    self.villages.remove(v)
                    self.points[j] += 100

        if self.stage != Stage.end:
            self.stage = next_stage[self.stage]
            return self

        # heal all units before end
        for u in filter(lambda u: u.hp < max_hp, self.units[j]):
            healed_hp = 2 if any(within_border(city, u) for city in self.cities[j]) else 4
            u.hp = min(u.hp + healed_hp, max_hp)

        next_player = (j + 1) % self.num_player
        while len(self.cities[next_player]) == 0 and next_player != j:
            self.turn_num += next_player == 0
            next_player = (next_player + 1) % self.num_player

        self.turn_num += next_player == 0
        self.cur_player = next_player
        self.stage = Stage.tech
        self.stars[j] += sum(get_star(city) for city in self.cities[j])
        return self

    def take_action(self, a):
        state = deepcopy(self)
        return state.take_action_inplace(a)

    def is_terminal(self):
        surviving = sum(len(city_set) > 0 for city_set in self.cities)
        return surviving <= 1 or self.turn_num >= max_turn

    def get_points(self, j):
        return self.points[j]

    def get_reward(self):
        surviving = sum(len(city_set) > 0 for city_set in self.cities)
        if surviving == 1:
            return len(self.cities[self.cur_player]) * 10000
        return self.get_points(self.cur_player)

    def winning(self):
        surviving = [len(city_set) > 0 for city_set in self.cities]
        if sum(surviving) == 1:
            return surviving[self.cur_player]
        return self.points[self.cur_player] >= max(self.points)

    def winner(self):
        surviving = [len(city_set) > 0 for city_set in self.cities]
        if sum(surviving) == 1:
            for j in range(self.num_player):
                if surviving[j]:
                    return j
        return max(range(self.num_player), key=lambda i: self.points[i])


def uniform_dist(N):
    return [1 / N] * N


def gen_myopic_policy(evaluate, link=math.exp):
    def myopic_policy(state):
        actions = state.get_available_actions()
        action_prob = [link(evaluate(state.take_action(a), state.cur_player)) for a in actions]
        total_prob = sum(action_prob)
        action_prob = [p / total_prob for p in action_prob]
        return actions, action_prob

    return myopic_policy


def random_policy(state):
    actions = state.get_possible_actions()
    action_prob = uniform_dist(len(actions))
    return actions, action_prob


def random_value(state):
    return random.random()


def self_value(state):
    return state.get_reward()


def winning_value(state):
    return int(state.wining())


def random_Q(state, action):
    return random.random()


class Appraiser:
    def __init__(self, params):
        self.building_reward = params['building_reward']
        self.unit_reward = params['unit_reward']
        self.pop_reward = params['pop_reward']
        self.star_reward = params['star_reward']
        self.growth_reward = params['growth_reward']
        self.tech_reward = {t: params['tech_{}_reward'.format(t.name)] for t in TechType}
        self.resource_reward = params['resource_reward']
        self.low_hp_penalty = params['low_hp_penalty']
        self.kill_reward = params['kill_reward']
        self.visibility_reward = params['visibility_reward']
        self.offensive_reward = params['offensive_reward']
        self.vulnerability_penalty = params['vulnerability_penalty']

    def evaluate(self, state, j):
        cities = state.cities[j]
        units = state.units[j]
        n_cities = len(cities)
        n_units = len(units)
        pop = sum(c.pop for c in cities)
        stars = state.stars[j]
        growth = sum(c.level + c.capital for c in cities)
        tech = state.tech[j]
        resource = sum(any(within_border(c, r) for c in cities) for r in state.resources)
        low_hp = sum(u.hp < 5 for u in units)
        kill = sum(u.kill for u in units)
        visibility = len(state.visible[j])

        offensive, vulnerability = 0, 0
        for u, v in product(units, chain(*state.units)):
            if v.player != j and abs(u.x - v.x) <= 1 and abs(u.y - v.y) <= 1:
                offensive += u.hp > v.hp + 2
                vulnerability += u.hp < v.hp - 2

        return (n_cities * self.building_reward
                + n_units * self.unit_reward
                + pop * self.pop_reward
                + stars * self.star_reward
                + growth * self.growth_reward
                + sum(self.tech_reward[t] for t in tech)
                + resource * self.resource_reward
                - low_hp * self.low_hp_penalty
                + kill * self.kill_reward
                + visibility * self.visibility_reward
                + offensive * self.offensive_reward
                - vulnerability * self.vulnerability_penalty
                + state.points[j]) / 100

    def value(self, state):
        return self.evaluate(state, state.cur_player)

    def policy(self, state):
        actions = state.get_possible_actions()
        action_prob = [math.exp(self.evaluate(state.take_action(a), state.cur_player)) for a in actions]
        total_prob = sum(action_prob)
        action_prob = [p / total_prob for p in action_prob]
        return actions, action_prob


default_tech_reward = {'tech_{}_reward'.format(t.name): 0 for t in TechType}

appraiser1_params = {
    'building_reward': 100,
    'unit_reward': 40,
    'pop_reward': 10,
    'star_reward': 1,
    'growth_reward': 100,
    'resource_reward': 10,
    'low_hp_penalty': 50,
    'kill_reward': 50,
    'visibility_reward': 10,
    'offensive_reward': 5,
    'vulnerability_penalty': 10
}

tech1 = {
    'tech_hunting_reward': 20,
    'tech_organization_reward': 20
}

appraiser1_params.update(default_tech_reward)
appraiser1_params.update(tech1)

appraiser2_params = {
    'building_reward': 50,
    'unit_reward': 20,
    'pop_reward': 20,
    'star_reward': 1,
    'growth_reward': 200,
    'resource_reward': 10,
    'low_hp_penalty': 50,
    'kill_reward': 50,
    'visibility_reward': 10,
    'offensive_reward': 50,
    'vulnerability_penalty': 10
}

tech2 = {
    'tech_hunting_reward': 20,
    'tech_organization_reward': 20
}

appraiser2_params.update(default_tech_reward)
appraiser2_params.update(tech2)

appraiser1 = Appraiser(appraiser1_params)
appraiser2 = Appraiser(appraiser2_params)


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
                self.values[i] = winning_value

    def get_rollout(self):
        def rollout(state):
            while not state.is_terminal():
                j = state.cur_player
                policy = self.policies[j](state)
                if sum(policy[1]) < 0.1:
                    policy[1] = uniform_dist(len(policy[1]))
                try:
                    a = np.random.choice(policy[0], 1, policy[1])[0]
                except IndexError:
                    raise Exception("Non-terminal state has no possible actions: " + str(state))
                state = state.take_action_inplace(a)
            return state

        return rollout


def run(agents):
    state = GameState()
    state.display()
    print(state.cur_player, state.winning(), state.get_points(0), state.get_points(1), state.turn_num)
    rollout = agents.get_rollout()
    terminal_state = rollout(state)
    print(terminal_state.cur_player, terminal_state.winning(), terminal_state.get_points(0),
          terminal_state.get_points(1), terminal_state.turn_num)

    terminal_state.display()
    print("The winner is {}".format(terminal_state.winner()))

    print(terminal_state.cities)
    print(terminal_state.units)


def multiple_run(agents, n=100):
    winner = []
    rollout = agents.get_rollout()

    for i in range(n):
        if i % 10 == 0:
            print("rolling out game {}".format(i))
        state = GameState()
        terminal_state = rollout(state)
        winner.append(terminal_state.winner())

    print(winner)
    print("Player 0 wins {}/{}".format(sum(w == 0 for w in winner), len(winner)))


agents1 = Agents([appraiser1.value, appraiser2.value], [appraiser1.policy, appraiser2.policy])
multiple_run(agents1, 100)
