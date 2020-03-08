from copy import deepcopy
from mcts import mcts
from functools import reduce
import operator
from collections import namedlist
from enum import Enum
from itertools import combinations, chain, product

City = namedlist('City', 'x y capital range player level pop trained')
Village = namedlist('Village', 'x y')
Unit = namedlist('Unit', 'x y player level hp')
Resource = namedlist('Resource', 'type x y')
ResourceType = Enum('ResourceType', 'fruit animal')
TechType = Enum('TechType', 'huting organization farm')

ActionStage = Enum('ActionState', 'tech resource train move attack end')

tech_cost = {TechType.huting: 2, TechType.organization: 2, TechType.farm: 5}


resource_cost = {ResourceType.fruit: 2, ResourceType.animal: 2}

max_hp = 10

unit_cost = 2

resource_tech = {ResourceType.fruit: TechType.organization, ResourceType.animal: TechType.huting}

def powerset(s):
	return chain.from_iterable(combinations(s,r) for r in range(len(s)+1))

def total_tech_cost(techcombo):
	return sum(tech_cost[tech] for tech in techcombo)

def total_upgrade_cost(citycombo):
	return sum(city_upgrade_cost[city.level] for city in citycombo)

def within_border(city, a):
	return abs(city.x - a.x) <= city.range and abs(city.y - a.y) < city.range

class GameState():
	def __init__(self):
		self.size = 5
		self.num_player = 2
		self.cur_player = 0
		self.villages =[Village(0,4), Village(2,2)]
		self.resources = [Resource(ResourceType.fruit, 0, 1), Resource(ResourceType.animal, 3,4)]
		self.cities = [[City(1,1,1,1,1,0,0)], [City(3,3,1,1,1,0,0)]]
		self.units = [[],[]]
		self.stars = [100,100]
		self.tech = [set([TechType.huting]), set[(TechType.organization)]]
		self.stage = ActionState.tech

	def getPossibleActions(self):
		j = self.cur_player
		stars = self.stars[j]

		if self.stage == ActionState.tech:
			new_techs = [t for t in TechType if t not in self.tech[j] and tech_cost[t] <= stars]
			return [techcombo for techcombo in powerset(new_techs) if total_tech_cost(techcombo) <= stars]

		if self.stage == ActionState.resource:
			avail_resources = []
			for r in self.resources:
				if resource_tech[r.type] in self.tech[j]:
					for city in self.cities[j]:
						if within_border(city, r):
							avail_resources.append(r)

			resource_combos = []
			for rcombo in powerset(avail_resources):
				if sum(resource_cost[r.type] for r in rcombo) <= stars:
					resource_combos.append(rcombo)

			return resource_combos

		occupied = {}

		for u in chian(*self.units):
			occupied[u.x, u.y] = u.player

		if self.stage == ActionState.train:
			trainable_cities = [c for c in self.cities[j] if c.trained < c.level and (c.x, c.y) not in occupied]
			return [ccombo for ccombo in pwerset(trainable_cities) if len(ccombo)*unit_cost <= stars]

		if self.stage == ActionStage.move:

			movable = []

			for u in self.units[j]:
				movable.append([(u, u.x, u.y)])
				for dx in range(-1,2):
					for dy in range(-1,2):
						vx, vy = u.x + dx, u.y + vy
						if 0 <= vx < self.size and 0 <= vy < self.size and (vx, vy) not in occupied:
							movable[-1].append((u,vx,vy))

			return product(movable)

		if self.stage == ActionStage.attack:
			occupied = {}

			for k, units in enumerate(self.units):
				for u in units:
					occupied[u.x,u.y] = u.player

			attackable = [[]]
			for u in self.units[j]:
				if not attackable[-1]:
					attackable.append([])
				for dx in range(-1,2):
					for dy in range(-1,2):
						vx, vy = u.x+dx, u.y+dy
						if 0 <= vx < self.size and 0 <= vy < self.size and (vx, vy) in occupied and occupied[vx,vy] != j:
							attackable[-1].append((u,vx,vy))

			if not attackable[-1]:
				attackable.pop()

			return product(attackable)

		return [1]



	def inc_pop(city, delta):
		city.pop += delta
		if city.pop >= city.level:
			city.pop -= city.level
			city.level += 1
			return True
		return False


	def takeAction(self, action):
		state = deepcopy(self)
		j = state.cur_player

		if state.stage == ActionStage.tech:
			total_cost = sum(tech_cost[tech] for tech in action) + len(state.citis[j])
			if total_cost <= state.stars[j]:
				state.stars[j] -= total_cost
				state.tech[j].update(action)
				state.stage = ActionStage.resource
				return state

		# TODO: check valid resource action
		if state.stage == ActionStage.resource:
			total_cost = sum(resource_cost[r] for r in action)
			if total_cost <= state.stars[j]:
				state.stars[j] -= total_cost
				state.resources = set(state.resources) - set(action)
				for r in action:
					for city in states.cities[j]:
						if within_border(r, city):
							inc_pop(city)
							break
				state.stage = ActionStage.train
				return state

		if state.stage == ActionStage.train:
			total_cost = unit_cost*len(action)



		if state.stage == ActionStage.end:
			for u in filter(lambda u: u.hp < max_hp, state.units[j]):
				healed_hp = 4
				for city in state.cities[j]:
					if within_border(city, u)
						healed_hp = 2
						break
				u.hp = min(u.hp + healed_hp, max_hp)

			state.cur_player = j = (j+1) % state.num_player
			state.stage = ActionStage.tech
			state.stars[j] += sum(get_star(city) for city in state.cities[j])
			return state


















