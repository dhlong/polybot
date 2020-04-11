from namedlist import namedlist
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import combinations, chain, product
import numpy as np
import json
import os

#!NOTE1: python Enum type auto starts from 1 
#        whereas C# enum starts from 0
#!NOTE2: the code relies on iterating over enum in definition order
#        this only works with python 3, not python 2

CONFIG_PATH = './config'
EARLY_TERMINATION = 15


#%% ENUM 
###############################################################################

with open(CONFIG_PATH + '/polyenum.json', 'r') as f:
    polyenum = json.load(f)

polyenum_str = {key: " ".join(values).replace("None", "none") for key, values in polyenum.items()}

TerrainType  = Enum('Terrain',    polyenum_str["Terrain"])
ResourceType = Enum('Resource',   polyenum_str["Resource"])
BuildingType = Enum('Building',   polyenum_str["Building"])
Reward       = Enum('Reward',     polyenum_str['Reward'])
ActionType   = Enum('ActionType', polyenum_str["ActionType"])
AbilityType  = Enum('Ability',    polyenum_str["Ability"])
Technology   = Enum('Technology', polyenum_str['Technology'])

Unit = namedlist('Unit', 'x y id player city type hp move_turn attack_turn kill veteran')
City = namedlist('City', 'x y id player capital level pop border star wall')

data = {}
for datafile in ['techTree', 'abilities', 'cities', 'units', 'unitSkills']:
    with open('{}/{}.txt'.format(CONFIG_PATH, datafile), 'r') as f:
        data[datafile] = json.load(f)



#%% TECH 
###############################################################################

tech_data = data['techTree']['technologies']
tech_names = [tech.name for tech in Technology]
tech_idx = {tech.name: tech.value-1 for tech in Technology}
tech_cost = [tech_data[tech.name]['cost'] if tech.name in tech_data else 0
             for tech in Technology]
tech_dependence = [-1]*len(tech_cost)

for parent, info in tech_data.items():
    try:
        for child in info['childs']:
            tech_dependence[tech_idx[child]] = tech_idx[parent]
    except KeyError:
        pass

tech_level = [0]*len(tech_names)

def find_tech_level(tech):
    if tech == -1 or tech_level[tech] != -1:
        return 1 if tech == -1 else tech_level[tech]
    tech_level[tech] = find_tech_level(tech_dependence[tech]) + 1
    return tech_level[tech]

for tech in range(len(tech_names)):
    find_tech_level(tech)


start_techs = [tech_idx[tech_name] 
               for tech_name in data['techTree']['startTechs']]

            

#%% UNIT 
###############################################################################

unit_data = data['units']
unit_names = list(unit_data.keys())
unit_idx = {name: i for i, name in enumerate(unit_names)}
unit_cost =[u['cost'] for u in unit_data.values()]

UnitStats = namedtuple('UnitStats', 'attack defence health movement range dash escape')
unit_stats = [UnitStats(unit_data[u]['attack'], 
                        unit_data[u]['defence'],
                        unit_data[u]['health'],
                        unit_data[u]['movement'] if 'movement' in u else 1,
                        unit_data[u]['range'], 
                        'Dash' in unit_data[u]['skills'], 
                        'Escape' in unit_data[u]['skills'])
             for u in unit_names]



#%% ABILITY 
###############################################################################

ability_data = data['abilities']
ability_names = [a.name for a in AbilityType]
ability_lookup = {a.name: a for a  in AbilityType}
ability_cost = {a: ability_data[a.name]['cost'] 
                if a.name in ability_data and 'cost'  in ability_data[a.name] 
                else 0 
                for a in AbilityType}
ability_pop = {a: ability_data[a.name]['population'] 
               if a.name in ability_data and 'population' in ability_data[a.name] 
               else 0 
               for a in AbilityType}

ability_dependence = {a: -1 for a in AbilityType}
for tech, info in tech_data.items():
    if 'abilities' in info:
        for name in info['abilities']:
            ability_dependence[ability_lookup[name]] = tech_idx[tech]

unit_dependence = [-1]*len(unit_names)

# City abilities, should always has 'unit' field
for a in AbilityType:
    if a.name in ability_data and 'unit' in ability_data[a.name]:
        for unit_name, u_id in unit_idx.items():
            if unit_name.endswith(ability_data[a.name]['unit']):
                unit_dependence[u_id] = ability_dependence[a]
        
city_max_pop = [0] + data['cities']['population']

building_after_ability = {
    AbilityType.Hunting: BuildingType.none,
    AbilityType.HarvestFruit: BuildingType.none,
    AbilityType.Destroy: BuildingType.none,
    AbilityType.Farm: BuildingType.Farm,
    AbilityType.Windmill: BuildingType.Windmill
}

tile_abilities = [a for a in AbilityType 
                  if a.name in ability_data 
                  and 'type' in ability_data[a.name] 
                  and ability_data[a.name]['type'] == 'Tile']

if AbilityType.Capture not in tile_abilities:
    tile_abilities.append(AbilityType.Capture)

tile_ability_idx = {a: i for i, a  in enumerate(tile_abilities)}
n_tile_abilities = len(tile_abilities)

# to be used in encoding and decoding actions
directions = [d for d in product(range(-2,3), repeat=2) if d != (0,0)]
dir_idx = {d: i for i, d in enumerate(directions)}



#%% HELPER 
###############################################################################

def max_health(unit):
    return unit_stats[unit.type].health + 5*unit.veteran


def apply_attack(attacker, defender, def_bonus):
    atk_stats = unit_stats[attacker.type]
    def_stats = unit_stats[defender.type]
    
    distance = max(abs(attacker.x - defender.x), abs(attacker.y - defender.y))
    if distance > atk_stats.range:
        return False
        
    atk_force = atk_stats.attack*attacker.hp/max_health(attacker)
    def_force = def_stats.defence*defender.hp/max_health(defender)*def_bonus
    total_dmg = atk_force + def_force
    
    accelerator = 4.5
    
    atk_res = atk_force/total_dmg*atk_stats.attack*accelerator
    def_res = def_force/total_dmg*def_stats.defence*accelerator
    
    defender.hp -= atk_res
    if defender.hp <= 0:
        attacker.kill += 1
        return True
    
    # TODO: check if range matters when retalliating
    if distance > def_stats.range:
        return True
    
    attacker.hp -= def_res
    defender.kill += attacker.hp <= 0
    return True


def inc_pop(city, delta=1):
    city.pop += delta
    if city.pop >= city_max_pop[city.level]:
        city.pop -= city.level
        city.level += 1
        city.star += 1
        return True
    return False



#%% Action space and coding 
###############################################################################

action_posibilities = {
    ActionType.EndTurn: 1,
    ActionType.ResearchTechnology: len(tech_cost),
    ActionType.UpgradeCity: len(Reward),
    ActionType.UnitMove: len(directions),
    ActionType.UnitAttack: len(directions),
    ActionType.UnitUpgrade: 1,
    ActionType.UnitAbility: 0,
    ActionType.TrainUnit: len(unit_names),
    ActionType.TileApplyAbility: n_tile_abilities
}

unit_action_types = [ActionType.UnitMove, ActionType.UnitAttack, ActionType.UnitUpgrade]
city_action_types = [ActionType.UpgradeCity, ActionType.TrainUnit]
tile_action_types = set(unit_action_types + city_action_types + [ActionType.TileApplyAbility])


# The code relies on Enum type maintaining its order
# perhaps can change to other container for faster and safer lookup
assert set(action_posibilities.keys()) == set(ActionType)

def get_n_actions(a, board_size):
    if a in tile_action_types:
        return board_size*board_size*action_posibilities[a]
    return action_posibilities[a]


def get_total_n_actions(board_size):
    return sum(get_n_actions(a, board_size) for a in ActionType)


def get_decoder(board_size = 11):
    def decode_action(action_code):
        for a in ActionType:
            n = get_n_actions(a, board_size)
            if action_code < n:
                if a in tile_action_types:
                    j = action_code % board_size
                    action_code //= board_size
                    i = action_code % board_size
                    param = action_code // board_size
                    return a, i, j, param
                return a, 0, 0, action_code
            action_code -= n
        return ActionType.EndTurn, 0, 0, 0
    return decode_action


def get_encoder(board_size = 11):
    def encode_action(action_type, i, j, param):
        action_code = 0
        for a in ActionType:
            if a == action_type:
                if a in tile_action_types:
                    return action_code + j + (i + param*board_size)*board_size
                return action_code + param
            action_code += get_n_actions(a, board_size)
        return 0
    return encode_action




#%% GAME STATE 
###############################################################################


class GameState:

    def __init__(self, json_file, history = None):
        self.size = -1
        self.n_players = -1
        self.cities = None
        self.units = None
        self.terrain = None
        self.terrain = None
        self.resource = None
        self.explored = None
        self.building = None
        self.territory = None
        self.turn = None
        self.stars = None
        self.has_tech = None
        self.player = -1
        self.child_visits = None
        self.num_actions = -1
        self.roll_out_flag = False
        self.history = history[:] if history is not None else []
        self.json_file = json_file

    def __read_json(self, json_file):
        self.json_file = json_file
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.size = n = data['worldMap']['size']
        self.n_players = len(data['players'])
        self.cities = {}
        for c in data['cities']:
            assert c['id'] == len(self.cities) + 1, 'cities are not listed in order'
            city = City(c['i'], c['j'], c['id'],
                        c['playerId'] - 1,
                        c['isCapital'],
                        c['level'],
                        c['population'],
                        c['border'] if 'border' in c else 1,
                        c['star'],
                        c['wall'])
            self.cities[city.id] = city

        self.units = {}
        for u in data['units']:
            unit = Unit(u['i'], u['j'], u['id'],
                        u['playerId'] - 1,
                        u['cityId'],
                        unit_idx[u['name']],
                        u['health'],
                        u['moveTurn'],
                        u['attackTurn'],
                        u['kill'],
                        u['isVeteran'])
            if not u['died']:
                self.units[unit.id] = unit

        self.terrain = [0] * (n**2)
        self.resource = [0] * (n**2)
        self.explored = [[False]*(n**2) for _ in range(self.n_players)]
        self.building = [0] * (n**2)
        self.territory = [-1]* (n**2)
        self.history = []

        for row in data['worldMap']['tiles']:
            for tile in row:
                idx = tile['i']*n + tile['j']

                self.terrain[idx] = tile['terrain']
                self.resource[idx] = tile['resource']
                self.building[idx] = tile['building']

                # python cities and players are indexed with 0-based indexing
                # C# cities and players are 1-based indexed
                self.territory[idx] = tile['territory']
                for player in tile['exploredPlayers']:
                    self.explored[player-1][idx] = True

        self.turn = [-1] * self.n_players
        self.stars = [0] * self.n_players
        self.has_tech = [[False] * len(Technology) for _ in range(self.n_players)]
        for i, player in enumerate(data['players']):
            self.stars[i] = player['star']
            self.turn[i] = player['turn']
            for tech in player['technologies']:
                self.has_tech[i][Technology[tech].value-1] = True

        self.player = min(range(self.n_players), key = self.turn.__getitem__)

        self.child_visits = []
        self.num_actions = get_total_n_actions(self.size)
        self.roll_out_flag = True
           
    def legal_actions(self):
        self.__roll_out()
        actions = []
        player = self.player
        stars = self.stars[player]
        n = self.size

        encoder = get_encoder(self.size)

        # research
        for tech in range(len(tech_cost)):
            if self.check_research_action(player, tech):
                actions.append(encoder(
                    ActionType.ResearchTechnology, 
                    0, 0,
                    tech
                ))

        # train unit
        for city in self.get_player_cities(player):
            if self.check_city_can_train(city):
                for u_type in range(len(unit_cost)):
                    if self.check_unit_train_action(player, city, u_type):
                        actions.append(encoder(
                            ActionType.TrainUnit,
                            city.x, city.y,
                            u_type
                        ))

        own_territory = []
        for idx, territory in enumerate(self.territory):
            if territory > 0 and self.cities[territory].player == player:
                own_territory.append(idx)
        
        # tile action
        for a in tile_abilities:
            if a != AbilityType.Capture and self.check_tile_ability(player, a):
                for i, j in map(self.tile_coord, own_territory):
                    if self.check_tile_apply_action(player, a, i, j):
                        actions.append(encoder(
                            ActionType.TileApplyAbility, 
                            i, j, 
                            tile_ability_idx[a] 
                        ))
        
        # unit move and attack
        # TODO: movement cost, block, bfs
        for unit in self.get_player_units(player):
            if unit.move_turn:
                for i2, j2 in self.neighbors(unit.x, unit.y, unit_stats[unit.type].movement):
                    if self.check_unit_move_action(player, unit, i2, j2):
                        actions.append(encoder(
                            ActionType.UnitMove, 
                            unit.x, unit.y, 
                            dir_idx[i2 - unit.x, j2 - unit.y]
                        ))
                    
            if unit.attack_turn:
                for v in self.get_alive_units():
                    if self.check_unit_attack_action(player, unit, v):
                        actions.append(encoder(
                            ActionType.UnitAttack, 
                            unit.x, unit.y,
                            dir_idx[v.x - unit.x, v.y - unit.y]
                        ))
                        
            if unit.kill >= 3 and not unit.veteran:
                actions.append(encoder(
                    ActionType.UnitUpgrade, 
                    unit.x, unit.y, 
                    0
                ))

            # Capture city
            if unit.move_turn:
                city = self.get_city_at(unit.x, unit.y)
                if city is not None and city.player != unit.player:
                    actions.append(encoder(
                        ActionType.TileApplyAbility,
                        unit.x, unit.y,
                        tile_ability_idx[AbilityType.Capture]
                    ))

        if len(actions) < 4 or (len(self.history) > 1 and self.history[-1][0]):
            actions.append(encoder(
                ActionType.EndTurn,
                0, 0,
                0
            ))

        return actions
    
    def apply(self, action):
        self.__roll_out()
        player = self.player

        self.history.append((player,action))

        action_type, i, j, param = get_decoder(self.size)(action)
        n = self.size

        if action_type == ActionType.EndTurn:
            self.end_turn()
            self.player = (self.player + 1) % self.n_players
            self.begin_turn()
            return

        if action_type == ActionType.ResearchTechnology:
            tech = param
            if self.check_research_action(player, tech):
                self.has_tech[player][tech] = True
                self.spend_stars(player, tech_cost[tech])
                return
                
        if action_type == ActionType.TrainUnit:
            city = self.get_city_at(i, j)
            u_type = param
            if (city is not None
                and city.player == player
                and self.check_city_can_train(city) 
                and self.check_unit_train_action(player, city, u_type)):
                        unit = self.spawn_unit(city, u_type)
                        self.units[unit.id] = unit
                        self.spend_stars(player, unit_cost[u_type])
                        return
            
        if action_type == ActionType.TileApplyAbility:
            a = tile_abilities[param]
            idx = self.tile_idx(i, j)
            if self.check_tile_ability(player, a) and self.check_tile_apply_action(player, a, i, j):
                if a == AbilityType.HarvestFruit or a == AbilityType.Hunting:
                    self.resource[idx] = ResourceType.none.value-1

                if a == AbilityType.Farm:
                    self.building[idx] = BuildingType.Farm.value-1
                
                if ability_pop[a] > 0:
                    inc_pop(self.cities[self.territory[idx]], ability_pop[a])
                
                if a == AbilityType.Capture:
                    city = self.get_city_at(i, j)
                    assert city is not None, "detect no city when tile_apply, one of the check does not work"
                    
                    if city.level == 0:
                        city.level += 1
                        self.expand_city_border(city, 1)
                    city.player = player
                    
                self.spend_stars(player, ability_cost[a])       
                return    
                  
        if action_type == ActionType.UnitMove:
            unit = self.get_unit_at(i,j)
            di, dj = directions[param]
            i2, j2 = i+di, j+dj
            if unit is not None and self.check_unit_move_action(player, unit, i2, j2):
                unit.x = i2
                unit.y = j2
                unit.move_turn = 0
                self.explore_tile(player, unit.x, unit.y)
                return
                
        if action_type == ActionType.UnitAttack:
            di, dj = directions[param]
            unit = self.get_unit_at(i, j)
            target = self.get_unit_at(i+di, j+dj)
            
            # note: rely on short-circuit logic evaluation
            if (unit is not None 
                and target is not None 
                and self.check_unit_attack_action(player, unit, target)
                and apply_attack(unit, target, self.get_def_bonus(target))):
                    unit.move_turn = 0
                    unit.attack_turn = 0
                    
                    # melee kill => move to target
                    if target.hp <= 0 and unit_stats[unit.type].range == 1: 
                        unit.x = target.x
                        unit.y = target.y
                    return
                        
        if action_type == ActionType.UnitUpgrade:
            unit = self.get_unit_at(i,j)
            if (unit is not None and unit.player == player 
                and unit.kill >= 3 and not unit.veteran):
                    unit.veteran = True
                    unit.hp = max_health(unit)
                    return
                
        if action_type == ActionType.UpgradeCity:
            city = self.get_city_at(i,j)
            if city is not None and city.player == player:
                if param == Reward.Star.value-1:
                    self.stars[player] += 10
                elif param == Reward.CityStar.value-1:
                    city.star += 2    
                return

    def terminal(self):
        self.__roll_out()
        surviving = set(city.player for city in self.cities.values() if city.player >= 0)
        if len(surviving) == 1:
            return True
        if min(self.turn) >= EARLY_TERMINATION:
            return True
        return False

    def terminal_value(self, player):
        self.__roll_out()
        
        surviving = list(set(city.player for city in self.cities.values() 
                             if city.player >= 0))
        if len(surviving) == 1:
            return 1 if player == surviving[0] else -1
        points = [self.get_points(player) for player in range(self.n_players)]
        winning_points = max(points)

        return 1 if points[player] == winning_points else -1

    def state_matrix(self):
        self.__roll_out()
        n, k = self.size, self.n_players
        
        city_channels = 4
        city_matrix = np.zeros((city_channels, n, n))
        village_matrix = np.zeros((1, n, n))
        for city in self.cities.values():
            if city.player >= 0:
                city_matrix[:, city.x, city.y] = [city.level, city.star, city.pop, city.capital]
            else:
                village_matrix[0, city.x, city.y] = 1
                    
        # Unit = namedlist('Unit', 'x y player city type hp move_turn attack_turn kill veteran')
        # UnitStats = namedtuple('UnitStats', 'attack defence health movement range dash escape')
        unit_channels = 9
        unit_matrix = np.zeros((unit_channels, n, n))
        for unit in self.get_alive_units():
            stats = unit_stats[unit.type]
            unit_matrix[:, unit.x, unit.y] = [unit.hp, unit.move_turn,
                                              unit.attack_turn, unit.kill, unit.veteran, 
                                              stats.attack, stats.defence, stats.movement, 
                                              stats.range]
            
        territory_matrix = np.zeros((k, n*n))
        for idx, c_id in enumerate(self.territory):
            if c_id > 0 and self.cities[c_id].player >= 0:
                territory_matrix[self.cities[c_id].player, idx] = 1
        territory_matrix = territory_matrix.reshape((-1, n, n))

        unit_ownership_matrix = np.zeros((k, n, n))
        for unit in self.get_alive_units():
            unit_ownership_matrix[unit.player, unit.x, unit.y] = 1
        
        player = self.player
        ownership_matrix = np.concatenate([territory_matrix, unit_ownership_matrix])
        ownership_matrix = np.roll(ownership_matrix, -player, axis=0)

        terrain_matrix  = np.array(self.terrain).reshape((n, n))  == np.arange(1,3).reshape(-1, 1, 1) # shape = 3,n,n
        resource_matrix = np.array(self.resource).reshape((n, n)) == np.arange(1,5).reshape(-1, 1, 1) # shape = 5,n,n
        explored_matrix = np.array(self.explored[player]).reshape((-1, n, n))
        
        map_matrix = np.concatenate([village_matrix,    # 1
                                     city_matrix,       # 4
                                     unit_matrix,       # 9
                                     ownership_matrix,  # 4
                                     terrain_matrix,    # 2
                                     resource_matrix,   # 4
                                     explored_matrix])*explored_matrix  # 1
                    
        player_vector = np.array(self.has_tech[player] + [self.stars[player]])
        player_matrix = np.tile(player_vector.reshape((-1, 1, 1)), (1, n, n))
        
        return np.concatenate([map_matrix, player_matrix]).astype(np.float32)
    


    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index: int):
        # Game specific feature planes.
        if state_index == -1 | state_index == len(self.history):
            return self.state_matrix()
        scratch_game = GameState(self.json_file, self.history[:state_index])
        return scratch_game.state_matrix()

    def make_target(self, state_index: int):
        return (self.terminal_value(self.history[state_index][0]),
                self.child_visits[state_index])

    def to_play(self):
        return self.history[-1][0] if self.history else 0

    def clone(self):
        game = GameState(self.json_file, self.history)
        return game

    def __roll_out(self):
        if not self.roll_out_flag:
            history = self.history
            self.__read_json(self.json_file)
            for player, action in history:
                assert self.player == player, "Wrong player turn encountered when rolling out the history"
                self.apply(action)
            self.roll_out_flag = True

    def unroll(self):
        self.cities = None
        self.units = None
        self.terrain = None
        self.resource = None
        self.explored = None
        self.building = None
        self.territory = None
        self.turn = None
        self.stars = None
        self.has_tech = None
        self.roll_out_flag = False


    # Helper functions

    def check_tech(self, player, tech):
        return tech == -1 or self.has_tech[player][tech]
    
    def check_tile(self, i, j):
        return 0 <= i < self.size and 0 <= j < self.size
    
    def neighbors(self, i, j, d=1):
        for di, dj in product(range(-d,d+1), repeat=2):
            if self.check_tile(i+di, j+dj):
                yield i+di, j+dj
                     
    def get_alive_units(self):
        for u in self.units.values():
            if u.hp > 0:
                yield u

    def get_player_units(self, player):
        for unit in self.units.values():
            if unit.player == player and unit.hp > 0:
                yield unit

    # TODO: optimize: lookup table
    def get_unit_at(self, i, j):
        for unit in self.units.values():
            if unit.hp > 0 and unit.x == i and unit.y == j:
                return unit
        return None

    def get_city_at(self, i, j):
        for city in self.cities.values():
            if city.x == i and city.y == j:
                return city
        return None
                
    def get_player_cities(self, player):
        for city in self.cities.values():
            if city.player == player:
                yield city
    
    def tile_idx(self, i, j):
        return i*self.size + j
    
    def tile_coord(self, idx):
        return idx//self.size, idx%self.size
    
    def check_stars(self, player, cost):
        return cost <= self.stars[player]

    def check_tile_apply_action(self, player, a_id, i, j):
        assert self.check_tile(i, j), "Accessing tile out of map"
        assert player < self.n_players, "Player index out of bound"
                
        idx = self.tile_idx(i, j)
        
        if a_id != AbilityType.Capture:
            if self.territory[idx] == 0: return False
            if self.cities[self.territory[idx]].player != player: return False
        
        resource_abilities = {
            AbilityType.HarvestFruit: ResourceType.Fruit, 
            AbilityType.Hunting: ResourceType.Animal,
            AbilityType.Farm: ResourceType.Crop
        }
        
        if a_id in resource_abilities:
            return (self.resource[idx] == resource_abilities[a_id].value-1
                    and self.building[idx] == BuildingType.none.value-1)
        
        if a_id == AbilityType.Windmill:
            return (
                self.terrain[idx] == TerrainType.Field.value-1
                and self.building[idx] == BuildingType.none.value-1
                and any(self.building[self.tile_idx(i2, j2)] 
                        == BuildingType.Farm.value-1
                        for i2, j2 in self.neighbors(i,j))
            )
       
        if a_id == AbilityType.Destroy:
            return self.building[idx] != BuildingType.none.value-1
        
        if a_id == AbilityType.Capture:
            unit = self.get_unit_at(i, j)
            if unit is None or unit.player != player or not unit.move_turn:
                return False
            city = self.get_city_at(i, j)
            return city is not None and city.player != player
        
        return True
    
    def check_research_action(self, player, tech):
        return (self.check_tech(player, tech_dependence[tech])
                and self.check_stars(player, tech_cost[tech]) 
                and not self.check_tech(player, tech))
            
    def check_city_can_train(self, city):
        return sum(self.cities[unit.city] == city for unit in self.get_alive_units()) < city_max_pop[city.level]
        
    def check_unit_train_action(self, player, city, u_type):
        return (self.check_stars(player, unit_cost[u_type])
                and self.check_tech(player, unit_dependence[u_type])
                and not any(unit.x == city.x and unit.y == city.y 
                            for unit in self.get_alive_units()))

    def check_tile_ability(self, player, a_id):
        return (a_id in tile_abilities
                and self.check_tech(player, ability_dependence[a_id])
                and self.check_stars(player, ability_cost[a_id]))
    
    def check_unit_move_action(self, player, unit, i, j):
        return (unit.move_turn 
                and unit.hp > 0
                and unit.player == player 
                and (unit.x, unit.y) != (i, j)
                and self.check_tile(i, j)
                and not any(v.x == i and v.y == j 
                            for v in self.get_alive_units())
                and abs(unit.x-i) <= unit_stats[unit.type].movement
                and abs(unit.y-j) <= unit_stats[unit.type].movement)
    
    def check_unit_attack_action(self, player, unit, target):
        return (unit.attack_turn 
                and unit.hp > 0
                and target.hp > 0
                and unit.player == player
                and unit.player != target.player
                and self.explored[player][self.tile_idx(target.x, target.y)]
                and abs(unit.x-target.x) <= unit_stats[unit.type].range
                and abs(unit.y-target.y) <= unit_stats[unit.type].range)
 
    def expand_city_border(self, city, border):
        city.border = border
        for i, j in self.neighbors(city.x, city.y, border):
            idx = self.tile_idx(i, j)
            if self.territory[idx] == 0:
                self.territory[idx] = city.id
                
    def explore_tile(self, player, i, j, d=1):
        for i2, j2 in self.neighbors(i, j, d):
            self.explored[player][self.tile_idx(i2,j2)] = True
                
    def spend_stars(self, player, cost):
        self.stars[player] -= cost
        assert self.stars[player] >= 0, "negative stars encountered"
        

    # Unit = namedlist('Unit', 'x y player city type hp move_turn attack_turn kill veteran')
    def spawn_unit(self, city, u_type):
        return Unit(city.x, city.y, max(self.units.keys(), default=0)+1, 
                    city.player, city.id, u_type, 
                    unit_stats[u_type].health, 
                    0, 0, 0, 0)
    
    def end_turn(self):
        # auto healed if unit has not moved or attacjed
        for unit in self.get_player_units(self.player):
            if unit.hp > 0 and unit.move_turn and unit.attack_turn:
                healed_hp = 2
                territory = self.territory[self.tile_idx(unit.x, unit.y)]
                if territory > 0:
                    if self.cities[territory].player == self.player:
                        healed_hp = 4
                unit.hp += healed_hp
                unit.hp = min(unit.hp, max_health(unit))

    def begin_turn(self):
        self.turn[self.player] += 1
        player = self.player
        self.stars[player] += sum(city.star for city in self.get_player_cities(player))
        for unit in self.get_player_units(player):
            unit.move_turn = unit.attack_turn = 1


    def get_def_bonus(self, unit):
        def_bonus = 1
        for city in self.get_player_cities(unit.player):
            if city.x == unit.x and city.y == unit.y:
                def_bonus = 2 if city.wall else 1.5
        return def_bonus

            

    # Points, used in case of early termination
    # follow polytopia point system
    # https://polytopia.fandom.com/wiki/Score
    #    +5 per population
    #    +5 per unit cost
    #    +100*tier for research
    #    +20 per territory tiles
    #    +city points by level: 100, 140, 175, 205, 230 250, 265, 275, 280
    def get_points(self, player):
        points = 0
        for city in self.get_player_cities(player):
            population_points = 5*city.pop
            level_points = 55 + 50*city.level -5*city.level*(city.level+1)//2
            points += population_points + level_points
        for unit in self.get_player_units(player):
            points += 5*unit_cost[unit.type]
        for tech, has_tech in enumerate(self.has_tech[player]):
            if has_tech:
                points += 100*tech_level[tech]
        for territory in self.territory:
            if territory > 0 and self.cities[territory].player == player:
                points += 20
        return points
                
    def display(self):
        # https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
        self.__roll_out()
        os.system('color')
        rch = {
            ResourceType.none: ' ',
            ResourceType.Fruit: 'f',
            ResourceType.Animal: 'a', 
            ResourceType.Metal: 'm',
            ResourceType.Crop: 'c'
        }
        tformat = {
            TerrainType.Field: 42,
            TerrainType.Forest: 45,
            TerrainType.Mountain: 40
        }
        resource = [rch[ResourceType(r+1)] for r in self.resource]
        terrain = [tformat[TerrainType(t+1)] for t in self.terrain]
        player = [33]*len(terrain)
        resourcefmt = [33]*len(terrain)
        player_symbol = '@$'

        for city in self.cities.values():
            idx = self.tile_idx(city.x, city.y)
            resource[idx] = 'v' if city.player < 0 else player_symbol[city.player]
            terrain[idx] = 44 if city.player < 0 else 46
            player[idx] = player[idx] if city.player < 0 else 31 if city.player == 0 else 37
            resourcefmt[idx] = player[idx]

        units = [' ' for i in range(len(terrain))]
        for unit in self.get_alive_units():
            idx = self.tile_idx(unit.x, unit.y)
            units[idx] = '*' if unit.player == 0 else '+'
            player[idx] = player[idx] if unit.player < 0 else 31 if unit.player == 0 else 37

        lines = []
        lines.append('cities:')
        for city in self.cities.values():
            lines.append('  {}{:x}{:x}=lvl{},pop{} '.format(
                player_symbol[city.player],
                city.x, city.y, city.level, city.pop))
        lines.append('units:')
        for unit in self.get_alive_units():
            lines.append('  {}{:x}{:x}=hp{} '.format(
                player_symbol[unit.player],
                unit.x, unit.y, unit.hp))

        lines.append("current player={}".format('@' if self.player==0 else '$'))

        curline = 0
        
        print(' ' + ''.join('{:2x}'.format(i) for i in range(self.size)))
        for i in range(self.size):
            s = '{:x} '.format(i)
            for j in range(self.size):
                idx = self.tile_idx(i,j)
                s += '\x1b[1;{};{}m{}\x1b[1;{};{}m{}'.format(
                    resourcefmt[idx], terrain[idx], resource[idx],
                    player[idx], terrain[idx],units[idx])
            s += '\x1b[0m ' + '{:x}'.format(i)
            if curline < len(lines):
                s+= '   ' + lines[curline]
                curline += 1
            print(s)
        print('  ' + ''.join('{:2x}'.format(i) for i in range(self.size)))

        for line in lines[curline:]:
            print('  '*(self.size+2) + '   ' + line)

        print('-'*50)


#%% UNIT TESTS 
###############################################################################

def unit_tests():

    stub_json = {
        'worldMap': {
            'size': 11,
            'tiles':
                [
                    [
                        {
                            'terrain': 0,
                            'resource': 0,
                            'building': 0,
                            'exploredPlayers': [],
                            'territory': 0,
                            'city': 0,
                            'unit': 0,
                            'i': i,
                            'j': j
                        }
                        for j in range(11)
                    ]
                    for i in range(11)
                ]
        },
        'players':
            [
                {
                    'id': 1,
                    'tribe': 0,
                    'star': 100,
                    'turn': 0,
                    'victory': 0,
                    'abilities': [0, 15, 1],
                    'technologies': ['Organization']
                },
                {
                    'id': 2,
                    'tribe': 0,
                    'star': 100,
                    'turn': 0,
                    'victory': 0,
                    'abilities': [0, 15, 1],
                    'technologies': ['Organization']
                }
            ],
        'cities':
            [
                {
                    'id': 1,
                    'playerId': 1,
                    'name': 'A',
                    'isCapital': True,
                    'level': 1,
                    'star': 1,
                    'population': 0,
                    'wall': 0,
                    'rewarded': 1,
                    'i': 3,
                    'j': 3
                },
                {
                    'id': 2,
                    'playerId': 2,
                    'name': 'A',
                    'isCapital': True,
                    'level': 1,
                    'star': 1,
                    'population': 0,
                    'wall': 0,
                    'rewarded': 1,
                    'i': 7,
                    'j': 7
                }
            ],
        'units': [],
        'finished': False,
        'generatedUnitId': 2,
        'turnPlayer': 0
    }

    village_ij = [(2, 8), (5, 2), (8, 4), (5, 9)]
    fruits = [(2, 3), (4, 4), (3, 2), (6, 7), (8, 8), (6, 6)]
    animals = [(2, 2), (4, 3), (8, 8), (7, 8)]

    for i, j in fruits:
        stub_json['worldMap']['tiles'][i][j]['resource'] = 1
    for i, j in animals:
        stub_json['worldMap']['tiles'][i][j]['resource'] = 3
    for id, city in enumerate(stub_json['cities']):
        i, j = city['i'], city['j']
        for di, dj in product([-1, 0, 1], repeat=2):
            stub_json['worldMap']['tiles'][i + di][j + dj]['territory'] = id + 1
            stub_json['worldMap']['tiles'][i + di][j + dj]['exploredPlayers'].append(id + 1)
            stub_json['worldMap']['tiles'][i + di][j + dj]['city'] = id + 1

    for i, j in village_ij:
        village = {
            'id': len(stub_json['cities']) + 1,
            'playerId': 0,
            'name': "Village",
            'isCapital': False,
            'level': 0,
            'star': 0,
            'population': 0,
            'wall': 0,
            'rewarded': 1,
            'i': i,
            'j': j
        }
        stub_json['cities'].append(village)
        stub_json['worldMap']['tiles'][i][j]['city'] = village['id']

    stub_json_file = 'poly_stub.txt'
    with open(stub_json_file, 'w') as f:
        json.dump(stub_json, f, indent=2)
    stub_game = GameState(stub_json_file)
    stub_game.display()
    #  0 1 2 3 4 5 6 7 8 9 a
    # 0                        0   cities:
    # 1                        1     @33=lvl1,pop0
    # 2     a f         v      2     $77=lvl1,pop0
    # 3     f @                3   units:
    # 4       a f              4   current player=@
    # 5     v             v    5
    # 6             f f        6
    # 7               $ a      7
    # 8         v       a      8
    # 9                        9
    # a                        a
    #    0 1 2 3 4 5 6 7 8 9 a

    actions = stub_game.legal_actions()
    encoder = get_encoder(stub_game.size)
    decoder = get_decoder(stub_game.size)
    assert encoder(ActionType.TrainUnit, 3, 3, 0) in actions
    assert encoder(ActionType.ResearchTechnology, 0, 0, tech_idx['Farming']) in actions
    assert encoder(ActionType.ResearchTechnology, 0, 0, tech_idx['Shields']) in actions
    assert encoder(ActionType.ResearchTechnology, 0, 0, tech_idx['FreeSpirit']) not in actions
    assert encoder(ActionType.ResearchTechnology, 0, 0, tech_idx['Forestry']) not in actions
    assert encoder(ActionType.TileApplyAbility, 3, 2, tile_ability_idx[AbilityType.HarvestFruit]) in actions
    assert encoder(ActionType.TileApplyAbility, 2, 3, tile_ability_idx[AbilityType.HarvestFruit]) in actions

    action = encoder(ActionType.TrainUnit, 3, 3, 0)
    print("Test action:", decoder(action))
    stub_game.apply(encoder(ActionType.TrainUnit, 3, 3, 0))
    assert len(stub_game.units) == 1
    stub_game.display()

    action = encoder(ActionType.TileApplyAbility, 3, 2, tile_ability_idx[AbilityType.HarvestFruit])
    print("Test action:", decoder(action))
    stub_game.apply(action)
    assert stub_game.resource[3 * 11 + 2] == ResourceType.none.value-1
    stub_game.display()

    # newly trained unit cannot move
    action = encoder(ActionType.UnitMove, 3, 3, dir_idx[1, 1])
    assert action not in stub_game.legal_actions()

    stub_game.apply(encoder(ActionType.EndTurn, 0, 0, 0))
    assert stub_game.player == 1

    action = encoder(ActionType.TrainUnit, 7, 7, 0)
    print("Test action:", decoder(action))
    stub_game.apply(action)
    assert len(stub_game.units) == 2
    stub_game.display()

    stub_game.apply(encoder(ActionType.EndTurn, 0, 0, 0))
    assert stub_game.player == 0

    actions = stub_game.legal_actions()
    action = encoder(ActionType.UnitMove, 3, 3, dir_idx[1, 1])
    assert action in actions
    print("Test action:", decoder(action))
    stub_game.apply(action)
    stub_game.display()
    assert stub_game.units[1].x == 4 and stub_game.units[1].y == 4

    stub_game.apply(encoder(ActionType.EndTurn, 0, 0, 0))
    assert stub_game.player == 1
    action = encoder(ActionType.UnitMove, 7, 7, dir_idx[-1, -1])
    print("Test action:", decoder(action))
    stub_game.apply(action)
    stub_game.display()
    assert (stub_game.units[2].x == 6) and (stub_game.units[2].y == 6)

    stub_game.apply(encoder(ActionType.EndTurn, 0, 0, 0))
    assert stub_game.player == 0
    action = encoder(ActionType.UnitMove, 4, 4, dir_idx[1, 1])
    print("Test action:", decoder(action))
    stub_game.apply(action)
    stub_game.display()
    assert (stub_game.units[1].x == 5) and (stub_game.units[1].y == 5)

    action = encoder(ActionType.UnitAttack, 5, 5, dir_idx[1,1])
    actions = stub_game.legal_actions()
    assert action in actions
    stub_game.apply(action)
    assert stub_game.units[1].hp < 10
    assert stub_game.units[2].hp < 10
    stub_game.display()

    game_umove = lambda i,j,i2,j2: encoder(ActionType.UnitMove, i,j,dir_idx[i2-i,j2-j])
    stub_game.apply(0)

    move1 = game_umove(6,6,5,7)
    move2 = game_umove(5,7,5,8)
    move3 = game_umove(5,8,5,9)

    stub_game.apply(move1)
    stub_game.apply(0)
    stub_game.apply(0)

    stub_game.apply(move2)
    stub_game.apply(0)
    stub_game.apply(0)

    stub_game.apply(move3)
    stub_game.apply(0)
    stub_game.apply(0)

    stub_game.display()
    stub_game.apply(0)
    stub_game.apply(0)
    stub_game.display()

    stub_game.apply(encoder(ActionType.TileApplyAbility,5,9,7))
    assert stub_game.get_city_at(5,9).player == 1
    stub_game.display()


if __name__ == "__main__":
    # execute only if run as a script
    unit_tests()
