import numpy as np
from polynet import PolyNet
from mcts import Node, AlphaZeroConfig, SharedStorage, ReplayBuffer
from polygame import GameState, get_decoder, get_encoder, ActionType, action_posibilities
from copy import deepcopy
import math
import multiprocessing as mp
from memory_profiler import profile

import sys
from multiprocessing.connection import Listener, Client
import gc


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay_once(config: AlphaZeroConfig, storage: SharedStorage,
                      replay_buffer: ReplayBuffer):
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: PolyNet):
    game = GameState(config.json_file)
    while not game.terminal() and len(game.history) < config.max_moves:
        print("******* Move #{}*********".format(len(game.history)))
        action, root = run_mcts(config, game, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game


# INFERENCE_SERVER = ('localhost', 6005)
# BUFFER_SERVER = ('localhost', 6006)
# STORAGE_SERVER = ('localhost', 6007)

INFERENCE_SERVER = r'\\.\pipe\inference2'
BUFFER_SERVER = r'\\.\pipe\buffer2'
STORAGE_SERVER = r'\\.\pipe\storage2'


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: GameState, network: PolyNet):
    root = Node(0)
    evaluate(root, game, network)
    add_exploration_noise(config, root)

    for i in range(config.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node)
            scratch_game.apply(action)
            search_path.append(node)

        if i % 100 == 0:
            print("Simulation #{:03}, Search path length = {}".format(i, len(search_path)))

        value = evaluate(node, scratch_game, network)
        backpropagate(search_path, value, scratch_game.to_play())
    return select_action(config, game, root), root


def select_action(config: AlphaZeroConfig, game: GameState, root: Node):
    visit_counts = [(child.visit_count, action)
                    for action, child in root.children.items()]
    if len(game.history) < config.num_sampling_moves:
        _, action = softmax_sample(visit_counts)
    else:
        _, action = max(visit_counts)
    return action


def softmax_sample(visit_counts):
    counts = [visit_count for visit_count, action in visit_counts]
    total_count = sum(counts)
    probabilty = [count / total_count for count in counts]
    idx = np.random.choice(range(len(counts)), p=probabilty)
    return visit_counts[idx]


# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
    action, child = max(node.children.items(),
                        key=lambda x: ucb_score(config, node, x[1]))
    # _, action, child = max((ucb_score(config, node, child), action, child) 
    #                         for action, child in node.children.items())
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
    pb_c = (math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
            + config.pb_c_init)
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


def inference_client(x):
    conn = Client(INFERENCE_SERVER, authkey=b'secret password')
    conn.send(x)
    v, p = conn.recv()
    conn.close()
    if len(p.shape) == 3:
        p = p[np.newaxis, :, :, :]
    return v, p


def buffer_client(game):
    conn = Client(BUFFER_SERVER, authkey=b'secret password')
    conn.send(game)
    conn.close()


def storage_client():
    conn = Client(STORAGE_SERVER, authkey=b'secret password')
    # conn.send('get')
    weights = conn.recv()
    conn.close()
    network = PolyNet()
    network.res_net_model.set_weights(weights)
    return network


import pickle


def buffer_server(replay_buffer):
    print("Start relay buffer server")
    listener = Listener(BUFFER_SERVER, authkey=b'secret password')
    count = 0
    while True:
        conn = listener.accept()
        replay_buffer.save_game(conn.recv())
        conn.close()
        count += 1
        print("************ Buffer ***************")
        with open("buffer.pkl", 'wb') as f:
            pickle.dump(relay_buffer, f)
    listener.close()


def storage_server(storage: SharedStorage):
    print("Start storage server")
    listener = Listener(STORAGE_SERVER, authkey=b'secret password')
    while True:
        conn = listener.accept()
        # command = conn.recv()
        # if command == 'get':
        conn.send(storage.latest_network())
        # elif command == 'post':
        # storage.save_network(*conn.recv())
        conn.close()
    listener.close()


def inference_server():
    print("Start inference server")
    listener = Listener(INFERENCE_SERVER, authkey=b'secret password')
    while True:
        conn = listener.accept()
        # queue_conn.append(conn)
        # queue_x.append(conn.recv())
        # if len(queue_conn) >= 4:
            # network = storage_client()
            # v, p = network.inference(np.stack(queue_x))
            # for i in range(len(queue_conn)):
            #     queue_conn[i].send((v[i], p[i]))
            #     queue_conn[i].close()
            # queue_conn = []
            # queue_x = []
        x = conn.recv()
        network = storage_client()
        v, p = network.inference(x[np.newaxis,:,:,:])
        conn.send((v, p))
        conn.close()
        # gc.collect()
        # print(gc.get_objects()[-600], sys.getsizeof(gc.get_objects()[-600]))
    listener.close()


import time


def run_selfplay_atomic(config: AlphaZeroConfig):
    network = storage_client()
    game = play_game(config, network)
    buffer_client(game)


def run_selfplay_parallel(config: AlphaZeroConfig, storage: SharedStorage,
                          replay_buffer: ReplayBuffer):
    buffer_server_process = mp.Process(target=buffer_server, args=(replay_buffer,))
    storage_server_process = mp.Process(target=storage_server, args=(storage,))
    inference_server_process = mp.Process(target=inference_server)
    buffer_server_process.name = "Buffer server"
    storage_server_process.name = "storage server"
    inference_server_process.name = "infer server"
    buffer_server_process.start()
    storage_server_process.start()
    time.sleep(5)
    inference_server_process.start()
    time.sleep(5)

    # def iterable_config():
    #     yield  config

    # pool = mp.Pool(8)
    # pool.map_async(run_selfplay_atomic, iterable_config())


    # clients = []
    for i in range(4):
        client = mp.Process(target=run_selfplay_atomic, args=(config,))
        # client.name="self play client"
        # clients.append(client)
        # clients[-1].start()
        client.start()
        print("Spawn {}".format(i))

    print("Finished main")


# We use the neural network to obtain a value and policy prediction.
def evaluate(node, game, network):
    # value, policy_logits = network.inference(game.make_image(-1))
    value, policy_logits = inference_client(game.make_image(-1))

    # from pprint import  pprint
    # pprint(action_posibilities)
    # print(sum(action_posibilities.values()))
    # exit()    

    # Expand the node.
    node.to_play = game.to_play()
    policy = {}
    decoder = get_decoder(game.size)
    for action in game.legal_actions():
        action_type, i, j, param = decoder(action)
        policy_code = 0
        for a in ActionType:
            if a == action_type:
                policy_code += param
                break
            policy_code += action_posibilities[a]
        policy[action] = math.exp(policy_logits[0, policy_code, i, j])
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
    policy = None
    del policy
    gc.collect()
    return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path, value, to_play):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else (1 - value)
        node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
    actions = node.children.keys()
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


if __name__ == "__main__":
    # from pprint import pprint

    print(INFERENCE_SERVER)
    print(BUFFER_SERVER)
    print(STORAGE_SERVER)

    alpha_config = AlphaZeroConfig('poly.txt')
    # network = PolyNet()
    # # action, node = run_mcts(alpha_config, state, network)
    # # # run_mcts(alpha_config, state, network)
    # # # print(action_decoder(action))
    # # # pprint({action_decoder(a): (v.visit_count, ucb_score(alpha_config, node, v)) for a, v in node.children.items()})
    # # # network = PolyNet(50, 3, sum(action_posibilities.values()), 11*11)
    config2 = deepcopy(alpha_config)
    config2.max_moves = 20
    # # config2.json_file = stub_json_file
    # game = play_game(config2, network)

    # if argv[1] == 'client':
    #     play_game()
    # elif argv[1] == 'server':

    # run_selfplay_parallel
    storage = SharedStorage()
    relay_buffer = ReplayBuffer(config2)
    run_selfplay_parallel(config2, storage, relay_buffer)
