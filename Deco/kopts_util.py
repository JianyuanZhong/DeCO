import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12/lib64'
import random
import math
import pickle
import networkx as nx
import numpy as np
from rl4co.envs.routing import TSPEnv, TSPGenerator
import multiprocessing as mp
from tqdm import tqdm

def generate_random_k_opts(tour, k, edge_weights=None):
    # Create an undirected graph from the tour
    G = nx.DiGraph()
    for i in range(len(tour)):
        G.add_edge(tour[i], tour[(i + 1) % len(tour)], weight=1)
    
    # S-step: Randomly select an edge to remove
    add_edges = []
    if edge_weights:
        weights = [edge_weights[(u, v)] for u, v in G.edges()]
        remove_edges = random.choices(list(G.edges()), weights=weights, k=1)
    else:
        remove_edges = random.choices(list(G.edges()), k=1)
    endpoints_1, endpoints_2 = remove_edges[0]
    anchor_node = endpoints_1
    shortest_paths_len = nx.single_source_shortest_path_length(G, anchor_node)

    G.remove_edge(*remove_edges[0])
    processed_remove_edges = set(remove_edges)

    # I-step: Randomly select an edge to add, with node rank constraint
    for _ in range(k - 1):
        max_path_len = max(shortest_paths_len[endpoints_1], shortest_paths_len[endpoints_2])
        if max_path_len == len(tour) - 1:
            add_edges.append((endpoints_1, endpoints_2))
            G.add_edge(endpoints_1, endpoints_2)
            new_tour = [1]
            u = 1
            while True:
                u = next(G.successors(u))
                new_tour.append(u)
                if len(new_tour) == len(tour):
                    break
            return remove_edges, add_edges, new_tour
        
        # Select insert_node based on edge weights
        available_nodes = [node for node in tour if shortest_paths_len[node] > max_path_len]
        if edge_weights:
            weights = [1 / edge_weights[(endpoints_1, node)] for node in available_nodes]
            insert_node = random.choices(available_nodes, weights=weights)[0]
        else:
            insert_node = random.choice(available_nodes)
        
        add_edges.append((endpoints_1, insert_node))
        G.add_edge(endpoints_1, insert_node)

        # select the edge to remove
        for edge in G.edges():
            if edge[0] == insert_node and edge not in processed_remove_edges:
                G.remove_edge(*edge)
                remove_edges.append(edge)
                processed_remove_edges.add(edge)
                break
        
        # reverse the direction of the path endpoints_2 to insert_node
        _, next_endpoint = remove_edges[-1]
        u = endpoints_2
        edge_to_reverse = []
        while u != insert_node:
            v = next(G.successors(u))
            edge_to_reverse.append((u, v))
            u = v
        
        for edge in edge_to_reverse:
            G.remove_edge(*edge)
            G.add_edge(edge[1], edge[0])
        
        endpoints_1_, endpoints_2_ = remove_edges[-1][1], remove_edges[-2][1]
        if len(list(G.neighbors(endpoints_1_))) == 1:
            endpoints_1 = endpoints_2_
            endpoints_2 = endpoints_1_
        else:
            endpoints_1 = endpoints_1_
            endpoints_2 = endpoints_2_

        G.add_edge(endpoints_1, endpoints_2)
        shortest_paths_len = nx.single_source_shortest_path_length(G, anchor_node)
        G.remove_edge(endpoints_1, endpoints_2)
    
    # E-step: Add the last edge, by connecting endpoints_1 and endpoints_2
    add_edges.append((endpoints_1, endpoints_2))
    G.add_edge(endpoints_1, endpoints_2)

    new_tour = [1]
    u = 1
    while True:
        u = next(G.successors(u))
        new_tour.append(u)
        if len(new_tour) == len(tour):
            break

    return remove_edges, add_edges, new_tour

def extract_action_sequence(add_edges, remove_edges):
    start_node = remove_edges[0][0]
    insertion_sequence = [v for u,v in add_edges[:-1]]
    end_edge = add_edges[-1]
    return start_node, insertion_sequence, end_edge

def apply_action_sequence(tour, action_sequence):
    start_node, insertion_sequence, end_edge = action_sequence
    
    # build the directed graph
    G = nx.DiGraph()
    for i in range(len(tour)):
        G.add_edge(tour[i], tour[(i + 1) % len(tour)], weight=1)
    shortest_paths_len = nx.single_source_shortest_path_length(G, start_node)
    
    # S-step: Remove the start edge
    anchor_node = start_node
    remove_edge = (anchor_node, next(G.successors(anchor_node)))
    endpoints_1, endpoints_2 = remove_edge
    G.remove_edge(*remove_edge)
    
    processed_remove_edges = set()
    processed_remove_edges.add(remove_edge)
    processed_add_edges = set()

    # I-step: Insert the insertion sequence
    for inode in insertion_sequence:
        max_path_len = max(shortest_paths_len[endpoints_1], shortest_paths_len[endpoints_2])
        assert shortest_paths_len[inode] > max_path_len
        G.add_edge(endpoints_1, inode)
        processed_add_edges.add((endpoints_1, inode))

        # remove the edge from inode to anchor_node
        remove_edge = (-1, -1)
        for edge in G.edges():
            if edge[0] == inode:
                G.remove_edge(*edge)
                remove_edge = edge
                processed_remove_edges.add(edge)
                break
        
        # reverse the direction of the path endpoints_2 to inode
        u = endpoints_2
        edge_to_reverse = []
        while u != inode:
            v = next(G.successors(u))
            edge_to_reverse.append((u, v))
            u = v

        for edge in edge_to_reverse:
            G.remove_edge(*edge)
            G.add_edge(edge[1], edge[0])

        endpoints_1_, endpoints_2_ = remove_edge[1], endpoints_2
        if len(list(G.neighbors(endpoints_1_))) == 1:
            endpoints_1 = endpoints_2_
            endpoints_2 = endpoints_1_
        else:
            endpoints_1 = endpoints_1_
            endpoints_2 = endpoints_2_

        G.add_edge(endpoints_1, endpoints_2)
        shortest_paths_len = nx.single_source_shortest_path_length(G, anchor_node)
        G.remove_edge(endpoints_1, endpoints_2)

    # E-step: Add the last edge
    G.add_edge(endpoints_1, endpoints_2)
    processed_add_edges.add(end_edge)

    new_tour = [1]
    u = 1
    while True:
        u = next(G.successors(u))
        new_tour.append(u)
        if len(new_tour) == len(tour):
            break
    return new_tour, processed_remove_edges, processed_add_edges


def simulated_annealing(tour, edge_weights, max_iterations, initial_temperature, cooling_rate):
    current_tour = tour
    best_tour = tour
    current_cost = calculate_tour_cost(current_tour, edge_weights)
    best_cost = current_cost

    temperature = initial_temperature

    action_sequences = []  # List to store all action sequences
    reverse_count = 0
    cost_history = []

    for iteration in range(max_iterations):
        reverse = False
        if random.random() < 0.5:
            current_tour = current_tour[::-1]
            reverse_count += 1
            reverse = True
        
        # Generate a new tour by applying a random k-opt swap
        k = random.randint(2, 10)  # Choose a random value of k between 2 and 10
        remove_edges, add_edges, new_tour = generate_random_k_opts(current_tour, k, edge_weights)

        action_sequence = extract_action_sequence(add_edges, remove_edges)

        # Calculate the cost of the new tour
        new_cost = calculate_tour_cost(new_tour, edge_weights)
        cost_history.append(new_cost)
        # Calculate the acceptance probability
        delta_cost = new_cost - current_cost
        try:
            acceptance_probability = math.exp(-delta_cost / temperature)
        except OverflowError:
            acceptance_probability = 0.0 if delta_cost > 0 else 1.0

        # Accept the new tour based on the acceptance probability
        if delta_cost < 0 or random.random() < acceptance_probability:
            current_tour = new_tour
            current_cost = new_cost
            action_sequences.append((action_sequence, reverse_count))  # Record the accepted action sequence
            # print(f"Iteration {iteration}: current_cost: {current_cost}, best_cost: {best_cost}")
        else:
            if reverse:
                reverse_count -= 1
                current_tour = current_tour[::-1]

        # Update the best tour if necessary
        if current_cost < best_cost:
            best_tour = current_tour
            best_cost = current_cost

        # Cool down the temperature with cosine restart
        temperature *= cooling_rate

    return best_tour, best_cost, action_sequences, current_tour, cost_history


def calculate_tour_cost(tour, edge_weights):
    cost = 0
    for i in range(len(tour)):
        cost += edge_weights[(tour[i], tour[(i+1) % len(tour)])]
    return cost


def solve_tsp(problem_instance):
    nodes = problem_instance.tolist()

    edge_weights = {}
    for j in range(len(nodes)):
        for k in range(j+1, len(nodes)):
            x1, y1 = nodes[j]
            x2, y2 = nodes[k]
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            edge_weights[(j+1, k+1)] = dist
            edge_weights[(k+1, j+1)] = dist

    # Generate initial greedy tour
    tour_nodes = [j+1 for j in range(len(nodes))]
    tour = [1]
    u = 1
    while len(tour) < len(nodes):
        next_node = min((v for v in tour_nodes if v not in tour), key=lambda v: edge_weights[(u, v)])
        tour.append(next_node)
        u = next_node

    # Run simulated annealing
    best_tour, best_cost, action_sequences, final_tour, cost_history = simulated_annealing(
        tour, edge_weights, max_iterations=10000, initial_temperature=0.1, cooling_rate=0.99
    )

    # print(f"number of action sequences: {len(action_sequences)}")

    return problem_instance, best_tour, best_cost, action_sequences, final_tour, cost_history

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/problems-1000.pkl"):
        generator = TSPGenerator(num_loc=100, seed=42, init_sol_type="greedy")
        problem = generator._generate(batch_size=[1000])
        problems = [problem['locs'][i].numpy() for i in range(problem['locs'].shape[0])]

        # Save the problems as a pickle file
        with open("data/problems-10000.pkl", "wb") as f:
            pickle.dump(problems, f)
    else:
        # Load the problems from the pickle file
        with open("data/problems-10000.pkl", "rb") as f:
            problems = pickle.load(f)

    # Create a multiprocessing pool
    pool = mp.Pool(mp.cpu_count())
    results = []
    for _ in tqdm(pool.imap_unordered(solve_tsp, problems), total=len(problems), desc="Processing TSP Problems"):
        results.append(_)
    pool.close()
    pool.join()

    # Unpack the results
    problem_instances, best_tours, best_costs, all_action_sequences, final_tours, cost_histories = zip(*results)

    # compute average cost
    avg_cost = sum(best_costs) / len(best_costs)
    print(f"Average cost: {avg_cost}")

    # Create data directory if it does not exist
    if not os.path.exists("data"):
        os.makedirs("data")

    # Save the results as pickle
    with open("data/results-10000.pkl", "wb") as f:
        pickle.dump({
            "problem_instances": problem_instances,
            "best_tours": best_tours,
            "best_costs": best_costs,
            "all_action_sequences": all_action_sequences,
            "final_tours": final_tours,
            "cost_histories": cost_histories
        }, f)
