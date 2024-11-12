import torch
import torch.nn.functional as F
from torch_geometric.data import Data

import os
import re
import random
import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def read_tsp_file(filepath):
    coords = []
    with open(filepath, 'r') as file:
        start_reading = False
        for line in file:
            if line.strip() == "EOF":
                break
            if start_reading:
                _, x, y = line.split()
                assert (float(x), float(y)) not in coords, f"repeat coordinates {(float(x), float(y))}"
                coords.append((float(x), float(y)))
            if line.strip() == "NODE_COORD_SECTION":
                start_reading = True
    return coords

def read_tour_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    tour = []
    tour_section_found = False
    
    for line in lines:
        if line.strip() == "TOUR_SECTION:":
            tour_section_found = True
            continue
        if tour_section_found:
            if line.strip() == "-1":
                break
            # Read the tour numbers from the line
            line = line.replace("-1", "")
            nodes = [int(i) - 1 for i in line.split()]
            tour.extend(nodes)
    
    return tour

def create_edge_labels(data, tour, mode="input"):
    """Marks edges as 1 if they are in the tour, otherwise 0."""
    num_nodes = data.x.size(0)
    edge_labels = torch.zeros(data.edge_index.size(1), dtype=torch.long)
    edge_map = {(i, j): idx for idx, (i, j) in enumerate(data.edge_index.t().tolist())}
    # print(edge_map.keys())
    
    # Mark the edges in the tour
    for i in range(len(tour)):
        if i < len(tour) - 1:
            node1, node2 = tour[i], tour[i + 1]
        else:
            node1, node2 = tour[-1], tour[0]  # Closing the loop of the tour
        key = tuple(sorted((node1, node2)))

        if mode == "test":
            assert key in edge_map, f"{key} not in edge_map, which is on the optimized tour!"

        if key in edge_map:
            edge_labels[edge_map[key]] = 1
        else:
            print(f"{key} not in edge_map")
    
    return edge_labels

def convert_to_pytorch_geom_data(coords):
    # Convert coordinates to tensor
    node_positions = torch.tensor(coords, dtype=torch.float)

    node_positions = node_positions.div(10000)

    # Number of nodes
    num_nodes = node_positions.size(0)
    
    # Compute pairwise distances using scipy for efficiency
    N = node_positions.size(0)
    dist_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        dist_matrix[i] = torch.norm(node_positions - node_positions[i], dim=1).numpy()
    
    # Compute the MST using scipy
    mst = minimum_spanning_tree(dist_matrix)
    mst_edges = np.array(mst.nonzero()).T

    # Determine the number of nearest neighbors to keep
    num_neighbors = int(50)
    
    # Create edge indices and edge weights
    edge_index = []
    edge_weight = []
    added_edges = set()  # To track added edges and avoid duplicates

    for i in range(num_nodes):
        # Calculate distances from node i to all other nodes
        distances = torch.norm(node_positions - node_positions[i], dim=1)
        
        # Get the indices of the nearest neighbors
        nearest_neighbors = torch.topk(distances, num_neighbors + 1, largest=False).indices[1:]  # Exclude the node itself
        
        # Create edges from node i to its nearest neighbors
        for neighbor in nearest_neighbors:
            edge = tuple(sorted((i, neighbor.item())))  # Ensure each edge is added in a consistent order
            if edge not in added_edges:
                added_edges.add(edge)
                edge_index.append([i, neighbor.item()])
                edge_weight.append(distances[neighbor].item())

    # Add MST edges
    for edge in mst_edges:
        i, j = edge
        edge = tuple(sorted((i, j)))
        if edge not in added_edges:
            added_edges.add(edge)
            edge_index.append([i, j])
            edge_weight.append(dist_matrix[i, j])

    # Convert edge_index and edge_weight to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    
    # Randomly assign 15% of drop_edges_index to be 1
    num_edges = edge_index.size(1)
    drop_edges_index = torch.zeros(num_edges, dtype=torch.float)
    # num_edges_to_drop = int(0.15 * num_edges)
    # drop_indices = random.sample(range(num_edges), num_edges_to_drop)
    # drop_edges_index[drop_indices] = 1

    # Create a dummy mst_label with zeros
    mst_label = torch.zeros(num_edges, dtype=torch.float)

    # Combine node positions into the feature matrix
    feature_tensor = node_positions
    
    # Create PyTorch Geometric data object
    data = Data(x=feature_tensor, pos=node_positions, edge_index=edge_index, edge_attr=edge_weight.unsqueeze(1), 
                drop_edge_label=drop_edges_index, mst_label=mst_label)
    
    return data

def generate_mst_label(node_positions, num_nodes, edge_index, added_edges):
    # Create a NetworkX graph
    G = nx.Graph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = torch.norm(node_positions[i] - node_positions[j]).item()
            G.add_edge(i, j, weight=distance)
    
    # Compute the MST
    mst = nx.minimum_spanning_tree(G)
    mst_edges = set(mst.edges())

    # Ensure MST edges are in the edge_index
    edge_index_set = {tuple(sorted(edge)) for edge in edge_index}
    for edge in mst_edges:
        u, v = edge
        edge_tuple = tuple(sorted((u, v)))
        if edge_tuple not in edge_index_set:
            edge_index.append([u, v])
            edge_index_set.add(edge_tuple)
    edge_index = list(edge_index_set)

    # Drop 10% of edges from edge_index
    num_edges_to_drop = int(0.05 * len(edge_index))

    if random.random() < 0.5:
        # Drop 5% of edges from the MST and 5% from other edges in edge_index
        mst_edges_in_index = [edge for edge in edge_index if tuple(sorted(edge)) in mst_edges]
        other_edges_in_index = [edge for edge in edge_index if tuple(sorted(edge)) not in mst_edges]
        
        mst_edges_to_drop = random.sample(mst_edges_in_index, int(0.025 * len(edge_index)))
        other_edges_to_drop = random.sample(other_edges_in_index, num_edges_to_drop - len(mst_edges_to_drop))
        edges_to_drop = mst_edges_to_drop + other_edges_to_drop

        # Compute the MST
        mst = nx.minimum_spanning_tree(G)
        mst_edges = set(mst.edges())
    else:
        # Drop 10% of edges not on the MST from edge_index
        other_edges_in_index = [edge for edge in edge_index if tuple(sorted(edge)) not in mst_edges]
        edges_to_drop = random.sample(other_edges_in_index, num_edges_to_drop)

    # Ensure MST edges are in the edge_index
    edge_index_set = {tuple(sorted(edge)) for edge in edge_index}
    for edge in mst_edges:
        u, v = edge
        edge_tuple = tuple(sorted((u, v)))
        if edge_tuple not in edge_index_set:
            edge_index.append([u, v])
            edge_index_set.add(edge_tuple)
    edge_index = list(edge_index_set)

    # Generate initial MST labels
    mst_label = []
    for edge in edge_index:
        u, v = edge
        if tuple(sorted((u, v))) in mst_edges:
            mst_label.append(1)
        else:
            mst_label.append(0)

    drop_edges_label = []
    for edge in edge_index:
        u, v = edge
        if tuple(sorted((u, v))) in edges_to_drop:
            drop_edges_label.append(1)
        else:
            drop_edges_label.append(0)
    # drop_edges_label = torch.LongTensor([edge_index.index(edge) for edge in edges_to_drop]).contiguous()
    # print(drop_edges_label)

    return torch.tensor(mst_label, dtype=torch.float), edge_index, torch.tensor(drop_edges_label, dtype=torch.float)

def find_extreme_cost_files(directory):
    # Regular expression to extract the cost from the file name
    cost_pattern = re.compile(r'_(\d+\.\d+)\.tsp$')

    highest_cost_file = None
    lowest_cost_file = None
    highest_cost = float('-inf')
    lowest_cost = float('inf')

    for file_name in os.listdir(directory):
        match = cost_pattern.search(file_name)
        if match:
            cost = float(match.group(1))
            if cost > highest_cost:
                highest_cost = cost
                highest_cost_file = file_name
            if cost < lowest_cost:
                lowest_cost = cost
                lowest_cost_file = file_name

    return highest_cost_file, lowest_cost_file


# Example usage
def process_tsp_and_tours(filepath, tour_pth):
    coords = read_tsp_file(filepath)
    data = convert_to_pytorch_geom_data(coords)

    # tour = read_tour_from_file(tour_pth)
    # tour_edge_attr = create_edge_labels(data, tour)
    # # print(data.edge_attr.shape, tour_edge_attr.shape)
    # data.edge_attr = torch.cat([data.edge_attr, tour_edge_attr.view(-1, 1)], dim=-1)

    tour = read_tour_from_file(tour_pth)
    label = create_edge_labels(data, tour, mode="input")
    data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, drop_edge_label=data.drop_edge_label, mst_label=data.mst_label, edge_label=label)
    
    return data


if __name__ == "__main__":
    # Paths
    problem = "/home/jianyuanzhong/COGraphTokenizer/scripts/data/problem_size_100/xqf_0_100.tsp"
    tour_pth = '/home/jianyuanzhong/COGraphTokenizer/scripts/data/concorde_solution_100/xqf_0_100.tour'

    data = process_tsp_and_tours(problem, tour_pth)
    print(data)
    print(data.x)