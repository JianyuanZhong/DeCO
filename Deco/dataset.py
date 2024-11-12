import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import pickle
from kopts_util import apply_action_sequence
from scipy.sparse.csgraph import minimum_spanning_tree
import math
import numpy as np
from tqdm import tqdm
import random
import os
import resource
from torch.utils.data import DataLoader


# Increase system file limit
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

class TSPDataset(Dataset):
    def __init__(self, data_path, num_samples=1000, num_nodes=100):
        # Load the data from the pickle file
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        
        self.problem_instances = self.data["problem_instances"][:num_samples]
        self.best_tours = self.data["best_tours"][:num_samples]    
        self.best_costs = self.data["best_costs"][:num_samples]
        self.all_action_sequences = self.data["all_action_sequences"][:num_samples]
        self.final_tours = self.data["final_tours"][:num_samples]
        self.num_actions = num_nodes + 4 # 0, 101, 102, 103

        # Check if processed data exists
        processed_path = data_path.replace('.pkl', '_processed.pkl')
        if os.path.exists(processed_path):
            with open(processed_path, 'rb') as f:
                processed_data = pickle.load(f)
                self.problem_instances = processed_data['problem_instances']
                self.pyg_data = processed_data['pyg_data']
                self.current_tours = processed_data['current_tours']
                self.optimized_tours = processed_data['optimized_tours']
                self.all_action_sequences = processed_data['all_action_sequences']
        else:
            self.process_data()
            # Save processed data
            processed_data = {
                'problem_instances': self.problem_instances,
                'pyg_data': self.pyg_data,
                'current_tours': self.current_tours,
                'optimized_tours': self.optimized_tours,
                'all_action_sequences': self.all_action_sequences
            }
            with open(processed_path, 'wb') as f:
                pickle.dump(processed_data, f)

    def __len__(self):
        return len(self.problem_instances) * 10

    def convert_to_pytorch_geom_data(self, coords):
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

    def process_data(self):
        data = list(zip(self.problem_instances, self.all_action_sequences))

        self.problem_instances = []
        self.all_action_sequences = []
        self.current_tours = []
        self.optimized_tours = []
        self.pyg_data = []

        for problem_instance, action_sequences in tqdm(data, desc="Processing data"):
            nodes = problem_instance.tolist()

            pyg_data = self.convert_to_pytorch_geom_data(nodes)

            edge_weights = {}
            for j in range(len(nodes)):
                for k in range(j+1, len(nodes)):
                    x1, y1 = nodes[j]
                    x2, y2 = nodes[k]
                    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    edge_weights[(j+1, k+1)] = dist
                    edge_weights[(k+1, j+1)] = dist

            tour_nodes = [j+1 for j in range(len(nodes))]
            tour = [1]
            u = 1
            while len(tour) < len(nodes):
                next_node = min((v for v in tour_nodes if v not in tour), key=lambda v: edge_weights[(u, v)])
                tour.append(next_node)
                u = next_node

            optimized_tours = []
            current_tours = []
            reverses = []
            action_sequences_ = []
            current_reverse_count = 0
            for action_sequence, reverse_count in action_sequences:
                reverse = False
                current_tours.append(tour)
                if reverse_count > current_reverse_count:
                    tour = tour[::-1]
                    current_reverse_count = reverse_count
                    reverse = True
                try:
                    tour, processed_remove_edges, processed_add_edges = apply_action_sequence(tour, action_sequence)
                except AssertionError:
                    tour = tour[::-1]
                    tour, processed_remove_edges, processed_add_edges = apply_action_sequence(tour, action_sequence)
                    reverse = True
                optimized_tours.append(tour)

                start_node, insertions_nodes, _ = action_sequence
                action_sequence_ = [reverse] + [start_node] + insertions_nodes
                action_sequences_.append(action_sequence_)

            
            self.problem_instances.append(problem_instance)
            self.pyg_data.append(pyg_data)
            self.current_tours.append(current_tours)
            self.optimized_tours.append(optimized_tours)
            self.all_action_sequences.append(action_sequences_)

            
    def __getitem__(self, idx):
        current_tours = self.current_tours[idx // 10]

        # decide a random index for the current tour
        current_tour_idx = random.randint(0, len(current_tours) - 1)

        # decide a number of steps to take
        num_steps = 1

        # Get the problem instance and corresponding best tour
        problem_instance = self.problem_instances[idx // 10]
        pyg_data = self.pyg_data[idx // 10]
        current_tour = self.current_tours[idx // 10][current_tour_idx]
        optimized_tour = self.optimized_tours[idx // 10][current_tour_idx]
        # action_sequence = sum(self.all_action_sequences[idx // 10][current_tour_idx:current_tour_idx+num_steps], [])
        action_sequence = self.all_action_sequences[idx // 10][current_tour_idx]
        # action_sequence = sum(action_sequence, [])
        action_sequence = [self.num_actions - 2 if x is True else self.num_actions - 3 if x is False else x for x in action_sequence]
        input_action_sequence  = [self.num_actions - 1] + action_sequence
        target_action_sequence = action_sequence + [self.num_actions - 1]

        seq_len = 8 * num_steps # TODO: make this dynamic
        # Pad action sequence to length 5
        if len(input_action_sequence) < seq_len:
            input_action_sequence.extend([0] * (seq_len - len(input_action_sequence)))
        elif len(input_action_sequence) > seq_len:
            input_action_sequence = input_action_sequence[:seq_len]
        
        if len(target_action_sequence) < seq_len:
            target_action_sequence.extend([0] * (seq_len - len(target_action_sequence)))
        elif len(target_action_sequence) > seq_len:
            target_action_sequence = target_action_sequence[:seq_len]

        # Convert data to tensors
        problem_tensor = torch.tensor(problem_instance, dtype=torch.float32)
        optimized_tour_tensor = torch.tensor(optimized_tour, dtype=torch.long)
        current_tour_tensor = torch.tensor(current_tour, dtype=torch.long)
        input_action_sequence_tensor = torch.tensor(input_action_sequence, dtype=torch.long)
        target_action_sequence_tensor = torch.tensor(target_action_sequence, dtype=torch.long)
        return problem_tensor, pyg_data, current_tour_tensor, optimized_tour_tensor, input_action_sequence_tensor, target_action_sequence_tensor
    
def collate_fn(batch):
    # Separate the batch into individual components
    problem_tensors, pyg_data_list, current_tours, optimized_tours, input_action_sequences, target_action_sequences = zip(*batch)
    
    # Stack regular tensors
    problem_tensors = torch.stack(problem_tensors)
    current_tours = torch.stack(current_tours)
    optimized_tours = torch.stack(optimized_tours)
    input_action_sequences = torch.stack(input_action_sequences)
    target_action_sequences = torch.stack(target_action_sequences)
    
    # Batch PyG data
    pyg_batch = Batch.from_data_list(pyg_data_list)
    
    return problem_tensors, pyg_batch, current_tours, optimized_tours, input_action_sequences, target_action_sequences


if __name__ == "__main__":
    dataset = TSPDataset("data/results-10000.pkl")
    print(len(dataset))
    problem_tensor, pyg_data, current_tour_tensor, optimized_tour_tensor, input_action_sequence_tensor, target_action_sequence_tensor = dataset.__getitem__(23)
    print(input_action_sequence_tensor)
    print(target_action_sequence_tensor)

    max_seq_len = 0
    action_sequences = []
    for data in tqdm(dataset, desc="Calculating sequence stats"):
        problem_tensor, pyg_data, current_tour_tensor, optimized_tour_tensor, input_action_sequence_tensor, target_action_sequence_tensor = data
        max_seq_len = max(max_seq_len, len(input_action_sequence_tensor))
        action_sequences.append(input_action_sequence_tensor.shape[-1])
        raise Exception("Stop here")
    
    action_sequences = np.array(action_sequences)
    print(f"Max sequence length: {max_seq_len}")
    print(f"Mean: {action_sequences.mean():.2f}")
    print(f"Std: {action_sequences.std():.2f}")
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Test the dataloaders
    for batch in tqdm(train_loader, desc="Testing dataloader"):
        pass