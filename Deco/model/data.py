import glob
import os
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader

import multiprocessing as mp
import pickle
import torch

from .util import process_tsp_and_tours

def solve_problem(args):
    prob, tours_dir = args
    prob_tour_dir = os.path.join(tours_dir, prob.split('/')[-1].replace('.tsp', '.tour'))
    try:
        data = process_tsp_and_tours(prob, prob_tour_dir)
        pickled_result = pickle.dumps(data)
        return pickled_result
    except Exception as e:
        return None

def create_data_list(problems, tours_dir, save_path, mode='train'):
    # Check if the data already exists
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded dataset from {save_path} with size: {len(results)}")
        return results

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Use multiprocessing to process problems
    with mp.Pool(mp.cpu_count()) as pool:
        args = [(prob, tours_dir) for prob in problems]
        results = list(tqdm(pool.imap_unordered(solve_problem, args), total=len(problems), desc="Processing TSP Problems"))

    # Filter out None results
    results = [pickle.loads(pickled_result) for pickled_result in results if pickled_result is not None]

    # Save the results to a pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Processed and saved dataset to {save_path} with size: {len(results)}")
    return results

# Function to split the data into train, validation, and test sets
def split_data(problems, val_size=0.1, test_size=0.1):
    train_problems, temp_problems = train_test_split(problems, test_size=val_size + test_size, random_state=42)
    val_problems, test_problems = train_test_split(temp_problems, test_size=test_size / (val_size + test_size), random_state=42)
    return train_problems, val_problems, test_problems

# Function to create data loaders
def create_data_loaders(train_problems, val_problems, test_problems, tours_dir, save_dir, batch_size=32):
    train_save_path = os.path.join(save_dir, 'train_data.pkl')
    val_save_path = os.path.join(save_dir, 'val_data.pkl')
    test_save_path = os.path.join(save_dir, 'test_data.pkl')

    train_dataset = create_data_list(train_problems, tours_dir, train_save_path)
    val_dataset = create_data_list(val_problems, tours_dir, val_save_path, mode='val')
    test_dataset = create_data_list(test_problems, tours_dir, test_save_path, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    problems = glob.glob('/home/zyzheng23/jianyuan/compute_math_discover/omega_algo/problem_size_100-500/*.tsp')
    tours_dir = '/home/zyzheng23/jianyuan/compute_math_discover/omega_algo/tours_size_100-500/'
    save_dir = '/path/to/save/directory'
    result = create_data_list(problems, tours_dir, os.path.join(save_dir, 'all_data.pkl'))