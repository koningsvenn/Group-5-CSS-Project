# Only used when the data is unsorted in data/data_unsorted
# This script sorts the data into the following structure:
#   data/data_sorted/grid_{grid_size}x{grid_size}/varying_m_c_psi_max_{psi_max}/
#   OR
#   data/data_sorted/grid_{grid_size}x{grid_size}/varying_psi_max_m_c_{m_c}/

import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

# Define the folder where CSV files are currently stored
source_folder = "data/data_unsorted"
destination_base = "data/data_sorted"

# Ensure the destination base directory exists
os.makedirs(destination_base, exist_ok=True)

# Define possible values for psi_max, m_c, and grid sizes
psi_max_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
m_c_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
grid_sizes = [5, 10, 20, 30]

def extract_parameters(file_path):
    """Extracts parameters from the last line of a CSV file."""
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            last_line = lines[-1].strip()
            params = dict(item.split(": ") for item in last_line.split("; "))
            return {k: float(v) for k, v in params.items()}
    except Exception as e:
        print(f"Error extracting parameters from {file_path}: {e}")
        return {}

def determine_grid_size(file_path):
    """Determines the grid size from unique agent IDs."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        unique_agents = df['ID'][:-1].astype(int).max()
        # Apply conditions for grid size selection
        if unique_agents < 5:
            return 5
        elif unique_agents >= 5 and unique_agents < 20:
            return 10
        elif unique_agents >= 20 and unique_agents < 80:
            return 20
        else:
            return 30
    except Exception as e:
        print(f"Error determining grid size for {file_path}: {e}")
        print(f"Unique agents: {unique_agents}")
        return None

# Process all CSV files in the source folder
for file_name in tqdm(os.listdir(source_folder)):
    file_path = os.path.join(source_folder, file_name)
    if not os.path.isfile(file_path) or not file_name.endswith(".csv"):
        continue
    
    try:
        params = extract_parameters(file_path)
        grid_size = determine_grid_size(file_path)
        # print("Params: ", params)
        # print("Grid size: ", grid_size)
        if grid_size not in grid_sizes:
            print(f"Skipping {file_path}, invalid grid size: {grid_size}")
            continue

        psi_max = params.get("psi_max", -1)
        m_c = params.get("m_c", -1)

        if psi_max not in psi_max_values or m_c not in m_c_values:
            print(f"Skipping {file_path}, invalid psi_max: {psi_max}, m_c: {m_c}")
            continue

        base_grid_folder = os.path.join(destination_base, f"grid_{grid_size}x{grid_size}")
        os.makedirs(base_grid_folder, exist_ok=True)
        
        # Folder 1: Sorting by psi_max with varying m_c
        psi_folder = os.path.join(base_grid_folder, f"varying_m_c_psi_max_{psi_max:.1f}")
        os.makedirs(psi_folder, exist_ok=True)
        shutil.copy(file_path, os.path.join(psi_folder, file_name))

        # Folder 2: Sorting by m_c with varying psi_max
        m_c_folder = os.path.join(base_grid_folder, f"varying_psi_max_m_c_{m_c:.1f}")
        os.makedirs(m_c_folder, exist_ok=True)
        shutil.copy(file_path, os.path.join(m_c_folder, file_name))

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("Sorting complete!")