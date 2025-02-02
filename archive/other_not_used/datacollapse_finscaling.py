import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define base directories for sorted data
base_directory = "data/data_finitite_size_scaling_sorted_v2"
base_directory_varying_mc = "data/data_finitite_size_scaling_sorted_v3"

# Grid sizes and corresponding colors
grid_sizes = [5, 10, 20, 30]
colors = ["blue", "red", "green", "purple"]  # Different colors for each grid size

# Different m_c values and psi_max values
m_c_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
psi_max_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Function to extract psi_max from the last line of the CSV file
def extract_psi_max(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        last_line = lines[-1]
        params = dict(item.split(": ") for item in last_line.strip().split("; "))
        return float(params.get("psi_max", 2))

# Function to process a file and return relevant metrics
def process_file(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    
    unique_agents = df['ID'].nunique()
    last_timestep = df['Time step'].max()
    rich_agents_last_step = df[df['Time step'] == last_timestep]['Rich'].sum()
    fraction_rich = rich_agents_last_step / unique_agents  # Fraction of rich agents

    psi_max = extract_psi_max(file_path)
    return psi_max, fraction_rich

# Function to process a specific directory
def process_directory(directory):
    results = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            psi_max, fraction_rich = process_file(file_path)
            results.append((psi_max, fraction_rich))
    
    return sorted(results)

# 🔹 Data Collapse Plot
nu = 4/3  # Scaling exponent
beta = 5/36  # Order parameter exponent
p_mid = 0.3  # Approximate transition region

plt.figure(figsize=(8,6))

for grid_idx, grid_size in enumerate(grid_sizes):
    grid_folder = f"grid_{grid_size}x{grid_size}"
    psi_max_path = os.path.join(base_directory, grid_folder, "varying_psi_max_m_c_0.3")
    
    # Process data from the directory
    psi_max_data = process_directory(psi_max_path) if os.path.exists(psi_max_path) else []
    
    if psi_max_data:
        psi_values, fraction_rich_values = zip(*psi_max_data)
        psi_values = np.array(psi_values)
        fraction_rich_values = np.array(fraction_rich_values)
        
        # Rescale variables
        x = (psi_values - p_mid) * (grid_size ** (1/nu))
        F = fraction_rich_values * (grid_size ** (-beta/nu))
        
        # Plot collapsed data
        plt.scatter(x, F, color=colors[grid_idx], label=f"Grid {grid_size}x{grid_size}")

plt.xlabel(r"$(p - p_{mid}) L^{1/\nu}$")
plt.ylabel(r"$f L^{-\beta/\nu}$")
plt.legend()
plt.title("Finite-Size Scaling Collapse")
plt.grid()
plt.show()
