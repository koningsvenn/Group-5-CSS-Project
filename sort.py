import os
import shutil
import pandas as pd
import numpy as np
import glob

# Define the folder where CSV files are currently stored
source_folder = "data/data_finitite_size_scaling"
destination_base = "data/data_finitite_size_scaling_sorted"

# Ensure the destination base directory exists
os.makedirs(destination_base, exist_ok=True)

# Define possible values for psi_max and m_c
psi_max_values = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
m_c_values = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}  # Adjust if necessary

# Function to extract parameters from the last line of a CSV file
def extract_parameters(file_path):
    """Extracts parameters from the last line of the CSV file."""
    with open(file_path, "r") as file:
        lines = file.readlines()
        last_line = lines[-1].strip()
        params = dict(item.split(": ") for item in last_line.split("; "))
        return {k: float(v) for k, v in params.items()}

# Function to determine the grid size from unique agent IDs
def determine_grid_size(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    unique_agents = df['ID'].nunique()
    area = unique_agents * 5
    possible_sizes = [5, 10, 20, 30]
    L = min(possible_sizes, key=lambda x: abs(x**2 - area))  
    return L
# Process all CSV files in the source folder
csv_files = glob.glob(os.path.join(source_folder, "*.csv"))

for file in csv_files:
    try:
        params = extract_parameters(file)
        grid_size = determine_grid_size(file)

        # Ensure grid size is valid
        if grid_size not in {5, 10, 20, 30}:
            print("grid does not exist")
            continue  # Skip if it doesn't match expected grid sizes

        # Extract required parameters
        psi_max = params.get("psi_max", -1)
        m_c = params.get("m_c", -1)

        # Define destination folders based on conditions
        if psi_max in psi_max_values == .1:
            target_folder = os.path.join(destination_base, f"grid_{grid_size}x{grid_size}", "varying_psi_max_m_c_0.1")
        elif m_c in m_c_values and psi_max == 0.3:
            target_folder = os.path.join(destination_base, f"grid_{grid_size}x{grid_size}", "varying_m_c_psi_max_0.3")
        else:
            continue  # Skip files that don't fit into these categories

        # Ensure the destination folder exists
        os.makedirs(target_folder, exist_ok=True)

        # Move the file to the appropriate folder
        shutil.move(file, os.path.join(target_folder, os.path.basename(file)))

    except Exception as e:
        print(f"Error processing {file}: {e}")

print("Sorting complete!")
