import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_file(filepath):
    """
    Read a csv file and return the data as a pandas dataframe and parameter list as a dictionary.

    Parameters:
    filepath (str): The path to the csv file.

    Returns:
    data (pd.DataFrame): The data from the csv file.
    """
    # Get the parameters used in the run
    with open(filepath, 'r') as file:
        lines = file.readlines()
        parameter_string = lines[-1].split("; ")
        parameters = {param.split(": ")[0]: float(param.split(": ")[1]) for param in parameter_string}

    # Convert data to pandas df
    data = pd.read_csv(filepath, low_memory=False)

    return data, parameters

def phase_transition_rich(parent_folder, vary_tax=True):
    """
    Reads multiple experiment files from a given folder, extracts psi_max (or m_c) from each file,
    calculates the average number of rich agents, and plots it against psi_max (or m_c) 
    to study phase transitions.

    Args:
        parent_folder (str): Path to the folder containing CSV files.
        vary_tax (bool): If True, extracts psi_max; otherwise, extracts m_c.

    Returns:
        None (Displays the plot)
    """
    # Initialize dictionary to store parameter values and corresponding average rich agent counts
    param_to_agents = {}

    # List all CSV files in the given folder
    csv_files = [f for f in os.listdir(parent_folder) if f.endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in {parent_folder}")
        return

    # Process each CSV file
    for file_name in csv_files:
        file_path = os.path.join(parent_folder, file_name)

        # Read file using read_file()
        df, parameters = read_file(file_path)

        # Determine the varying parameter
        param_label = "psi_max" if vary_tax else "m_c"
        param_value = parameters.get(param_label)

        if param_value is None:
            print(f"Warning: Parameter '{param_label}' not found in {file_path}. Skipping.")
            continue

        # Count number of rich agents per timestep and average across time
        rich_counts = df[df['Rich'] == 1].groupby('Time step').size().mean()

        # Number of rich agents in the last time step
        rich_counts = df[df['Rich'] == 1].groupby('Time step').size().iloc[-1]

        # Store results based on the extracted parameter value
        if param_value in param_to_agents:
            param_to_agents[param_value].append(rich_counts)
        else:
            param_to_agents[param_value] = [rich_counts]

    # Compute the average across runs for each parameter value
    param_values = sorted(param_to_agents.keys())  # Ensure correct plotting order
    avg_poor_counts = [sum(runs) / len(runs) for runs in [param_to_agents[p] for p in param_values]]

    # Plot psi_max (or m_c) vs. average number of rich agents
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, avg_poor_counts, marker='o', linestyle='-')

    # Labels and title
    plt.xlabel(f"{param_label}")
    plt.ylabel("Average Number of Poor Agents")
    plt.title(f"Phase Transition of Poor Agents (Varying {param_label})")
    plt.grid(True)

    # Save plot to file
    plot_filename = parent_folder.split("/", 1)[-1]  # Extract folder name from path
    plt.savefig(f"plots/poor_vary_mc/{plot_filename}.png") 

    # Show plot
    plt.show()

def get_relative_folders(parent_folder):
    """Returns a list of relative folder paths inside the given parent folder."""
    return [os.path.join(parent_folder, folder) for folder in os.listdir(parent_folder) 
            if os.path.isdir(os.path.join(parent_folder, folder))]

def main():
    all_filepaths = []
    subfolders = ["grid_5x5", "grid_10x10", "grid_20x20", "grid_30x30"]
    base_folder = "data/data_sorted"
    
    for folder in subfolders:
        all_filepaths.extend(get_relative_folders(os.path.join(base_folder, folder)))

    for folder in all_filepaths:
        phase_transition_rich(folder, vary_tax=False)

main()