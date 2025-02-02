import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to extract psi_max from the last line of the CSV file
def extract_psi_max(file_path):
    """
    Extract psi_max from the last line of the CSV file.
    
    Parameters:
        file_path (str): The path to the CSV file.
    
    Returns:
        psi_max (float): The value of psi_max extracted from the last line of the CSV file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        parameter_string = lines[-1].split("; ")
        parameters = {param.split(": ")[0]: float(param.split(": ")[1]) for param in parameter_string}

        return parameters.get("psi_max", -1)
    
# Function to extract m_c from the last line of the CSV file
def extract_m_c(file_path):
    """
    Extract m_c from the last line of the CSV file.
    
    Parameters:
        file_path (str): The path to the CSV file.
    
    Returns:
        m_c (float): The value of m_c extracted from the last line of the CSV file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        parameter_string = lines[-1].split("; ")
        parameters = {param.split(": ")[0]: float(param.split(": ")[1]) for param in parameter_string}

        return parameters.get("m_c", -1)

# Function to process a file and return relevant metrics
def process_file(file_path, vary_psi_max):
    """
    Process a CSV file and return the relevant metrics.
    
    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        psi_max (float): The value of psi_max extracted from the last line of the CSV file.
        fraction_rich (float): The fraction of rich agents in the last time step.
    """
    # Read the CSV file
    df = pd.read_csv(file_path, low_memory=False)
    
    # Calculate fraction of rich agents
    num_unique_agents = df['ID'].nunique()
    last_timestep = df['Time step'].max()
    rich_agents_last_step = df[df['Time step'] == last_timestep]['Rich'].sum()
    fraction_rich = rich_agents_last_step / num_unique_agents

    # Extract psi_max or m_c based on the flag
    if vary_psi_max:
        psi_max = extract_psi_max(file_path)
        return psi_max, fraction_rich
    else:
        m_c = extract_m_c(file_path)
        return m_c, fraction_rich

# Function to process a specific directory
def process_directory(directory, vary_psi_max):
    """
    Process a directory containing CSV files and return the relevant metrics.
    
    Parameters: 
        directory (str): The path to the directory containing CSV files.
        
    Returns:
        results (pd.DataFrame): A DataFrame containing the relevant metrics for each file in the directory.
    """
    # New dataframe to store psi_max or m_c and fraction_rich
    data = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            if vary_psi_max:
                psi_max, fraction_rich = process_file(file_path, vary_psi_max)
                # Add it to dataframe results
                data.append([psi_max, fraction_rich])
            else:
                m_c, fraction_rich = process_file(file_path, vary_psi_max)
                # Add it to dataframe results
                data.append([m_c, fraction_rich])

    if vary_psi_max:
        results = pd.DataFrame(data, columns=['psi_max', 'fraction_rich'])
        results.sort_values(by='psi_max', ascending=True)
    else:
        results = pd.DataFrame(data, columns=['m_c', 'fraction_rich'])
        results.sort_values(by='m_c', ascending=True)
    
    return results

def plot_mult_fss(base_directory, grid_sizes, colors, m_c_values, psi_max_values, vary_psi_max):
    """
    Plot the fraction of rich agents against psi_max or m_c for different grid sizes.
    
    Parameters:
        base_directory (str): The base directory containing the data.
        grid_sizes (list): A list of grid sizes to consider.
        colors (list): A list of colors for the grid sizes.
        m_c_values (list): A list of m_c values to consider.
        psi_max_values (list): A list of psi_max values to consider.
        vary_psi_max (bool): If True, vary psi_max; otherwise, vary m_c.
        
    Returns:
        None (Displays the plot)
    """
    # Calculate number of subplots if subfigures needed in a single row or column
    num_subplots = (len(grid_sizes) * len(m_c_values))//len(grid_sizes) \
    if vary_psi_max else (len(grid_sizes) * len(psi_max_values))//len(grid_sizes)

    # Plot the data by accessing different directories in the base directory
    # Change number of rows and columns as needed (can also do (1, num_subplots) or (num_subplots, 1))
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()  # Flatten axes for easy iteration

    for idx, m_c in enumerate(m_c_values):
        ax = axes[idx]
        x_label_subplot = ""
        y_label_subplot = "Fraction of Rich Agents"

        # Iterate over grid sizes
        for grid_idx, grid_size in tqdm(enumerate(grid_sizes)):
            grid_folder = f"grid_{grid_size}x{grid_size}"
            if vary_psi_max:
                # Access the directory for varying psi_max
                psi_max_path = os.path.join(base_directory, grid_folder, f"varying_psi_max_m_c_{m_c:.1f}")
                
                # Process data from the directory
                assert os.path.exists(psi_max_path), f"Path {psi_max_path} does not exist"
                data = process_directory(psi_max_path, vary_psi_max) if os.path.exists(psi_max_path) else pd.DataFrame()
                
                # Set x label for the subplot
                x_label_subplot = r"Tax Rate ($\psi_{max}$)"
            else:
                # Access the directory for varying m_c
                m_c_path = os.path.join(base_directory, grid_folder, f"varying_m_c_psi_max_{m_c:.1f}")
                
                # Process data from the directory or return empty dataframe if directory does not exist
                assert os.path.exists(m_c_path), f"Path {m_c_path} does not exist"
                data = process_directory(m_c_path, vary_psi_max) if os.path.exists(m_c_path) else pd.DataFrame()
                
                # Set x label for the subplot
                x_label_subplot = r"Charity Contribution (m_c)"
            
            # If data exists, plot it
            psi_max_values, fraction_rich_values = data['psi_max'], data['fraction_rich']
            ax.plot(psi_max_values, fraction_rich_values, color=colors[grid_idx], label=f"Grid {grid_size}x{grid_size}")

        ax.set_title(f"m_c = {m_c:.1f}" if vary_psi_max else f"$\psi_{{max}}$ = {m_c:.1f}")
        ax.set_xlabel(x_label_subplot)
        ax.set_ylabel(y_label_subplot)
        ax.grid()

    # Create a single legend for all subplots
    handles = [plt.Line2D([0], [0], color=colors[i], label=f"Grid {grid_sizes[i]}x{grid_sizes[i]}", linestyle="-") for i in range(len(grid_sizes))]
    plt.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=12)
    # Add a global title
    plt.suptitle(f"Fraction of Rich Agents vs. {x_label_subplot}", fontsize=16, fontweight="bold")

    plt.xlabel(x_label_subplot)
    plt.ylabel(y_label_subplot)
    # Uncomment if saving plots and edit filename if needed
    # plt.savefig("plots/finite_size_scaling.png")
    plt.show()

def plot_1_fss(base_directory, grid_sizes, colors, m_c_values, psi_max_values, vary_psi_max):
    """
    Plot the fraction of rich agents against psi_max or m_c for different grid sizes 
    for a single m_c or psi_max value.
    
    Parameters:
        base_directory (str): The base directory containing the data.
        grid_sizes (list): A list of grid sizes to consider.
        colors (list): A list of colors for the grid sizes.
        m_c_values (list): A list of m_c values to consider.
        psi_max_values (list): A list of psi_max values to consider.
        vary_psi_max (bool): If True, vary psi_max; otherwise, vary m_c.
        
    Returns:
        None (Displays the plot)
    """
    # Assuming only 1 psi_max value if vary_psi_max is True
    if vary_psi_max:
        assert len(m_c_values) == 1, "Only 1 m_c value should be provided when varying psi_max"
    else:
        assert len(psi_max_values) == 1, "Only 1 psi_max value should be provided when varying m_c"

    # Plot the data by accessing different directories in the base directory
    fig = plt.figure(figsize=(8, 6))

    for index, m_c in enumerate(m_c_values):
        for grid_idx, grid_size in enumerate(grid_sizes):
            grid_folder = f"grid_{grid_size}x{grid_size}"
            if vary_psi_max:
                psi_max_path = os.path.join(base_directory, grid_folder, f"varying_psi_max_m_c_{m_c:.1f}")
                # Process data from the directory
                data = process_directory(psi_max_path) if os.path.exists(psi_max_path) else pd.DataFrame()
                # print(data)
            else:
                m_c_path = os.path.join(base_directory, grid_folder, f"varying_m_c_psi_max_{m_c:.1f}")
                # Process data from the directory
                data = process_directory(m_c_path) if os.path.exists(m_c_path) else pd.DataFrame()
                # print(data)
            
            # If data exists, plot it
            psi_max_values_plot, fraction_rich_values = data['psi_max'], data['fraction_rich']
            plt.plot(psi_max_values_plot, fraction_rich_values, color=colors[grid_idx], label=f"Grid {grid_size}x{grid_size}")
        
        plt.title(f"Fraction of Rich Agents (m_c = {m_c:.1f})")
        plt.xlabel(r"Tax Rate ($\psi_{max}$)")
        plt.ylabel("Fraction of Rich Agents")
        plt.grid()
        plt.legend()
        plt.show()

    return None

# Define base directories for sorted data
base_directory = "data/data_sorted"

# Grid sizes and corresponding colors
# Can be extended to include more grid sizes and colors if simulation.py changed to include them
grid_sizes = [5, 10, 20, 30]
colors = ["blue", "red", "green", "purple"]  # Different colors for each grid size

# Different m_c values and psi_max values
# Can be extended to include more values if simulation.py changed to include them
m_c_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
psi_max_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Call the functions to plot the data
plot_mult_fss(base_directory, grid_sizes, colors, m_c_values, psi_max_values, vary_psi_max=False)
# plot_1_fss(base_directory, grid_sizes, colors, m_c_values, psi_max_values, vary_psi_max=True)