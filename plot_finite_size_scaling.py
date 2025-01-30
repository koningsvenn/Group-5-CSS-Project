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

# 🔹 First Figure: Varying psi_max for fixed m_c
fig1, axes1 = plt.subplots(2, 4, figsize=(15, 10), sharex=True, sharey=True)
axes1 = axes1.flatten()  # Flatten axes for easy iteration

for idx, m_c in enumerate(m_c_values):
    ax = axes1[idx]
    
    for grid_idx, grid_size in enumerate(grid_sizes):
        grid_folder = f"grid_{grid_size}x{grid_size}"
        psi_max_path = os.path.join(base_directory, grid_folder, f"varying_psi_max_m_c_{m_c:.1f}")
        
        # Process data from the directory
        psi_max_data = process_directory(psi_max_path) if os.path.exists(psi_max_path) else []
        
        # If data exists, plot it
        if psi_max_data:
            psi_values, fraction_rich_values = zip(*psi_max_data)
            ax.plot(psi_values, fraction_rich_values, color=colors[grid_idx], label=f"Grid {grid_size}x{grid_size}")

    # Formatting each subplot
    ax.set_title(f"m_c = {m_c:.1f}")
    ax.set_xlabel("Tax Rate (psi_max)")
    ax.set_ylabel("Fraction of Rich Agents")
    ax.grid()

# Create a single legend for all subplots
handles = [plt.Line2D([0], [0], color=colors[i], label=f"Grid {grid_sizes[i]}x{grid_sizes[i]}", linestyle="-") for i in range(len(grid_sizes))]

# Add a global title
fig1.suptitle("Fraction of Rich Agents vs. Tax Rate for Different m_c Values", fontsize=16, fontweight="bold", y=1.02)

# Adjust layout to ensure space for the legend
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leaves space at the top for the title and at the bottom for the legend

# Add legend below the plots
fig1.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=12)

# Show the first plot
plt.show()









# Grid sizes and corresponding colors
grid_sizes = [5, 10, 20, 30]
colors = ["blue", "red", "green", "purple"]  # Different colors for each grid size

# Different psi_max values (Each subplot represents one psi_max value)
psi_max_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Function to process a file and return relevant metrics
def process_file(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    
    unique_agents = df['ID'].nunique()
    last_timestep = df['Time step'].max()
    rich_agents_last_step = df[df['Time step'] == last_timestep]['Rich'].sum()
    fraction_rich = rich_agents_last_step / unique_agents  # Fraction of rich agents

    # Extract m_c value from the last line of the file
    with open(file_path, "r") as file:
        lines = file.readlines()
        last_line = lines[-1]
        params = dict(item.split(": ") for item in last_line.strip().split("; "))
        m_c = float(params.get("m_c", -1))

    return m_c, fraction_rich

# Function to process a specific directory (varying m_c for a fixed psi_max)
def process_directory(directory):
    results = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            m_c, fraction_rich = process_file(file_path)
            results.append((m_c, fraction_rich))
    
    return sorted(results)  # Ensure results are sorted by m_c

# 🔹 Second Figure: Varying m_c for fixed psi_max
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 10), sharex=True, sharey=True)
axes2 = axes2.flatten()  # Flatten axes for easy iteration

for idx, psi_max in enumerate(psi_max_values):
    ax = axes2[idx]
    
    for grid_idx, grid_size in enumerate(grid_sizes):
        grid_folder = f"grid_{grid_size}x{grid_size}"
        m_c_path = os.path.join(base_directory_varying_mc, grid_folder, f"varying_m_c_psi_max_{psi_max:.1f}")
        
        # Process data from the directory
        m_c_data = process_directory(m_c_path) if os.path.exists(m_c_path) else []
        
        # If data exists, plot it
        if m_c_data:
            m_c_values_sorted, fraction_rich_values = zip(*m_c_data)
            ax.plot(m_c_values_sorted, fraction_rich_values, color=colors[grid_idx], label=f"Grid {grid_size}x{grid_size}")

    # Formatting each subplot
    ax.set_title(f"ψ_max = {psi_max:.1f}")
    ax.set_xlabel("m_c")
    ax.set_ylabel("Fraction of Rich Agents")
    ax.grid()

# Create a single legend for all subplots
handles2 = [plt.Line2D([0], [0], color=colors[i], label=f"Grid {grid_sizes[i]}x{grid_sizes[i]}", linestyle="-") for i in range(len(grid_sizes))]

# Add a global title
fig2.suptitle("Fraction of Rich Agents vs. m_c for Different ψ_max Values", fontsize=16, fontweight="bold", y=1.02)

# Adjust layout to ensure space for the legend
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leaves space at the top for the title and at the bottom for the legend

# Add legend below the plots
fig2.legend(handles=handles2, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=12)

# Show the second plot
plt.show()