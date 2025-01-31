import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define base directory for the new structure
base_directory_varying_mc = "data/data_finitite_size_scaling_sorted_v3"

# Grid sizes and corresponding colors
grid_sizes = [5, 10, 20, 30]
colors = ["blue", "red", "green", "purple"]  # Different colors for each grid size

# Different m_c values and psi_max values
m_c_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
psi_max_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Function to extract m_c from the last line of the CSV file
def extract_m_c(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        last_line = lines[-1]
        params = dict(item.split(": ") for item in last_line.strip().split("; "))
        return float(params.get("m_c", -1))

# Function to process a file and return relevant metrics
def process_file(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    
    unique_agents = df['ID'].nunique()
    last_timestep = df['Time step'].max()
    rich_agents_last_step = df[df['Time step'] == last_timestep]['Rich'].sum()
    fraction_rich = rich_agents_last_step / unique_agents  # Fraction of rich agents

    m_c = extract_m_c(file_path)
    return m_c, fraction_rich

# Function to process a specific directory
def process_directory(directory):
    results = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            m_c, fraction_rich = process_file(file_path)
            results.append((m_c, fraction_rich))
    
    return sorted(results)

# 🔹 Second Figure: Varying m_c for fixed psi_max
fig2, axes2 = plt.subplots(1, 7, figsize=(21, 4), sharex=True, sharey=True)  # 1 row, 7 columns

for idx, psi_max in enumerate(psi_max_values):
    ax = axes2[idx]
    
    for grid_idx, grid_size in enumerate(grid_sizes):
        grid_folder = f"grid_{grid_size}x{grid_size}"
        mc_path = os.path.join(base_directory_varying_mc, grid_folder, f"varying_m_c_psi_max_{psi_max:.1f}")
        
        # Process data from the directory
        mc_data = process_directory(mc_path) if os.path.exists(mc_path) else []
        
        # If data exists, plot it
        if mc_data:
            mc_values, fraction_rich_values = zip(*mc_data)
            ax.plot(mc_values, fraction_rich_values, color=colors[grid_idx], label=f"Grid {grid_size}x{grid_size}")

    # Formatting each subplot
    ax.set_title(f"ψ_max = {psi_max:.1f}")
    ax.set_xlabel("Charity Contribution(m_c)")
    if idx == 0:
        ax.set_ylabel("Fraction of Rich Agents")  # Only leftmost plot has y-label
    ax.grid()

# Create a single legend for all subplots
handles = [plt.Line2D([0], [0], color=colors[i], label=f"Grid {grid_sizes[i]}x{grid_sizes[i]}", linestyle="-") for i in range(len(grid_sizes))]

# Add a global title
fig2.suptitle("Fraction of Rich Agents vs. Charity Contribution for Different ψ_max Values", fontsize=16, fontweight="bold", y=1.05)

# Adjust layout and move legend slightly higher
plt.tight_layout(rect=[0, 0.1, 1, 0.95])  
fig2.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=12)  # Move legend higher
plt.subplots_adjust(bottom=0.15)  # Add space below the figure for the legend

# Show the second plot
plt.show()
