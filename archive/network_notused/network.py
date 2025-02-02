import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np

# Define file paths
file_paths = [
    r'data/data_finitite_size_scaling_sorted_v2/grid_5x5/varying_psi_max_m_c_0.0/transaction_log_run_1.csv',
    r'data/data_finitite_size_scaling_sorted_v2/grid_10x10/varying_psi_max_m_c_0.0/transaction_log_run_50.csv',
    r'data/data_finitite_size_scaling_sorted_v2/grid_20x20/varying_psi_max_m_c_0.0/transaction_log_run_99.csv',
    r'data/data_finitite_size_scaling_sorted_v2/grid_30x30/varying_psi_max_m_c_0.0/transaction_log_run_148.csv'
]

plt.figure(figsize=(10, 5))

for file_path in file_paths:
    df = pd.read_csv(file_path)
    
    # Extract grid size from file path
    grid_size = file_path.split('/')[-3]
    
    # Ensure Position column is string and handle NaN values
    df['Position'] = df['Position'].astype(str).fillna('')

    def parse_position(pos):
        try:
            if pos.lower() == 'nan' or pos.strip() == '':
                return None
            return tuple(map(int, pos.strip('"').split(',')))
        except Exception as e:
            print(f"Error parsing position: {pos}, Error: {e}")
            return None

    df['Position'] = df['Position'].apply(parse_position)
    df = df.dropna(subset=['Position'])  # Remove rows with invalid positions

    # Get unique time steps
    time_steps = df['Time step'].unique()
    time_steps = np.arange(0, min(1000, max(time_steps)+1))


    largest_components = []
    network_snapshots = {}
    persistent_graph = nx.Graph()  # Store persistent connections

    # Define function to get neighbors in a 2D grid
    def get_neighbors(pos, occupied_positions):
        x, y = pos
        possible_moves = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        return [p for p in possible_moves if p in occupied_positions]

    # Process each time step
    for t in time_steps:
        G = nx.Graph()
        time_df = df[df['Time step'] == t]
        occupied_positions = set(time_df['Position'])

        # Add nodes and persistent edges
        for pos in occupied_positions:
            G.add_node(pos)
            for neighbor in get_neighbors(pos, occupied_positions):
                persistent_graph.add_edge(pos, neighbor)  # Maintain persistent links

        # Copy persistent graph structure to the current snapshot
        G.add_edges_from(persistent_graph.edges)

        # Track largest component size
        if len(G.nodes) > 0:
            largest_component = max(nx.connected_components(G), key=len)
            largest_components.append(len(largest_component))
        else:
            largest_components.append(0)

        network_snapshots[t] = G

    # Plot largest component size over time
    plt.plot(time_steps, largest_components, linestyle='-', label=f"Grid {grid_size}")

plt.xlabel('Time step')
plt.ylabel('Largest Component Size')
plt.title('Largest Component Size Over Time for Different Grid Sizes')
plt.legend()
plt.grid()
plt.show()
