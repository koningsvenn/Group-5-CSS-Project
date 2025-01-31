import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

# Load dataset
file_path = r'data/data_finitite_size_scaling_sorted_v2/grid_20x20/varying_psi_max_m_c_0.0/transaction_log_run_99.csv'
df = pd.read_csv(file_path)

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

largest_components = []
average_degrees = []
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



    # Compute average degree
    if len(G.nodes) > 0:
        avg_degree = np.mean([d for _, d in G.degree()])
    else:
        avg_degree = 0
    average_degrees.append(avg_degree)
    
    network_snapshots[t] = G

# Solve for theoretical giant component size using S = 1 - exp(-<k>S)
def giant_component_eq(S, k):
    return S - (1 - np.exp(-k * S))

# Compute theoretical S
theoretical_S = [fsolve(giant_component_eq, 0.5, args=(k))[0] if k > 0 else 0 for k in average_degrees]

# Plot largest component size over time vs. theoretical prediction
plt.figure(figsize=(10, 5))
plt.plot(time_steps, largest_components, marker='o', linestyle='-', label="Simulated Giant Component Size")
plt.plot(time_steps, np.array(theoretical_S) * len(network_snapshots[time_steps[-1]]), linestyle='--', label="Theoretical Prediction", color='red')
plt.xlabel('Time step')
plt.ylabel('Largest Component Size')
plt.title('Largest Component Size Over Time vs Theoretical Prediction')
plt.legend()
plt.grid()
plt.show()

# Compute component size distribution at the final time step
final_components = [len(c) for c in nx.connected_components(network_snapshots[time_steps[-1]])]

# Plot component size distribution (log-log scale)
plt.figure(figsize=(8, 5))
plt.hist(final_components, bins=np.logspace(0.1, np.log10(max(final_components)), 20), density=True, alpha=0.75)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Component Size')
plt.ylabel('Probability Density')
plt.title('Component Size Distribution')
plt.grid()
plt.show()
