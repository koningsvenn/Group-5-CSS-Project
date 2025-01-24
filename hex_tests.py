from hex_agents import AgentSimulation
from hex_visualize import Visualization
# from hex_grid import HexagonalGrid

# Run the simulation
simulation = AgentSimulation(grid_size=20, num_agents=50, initial_wealth=10)
visualization = Visualization(simulation)
visualization.animate()