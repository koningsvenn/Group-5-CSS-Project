import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import RegularPolygon
from hex_agents import AgentSimulation

# class Visualize():
#     def __init__(self, agents, HexagonalGrid, grid_size, interval, steps):
#         self.agents = agents
#         self.grid = HexagonalGrid(grid_size)
#         self.interval = interval
#         self.steps = steps

#     def hex_to_pixel(x, y):
#             """Convert hexagonal grid coordinates to pixel coordinates."""
#             dx = 3 / 2 * x
#             dy = np.sqrt(3) * (y + (x % 2) / 2)
#             return dx, dy
    
#     def update(frame, ax, agents):
#             # fig, ax = plt.subplots(figsize=(8, 8))
#             ax.clear()
#             ax.set_xlim(-1, self.grid.size * 1.5)
#             ax.set_ylim(-1, self.grid.size * np.sqrt(3))
#             ax.set_aspect('equal')
            
#             # Plot agents as hexagons with wealth intensity
#             for (x, y), wealth in self.agents.items():
#                 px, py = hex_to_pixel(x, y)
#                 color = plt.cm.viridis(wealth / max(self.agents.values()))
#                 hex_patch = RegularPolygon(
#                     (px, py),
#                     numVertices=6,
#                     radius=0.5 / np.cos(np.pi / 6),
#                     orientation=np.pi / 6,
#                     color=color
#                 )
#                 ax.add_patch(hex_patch)

#             # Simulation steps
#             self.propagate()
#             self.transactions(delta_m=1, p_transaction=0.7, inequality_param=0.1)
#             self.apply_tax_and_charity(tax_rate=0.1, tax_threshold=5, charity_rate=0.5)
        
#     def visualize(self, interval=100, steps=50):
#         """Visualize the simulation using matplotlib.animation."""
#         fig, ax = plt.subplots(figsize=(8, 8))
#         ani = FuncAnimation(fig, update, frames=steps, interval=interval, repeat=False)
#         plt.show()

# Create a class for visualization
class Visualization:
    def __init__(self, agent_sim):
        """Initialize the visualization class."""
        self.agent_sim = agent_sim
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

    def hex_to_pixel(self, x, y):
        """Convert hexagonal grid coordinates to pixel coordinates."""
        dx = 3 / 2 * x
        dy = np.sqrt(3) * (y + (x % 2) / 2)
        return dx, dy

    def update(self, frame):
        self.ax.clear()
        self.ax.set_xlim(-1, self.agent_sim.grid.size * 1.5)
        self.ax.set_ylim(-1, self.agent_sim.grid.size * np.sqrt(3))
        self.ax.set_aspect('equal')
        
        # Plot agents as hexagons with wealth intensity
        for (x, y), wealth in self.agent_sim.agents.items():
            px, py = self.hex_to_pixel(x, y)
            color = plt.cm.viridis(wealth / max(self.agent_sim.agents.values()))
            hex_patch = RegularPolygon(
                (px, py),
                numVertices=6,
                radius=0.5 / np.cos(np.pi / 6),
                orientation=np.pi / 6,
                color=color
            )
            self.ax.add_patch(hex_patch)

        # Simulation steps
        self.agent_sim.propagate()
        self.agent_sim.transactions(delta_m=1, p_transaction=0.7, inequality_param=0.1)
        self.agent_sim.apply_tax_and_charity(tax_rate=0.1, tax_threshold=5, charity_rate=0.5)

    def animate(self, interval=100, steps=50):
        ani = FuncAnimation(
            self.fig, self.update, frames=steps, interval=interval, repeat=False
        )
        plt.show()