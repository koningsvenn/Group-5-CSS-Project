import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import RegularPolygon
from hex_grid import HexagonalGrid

class AgentSimulation:
    def __init__(self, grid_size, num_agents, initial_wealth):
        """Initialize the simulation."""
        self.grid = HexagonalGrid(grid_size)
        self.grid.random_initialize(num_agents)
        self.agents = {
            (x, y): initial_wealth
            for x, y in zip(*np.where(self.grid.grid == 1))
        }
        self.num_agents = num_agents

    def propagate(self):
        """Move agents to random neighboring cells."""
        new_agents = {}
        for (x, y), wealth in self.agents.items():
            neighbors = self.grid.find_neighbors(x, y)
            empty_neighbors = [n for n in neighbors if n not in self.agents]
            if empty_neighbors:
                new_position = empty_neighbors[np.random.randint(len(empty_neighbors))]
                new_agents[new_position] = wealth
            else:
                new_agents[(x, y)] = wealth
        self.agents = new_agents

    def transactions(self, delta_m, p_transaction, inequality_param):
        """Handle transactions between neighboring agents."""
        updated_agents = self.agents.copy()
        for (x, y), wealth in self.agents.items():
            neighbors = self.grid.find_neighbors(x, y)
            for nx, ny in neighbors:
                if (nx, ny) in self.agents:
                    if np.random.rand() < p_transaction:
                        neighbor_wealth = self.agents[(nx, ny)]
                        if wealth > neighbor_wealth:
                            probability = 0.5 + inequality_param
                        else:
                            probability = 0.5 - inequality_param
                        if np.random.rand() < probability:
                            updated_agents[(x, y)] = max(wealth - delta_m, 0)
                            updated_agents[(nx, ny)] += delta_m
        self.agents = updated_agents

    def apply_tax_and_charity(self, tax_rate, tax_threshold, charity_rate):
        """Apply taxation mechanisms."""
        total_tax = 0
        for position, wealth in self.agents.items():
            if wealth > tax_threshold:
                tax = tax_rate * (wealth - tax_threshold)
                total_tax += tax
                self.agents[position] -= tax
        charity_pool = total_tax * charity_rate
        poor_agents = [pos for pos, wealth in self.agents.items() if wealth < tax_threshold]
        if poor_agents:
            donation = charity_pool / len(poor_agents)
            for pos in poor_agents:
                self.agents[pos] += donation


