import numpy as np
import numpy.random as random
from typing import List
from archive.refactoring_attempt.agent import Agent

class Grid:
    def __init__(self: object, 
                 height: int, 
                 width: int, 
                 density: float, 
                 initial_wealth: float,
                 move_probability: float):
        self.height = height
        self.width = width
        self.density = density
        self.initial_wealth = initial_wealth
        self.move_probability = move_probability
        self.agents_in_grid = [] 
        self.visited_in_timestep = []

    def get_agents(self: object):
        """Return the list of agents in the grid."""
        return self.agents_in_grid

    def initialize_grid(self: object):
        """Create a grid with some agents randomly placed."""
        # Create a grid with zeros representing empty cells or ones representing agents
        grid = np.zeros((self.height, self.width))
        # Calculate the number of agents to create based on density
        num_of_total_agents = int(self.height * self.width * self.density)

        num_of_created_agents = 0

        while num_of_created_agents < num_of_total_agents:
            # Get random coordinates to place agent
            # There are 'height' number of rows and 'width' number of columns
            col, row = random.randint(0, self.width - 1), random.randint(0, self.height - 1)

            # If the cell is empty, place an agent
            if grid[row, col] == 0:
                grid[row, col] = 1              # Place agent

                # Agent's attributes
                location = [row, col]           # Agent's location
                money = self.initial_wealth     # Agent's money
                win = False                     # Agent's win status
                transactions = []               # Agent's transaction history
                tax_paid = 0                    # Agent's tax paid
                tax_received = 0                # Agent's tax received
                charity = 0                     # Agent's charity given
                poor = False                    # Agent's poor status
                rich = False                    # Agent's rich status
                moved_tracker = False           # Agent's moved status

                # Create an agent object
                agent = Agent(num_of_created_agents, location, money, \
                              win, transactions, tax_paid, tax_received, \
                              charity, poor, rich, moved_tracker)
                
                # Add agent to the list of agents
                self.agents_in_grid.append(agent)

                # Increment the number of created agents
                num_of_created_agents += 1
                
        return grid
    
    def get_new_location(self: object, 
                         row: int, 
                         col: int):
        """Move agents randomly according to LGCA rules."""
        # Agents can only move one step up, down, left or right
        possible_directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

        # Choose a random direction
        chosen_direction = possible_directions[random.randint(0, possible_directions.shape[0] - 1)]
        
        # Calculate the new coordinates after moving
        row_new = (row + chosen_direction[0]) % self.height
        col_new = (col + chosen_direction[1]) % self.width

        return row_new, col_new
    
    def move_in_timestep(self: object,
                         grid: np.ndarray):
        """Perform a time step where agents move randomly without merging."""
        # Create a new grid to store the updated positions of agents
        new_grid = np.zeros_like(grid)

        for agent in self.agents_in_grid:
            row, col = agent.location

            # Move the agent with a probability of prob_move
            move_condition = random.random()

            if move_condition < self.move_probability:
                # Calculate the new coordinates after moving
                row_new, col_new = self.get_new_location(row, col)
                
                # If the new cell is empty and not visited, move the agent
                if new_grid[row_new, col_new] == 0 and (row_new, col_new) not in self.visited_in_timestep:
                    new_grid[row_new, col_new] = 1          # Place agent
                    self.visited_in_timestep.append((row_new, col_new))    # Mark cell as visited
                    agent.location = [row_new, col_new]     # Update agent's location
                else:
                    new_grid[row, col] = 1                  # Place agent in the same cell
            else:
                new_grid[row, col] = 1                      # Place agent in the same cell

        return new_grid
    
    def economic_transaction(self: object,
                             grid: np.ndarray,
                             delta_m: float,
                             prob_transaction: float,
                             prob_inequality: float):
        """Perform economic transactions between neighboring agents."""
        for agent in self.agents_in_grid:
            # Get the agent's location
            row, col = agent.location
            # List of possible neighboring locations (Moore's neighborhood)
            possible_neighbor_locations = [[row, col+1], [row, col-1], [row+1, col], [row-1, col], \
                                           [row+1, col+1], [row+1, col-1], [row-1, col+1], [row-1, col-1]]
            # List to store neighboring agents
            neighbors = []      
            
            for location in possible_neighbor_locations:
                row_new = location[0] % self.height
                col_new = location[1] % self.width

                # If cell is occupied, add agent to neighbors
                if grid[row_new, col_new] == 1:
                    for neighbor in self.agents_in_grid:
                        if neighbor.location == [row_new, col_new]:
                            neighbors.append(neighbor)
                            # Break the loop if at least 1 neighbor is found
                            break
                if len(neighbors) == 1:
                    break

            # Perform economic transaction
            agent.perform_transaction(neighbors, self.visited_in_timestep, delta_m, \
                                       prob_transaction, prob_inequality)
            
        return None
    
    def redistribute_tax(self: object,
            omega: float,
            psi_max: float,
            delta_m: float,
            tax_threshold: float):
        """Redistribute income tax collected from winning agents."""
        total_tax_collected = 0
        m_max = max([agent.money for agent in self.agents_in_grid])

        for agent in self.agents_in_grid:
            total_tax_collected += agent.collect_tax(m_max, omega, psi_max, delta_m, tax_threshold)

        # Redistribute to all agents
        redistribution_amount = total_tax_collected / len(self.agents_in_grid)
        for agent in self.agents_in_grid:
            agent.money += redistribution_amount
        
        return None
    
    def give_charity(self: object,
                     richness_threshold: float,
                     charity_contribution: float,
                     poverty_threshold: float,
                     charity_probability: float):
        """Give charity to poor agents."""
        poor_agents = [agent for agent in self.agents_in_grid if agent.money < poverty_threshold]
        
        
        return None
    
    

# Initialize the grid
# def initialize_grid(height, width, density, m0):
#     """Create a height x width grid with zeros representing empty cells or ones representing agents."""
#     grid = np.zeros((height, width))
#     no_of_agents = int(height * width * density)
    
#     agents = 0
#     money_of_agent = {}
#     while agents < no_of_agents:
#         n, m = random.randint(0, width - 1), random.randint(0, height - 1)
#         if grid[m, n] == 0:
#             grid[m, n] = 1
#             money_of_agent[agents] = [[m, n], m0, False, [], 0, 0, 0, False, False, False]
#             agents += 1
    
#     return grid, money_of_agent

# def move(m, n, height, width):
#     """Move agents randomly."""
#     directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
#     direction = directions[random.randint(0, directions.shape[0] - 1)]
#     m_new, n_new = (m + direction[0]) % height, (n + direction[1]) % width
#     return m_new, n_new

# def time_step_randwalk(grid, prob_move, money_of_agent):
#     """Perform a time step where agents move randomly without merging."""
#     height, width = grid.shape
#     new_grid = np.zeros_like(grid)
#     visited = set()
    
#     for agent_id, (location, *_) in money_of_agent.items():
#         m, n = location
#         if random.random() < prob_move:
#             m_new, n_new = move(m, n, height, width)
#             if new_grid[m_new, n_new] == 0 and (m_new, n_new) not in visited:
#                 new_grid[m_new, n_new] = 1
#                 visited.add((m_new, n_new))
#                 money_of_agent[agent_id][0] = [m_new, n_new]
#             else:
#                 new_grid[m, n] = 1
#         else:
#             new_grid[m, n] = 1
    
#     return new_grid