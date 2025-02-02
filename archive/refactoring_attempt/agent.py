import numpy as np
import numpy.random as random
from typing import List

# Agent class that performs the same functions as the dictionary
class Agent:
    def __init__(self: object, 
                 id: int,
                 location: List[int, int], 
                 money: float, 
                 win: bool, 
                 transactions: List[float], 
                 tax_paid: float, 
                 tax_received: float, 
                 charity: float, 
                 poor: bool, 
                 rich: bool, 
                 moved_tracker: bool):
        self.id = id
        self.location = location
        self.money = money
        self.win = win
        self.transactions = transactions
        self.tax_paid = tax_paid
        self.tax_received = tax_received
        self.charity = charity
        self.poor = poor
        self.rich = rich
        self.moved_tracker = moved_tracker

    def __repr__(self):
        return f"""Agent {self.id}:\n
        Location = {self.location}\n
        Money = {self.money}\n 
        Did it win? = {self.win}\n
        Transactions = {self.transactions}\n
        How much tax paid? = {self.tax_paid}\n 
        How much tax received? = {self.tax_received}\n
        Amount given to charity? = {self.charity}\n
        Is it poor? = {self.poor}\n
        Is it rich? = {self.rich}"""
    
    def perform_transaction(self: object, 
                            neighbors: List[object],
                            visited: List[tuple], 
                            delta_m: int,
                            prob_transaction: float,
                            prob_inequality: float):
        """
        Perform economic transactions between neighboring agents.
        """
        for neighbor in neighbors:
            # Skip if transaction has already occurred
            if (self.id, neighbor.id) in visited or (neighbor.id, self.id) in visited:
                continue 
            else:
                visited.append((self.id, neighbor.id))
            
            # Determines if current agent wins the transaction (value between 0 and 1)
            win_condition = random.random() 

            # Perform transaction if agent has more money than neighbor
            if self.money >= neighbor.money > 0:
                # Agent wins if win_condition less than probability of transaction + inequality
                if win_condition < (prob_transaction / 2 + prob_inequality):
                    self.money += delta_m               # Agent gains money
                    neighbor.money -= delta_m           # Neighbor loses money
                    self.transactions.append(delta_m)   # Record transaction amount
                    self.win = True                     # Agent wins transaction
                # Neighbor wins if win_condition greater than probability of transaction - inequality
                elif win_condition < (prob_transaction / 2 - prob_inequality):
                    self.money -= delta_m               # Agent loses money
                    neighbor.money += delta_m           # Neighbor gains money
                    self.transactions.append(delta_m)   # Record transaction amount
                    self.win = False                    # Agent loses transaction
                # No transaction if agent does not win
                else:
                    pass

        return None
    
    def collect_tax(self: object, 
                  m_max: float, 
                  omega: float, 
                  psi_max: float, 
                  delta_m: float, 
                  tax_threshold: float):
        """
        Collect income tax from winning agents.
        """
        # Check if agent has enough money to pay tax and has won in current timestep
        if self.money > tax_threshold and self.win:
            # Calculate tax liability based on income and maximum wealth among all agents
            psi_i = ((self.money / m_max) ** omega) * psi_max
            tax_liability = psi_i * delta_m

            # Deduct tax liability from agent's money and add to tax paid
            self.money -= tax_liability
            self.tax_paid += tax_liability

        else:
            # If agent is poor or has not won, skip tax payment
            pass
        
        return tax_liability
    
    def collect_charity(self: object,
                        richness_threshold: float,
                        charity_contribution: float,
                        charity_probability: float):
        """
        Collect charity donations from winning agents if there are poor agents in the grid.
        """
        # Random number to determine charity donation
        donation_condition = random.random()
        
        # Check if agent is rich and has won in current timestep
        if self.money > richness_threshold:
            self.rich = True
            if self.win:
                # Agent donates to charity based on charity probability
                if donation_condition < charity_probability:
                    self.money -= charity_contribution
                    self.charity += charity_contribution

        else:
            # If agent is poor or has not won, skip charity donation
            pass
        
        return charity_contribution
        

### TRANSACTIONS ###
# def economic_transaction(self, grid, delta_m, p_t, p_i):
#     """
#     Perform economic transactions between neighboring agents.
#     """
##### Grid behavior #####
#     visited = []
#     height, width = grid.shape

#     for agent_id, (location, money, win, transactions, tax_paid, tax_rec, charity, poor, rich, moved_tracker) in money_of_agent.items():
#         m, n = location
#         neighbors = []

#         # Identify 8 neighbors (D2N8 model)
#         for i in range(-1, 2):
#             for j in range(-1, 2):
#                 if i == 0 and j == 0:
#                     continue
#                 m_new, n_new = (m + i) % height, (n + j) % width
#                 if grid[m_new, n_new] == 1:
#                     for neighbor_id, (neighbor_loc, neighbor_money, *_) in money_of_agent.items():
#                         if neighbor_loc == [m_new, n_new]:
#                             neighbors.append((neighbor_id, neighbor_loc, neighbor_money))
#                             break

#     return money_of_agent, total_transaction, transaction_count, transaction_amounts

### TAX ###
# def tax(money_of_agent, delta_m, psi_max, omega, m_tax):

#     total_tax_revenue = 0
#     m_max = max(agent[1] for agent in money_of_agent.values())

#     ##### Grid behavior #####
#     redistribution = total_tax_revenue / len(money_of_agent)
#     for agent_id in money_of_agent:
#         money_of_agent[agent_id][1] += redistribution

#         return money_of_agent 

### CHARITY ###
# Redistribute charity donations among poor agents
# charity_redistribution = charity_contribution / len(poor_agents)
# for poor_agent in poor_agents:
#     assert poor_agent.poor == True or poor_agent.money < poverty_threshold, \
#         "Poor agent not identified correctly"
#     # Add charity donation to poor agent's money
#     poor_agent.money += charity_redistribution