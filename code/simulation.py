import numpy as np 
import numpy.random as random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from plotting_scripts.plot_utils import *

# Only needed if you want to run this script from the command line
# import argparse

# Dictionary to track attributes of each agent
money_of_agent = {}
data_log = []  # To store the transaction details per timestep

# Set up the grid
def initialize_grid(height, width, density, m0):
    """Create a height x width grid with zeros representing empty cells or integers 
    to represent person size
    
    Parameters:
        height (int): The height of the grid.
        width (int): The width of the grid.
        density (float): The density of agents in the grid.
        m0 (int): The initial amount of money each agent has.
    
    Returns:
        grid (np.array): A grid with agents randomly placed.
    """
    # Ensure global variables are used
    global money_of_agent
    # Ensure empty dictionary
    money_of_agent.clear()

    # Create empty grid
    grid = np.zeros((height, width))  

    # Calculate number of agents
    no_of_agents = int(height * width * density) 
    
    # select random coordinates for each person
    agents = 0
    while agents < no_of_agents:
        # Get random coordinates
        n = random.randint(0, width - 1)       # y-coordinate
        m = random.randint(0, height - 1)      # x-coordinate

        if grid[m, n] == 0:
            grid[m, n] = 1      # add agent to grid
            # add agent info
            location = [m, n]
            money = m0
            win = False
            moved_tracker = False
            transactions = []
            tax_amt_paid = 0
            tax_amt_received = 0
            charity_amt = 0
            poor = False
            rich = False
            money_of_agent[agents] = [location, money, win, transactions, tax_amt_paid, \
                                      tax_amt_received, charity_amt, poor, rich, moved_tracker]
        # Increment agents
        agents += 1
    return grid 

def record_transaction_data(timestep):
    """Record transaction data for each agent at a given timestep.
    
    Args:
        timestep: The current timestep.
        m_0: The initial amount of money each agent has.
    """
    global data_log

    for agent_id, (location, money, win, transactions, tax_amt_paid, tax_amt_received, charity_amt, poor, rich, moved_tracker) in money_of_agent.items():
        num_neighbors = len([neighbor_id for neighbor_id, (neighbor_loc, _, _, _, _, _, _, _, _, _) in money_of_agent.items() if np.linalg.norm(np.array(location) - np.array(neighbor_loc)) <= np.sqrt(2)])
        transacted = any(t != 0 for t in transactions)
        location_string = f"{location[0]},{location[1]}"

        data_log.append({
            "ID": agent_id,
            "Time step": timestep,
            "Position": location_string,
            "Number of neighbors": num_neighbors,
            "Moved": moved_tracker,
            "Transacted": transacted,
            "Won": win,	
            "Poor": poor,
            "Rich": rich,
            "Amount of income gained/lost": [transactions[-1] if len(transactions) > 0 else 0][0],
            "Amount of tax paid": tax_amt_paid,
            "Amount of tax received": tax_amt_received,
            "Amount of charity given": -1*charity_amt if charity_amt < 0 else 0,
            "Amount of charity received": charity_amt if charity_amt > 0 else 0,
            "Total wealth": money
        })

def move(m, n, height, width):
    """
    Move the persons randomly within the given boundaries.
    
    Parameters:
        m (int): The current row position.
        n (int): The current column position.
        height (int): The total number of rows (height of the grid).
        width (int): The total number of columns (width of the grid).
    
    Returns:
        tuple: A tuple containing the new row and column positions (m_new, n_new).
    """
    # direction options
    directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])  # right, left, bottom, top
    direction = directions[random.randint(0, directions.shape[0] - 1)]  # choose a random row
    
    # calculate new coordinates based on direction
    m_new = (m + direction[0]) % height  # keep within boundary
    n_new = (n + direction[1]) % width   # keep within boundary
    
    return m_new, n_new

def economic_transaction(grid, money_of_agent, delta_m, p_t, p_i):
    """
    Perform economic transactions between neighboring agents. Also checks total amount of transacted money per timestep.
    
    Parameters:
        grid: The grid representing agent positions.
        money_of_agent: A dictionary mapping agent IDs to their location and money.
        delta_m: Fixed amount of money transferred in a transaction.
        p_t: Probability of a transaction occurring between neighbors.
        p_i: Inequality parameter to control transaction probabilities.
    
    Returns:
        Updated money_of_agent dictionary.
    """
    total_transaction = 0       # Total money transacted (for possible graphs)
    transaction_count = 0       # Total number of transactions (for possible graphs)
    transaction_amounts = []    # List for tracking transaction amounts (for possible graphs)

    height, width = grid.shape  # Get grid dimensions
    visited = set() # Set of tuples to track visited transactions

    # Iterate over all agents
    for agent_id, (location, money, win, transactions, tax_paid, \
                   tax_rec, charity, poor, rich, moved_tracker) in money_of_agent.items():
        m, n = location
        neighbors = []

        # Identify 8 neighbors (D2N8 model, Moore's neighborhood)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    # Skip the current agent
                    continue
                # Calculate new coordinates
                m_new, n_new = (m + i) % height, (n + j) % width
                if grid[m_new, n_new] == 1:  # Check if a neighbor is present
                    for neighbor_id, (neighbor_loc, neighbor_money, n_win, \
                                      n_transactions, n_tax_paid, n_tax_received, n_charity, \
                                      n_poor, n_rich, n_moved_tracker) in money_of_agent.items():
                        if neighbor_loc == [m_new, n_new]:
                            # Add neighbor to the list
                            neighbors.append((neighbor_id, neighbor_loc, neighbor_money))
                            break

        # Perform transactions with neighbors
        for neighbor_id, neighbor_loc, neighbor_money in neighbors:
            if (agent_id, neighbor_id) in visited or (neighbor_id, agent_id) in visited:
                continue  # Avoid duplicate transactions
            visited.add((agent_id, neighbor_id)) # Mark transaction as visited

            R = random.random()  # random probability
            # transaction logic for the four cases
            if money == 0 and neighbor_money == 0:  # case 1: both agents have no money
                continue
            elif money == 0 and neighbor_money > 0:  # case 2: agent Ai has no money
                if R < p_t:  # agent Ai wins money
                    money_of_agent[agent_id][1] += delta_m
                    money_of_agent[agent_id][3].append(delta_m) # log transaction
                    money_of_agent[neighbor_id][1] -= delta_m
                    money_of_agent[neighbor_id][3].append(-delta_m) # log transaction
                    money_of_agent[agent_id][2] = True
                    money_of_agent[neighbor_id][2] = False
                    total_transaction += delta_m
                    transaction_count += 1
                    transaction_amounts.append(delta_m)
                    
            elif money >= neighbor_money > 0:  # case 3: agent Ai has more money
                if R < (p_t / 2 + p_i):  # case 3.1: agent Ai wins money
                    money_of_agent[agent_id][1] += delta_m
                    money_of_agent[agent_id][3].append(delta_m)
                    money_of_agent[neighbor_id][1] -= delta_m
                    money_of_agent[neighbor_id][3].append(-delta_m)
                    money_of_agent[agent_id][2] = True
                    money_of_agent[neighbor_id][2] = False
                    total_transaction += delta_m
                    transaction_count += 1
                    transaction_amounts.append(delta_m)
                    

                elif R < (p_t/2 -p_i):  # case 3.2: agent Ai loses money
                    money_of_agent[agent_id][1] -= delta_m
                    money_of_agent[agent_id][3].append(-delta_m)
                    money_of_agent[neighbor_id][1] += delta_m
                    money_of_agent[neighbor_id][3].append(delta_m)
                    money_of_agent[neighbor_id][2] = True
                    money_of_agent[agent_id][2] = False
                    total_transaction += delta_m
                    transaction_count += 1
                    transaction_amounts.append(delta_m)
                    
            elif neighbor_money > money > 0:  # case 4
                if R < (p_t / 2 - p_i):  # case 4.1: agent Ai wins money
                    money_of_agent[agent_id][1] += delta_m
                    money_of_agent[agent_id][3].append(delta_m)
                    money_of_agent[neighbor_id][1] -= delta_m
                    money_of_agent[neighbor_id][3].append(-delta_m)
                    money_of_agent[agent_id][2] = True
                    money_of_agent[neighbor_id][2] = False
                    total_transaction += delta_m
                    transaction_count += 1
                    transaction_amounts.append(delta_m)
                    
                elif R < (p_t/2 + p_i):  # case 4.2: agent Ai loses money
                    money_of_agent[agent_id][1] -= delta_m
                    money_of_agent[agent_id][3].append(-delta_m)
                    money_of_agent[neighbor_id][1] += delta_m
                    money_of_agent[neighbor_id][3].append(delta_m)
                    money_of_agent[neighbor_id][2] = True
                    money_of_agent[agent_id][2] = False
                    total_transaction += delta_m
                    transaction_count += 1
                    transaction_amounts.append(delta_m)
                    

            # ensure money remains non-negative
            money_of_agent[agent_id][1] = max(0, money_of_agent[agent_id][1])
            money_of_agent[neighbor_id][1] = max(0, money_of_agent[neighbor_id][1])

            assert money_of_agent[agent_id][1] >= 0, f"Agent {agent_id} has negative money!"
            assert money_of_agent[neighbor_id][1] >= 0, f"Agent {neighbor_id} has negative money!"

    return money_of_agent,total_transaction,transaction_count,transaction_amounts

def tax(money_of_agent, delta_m, psi_max, omega, m_tax):
    """
    Apply income tax and redistribute the revenue equally among agents.

    Parameters:
        money_of_agent: Dictionary containing agent locations and money values.
        delta_m: Fixed transaction unit of money.
        psi_max: Maximum tax rate.
        omega: Empirical parameter for tax rate calculation.
        m_tax: Critical money threshold for taxation.

    Returns:
        Updated money_of_agent dictionary.
    """
    total_tax_revenue = 0  # initialize total tax revenue
    m_max = max(agent[1] for agent in money_of_agent.values())  # find the max money among agents

    winner_agents = [agent_id for agent_id, (location, money, win, transaction, tax_paid, tax_rec, charity, poor, rich,moved_tracker) in money_of_agent.items() if win == True]
    # calculate tax liability and collect tax revenue
    for agent_id in winner_agents:
        money = money_of_agent[agent_id][1]

        if money > m_tax:
            psi_i = ((money / m_max) ** omega) * psi_max  # calculate average tax rate 
            tax_liability = psi_i * delta_m  # calculate tax liability
        
            money_of_agent[agent_id][1] -= tax_liability  # deduct tax liability from the agent's money
            money_of_agent[agent_id][4] = tax_liability  # log tax amount
            total_tax_revenue += tax_liability  # add to total tax revenue


    # Redistribute tax revenue equally
    redistribution = total_tax_revenue / len(money_of_agent)

    for agent_id in money_of_agent:
        money_of_agent[agent_id][1] += redistribution  # add redistributed amount to each agent's money
        money_of_agent[agent_id][5] = redistribution  # log tax amount received

    # ensure no agent has negative money
    for agent_id in money_of_agent:
        money_of_agent[agent_id][1] = max(0, money_of_agent[agent_id][1])

    return money_of_agent

def charity(money_of_agent, m_r, m_p, m_c, charity_probability):
    """
    Collect charity donations and redistribute the revenue equally among agents.

    Parameters:
        money_of_agent: Dictionary containing agent locations and money values.
        m_r: Amount of money needed to be considered rich.
        m_p: Amount of money needed to be considered poor.
        m_c: Fixed amount of charity contribution.

    Returns:
        Updated money_of_agent dictionary.
    """
    total_charity_revenue = 0  # initialize total charity revenue
    R = random.random()  # random probability
    
    # Define poor agents (money < m_p) and rich agents (money > m_r)
    poor_agents = [agent_id for agent_id, (location, money, win, transactions, \
                                           tax_paid, tax_rec, charity, poor, rich, \
                                           moved_tracker) in money_of_agent.items() if money < m_p]
    rich_agents = [agent_id for agent_id, (location, money, win, transactions, \
                                           tax_paid, tax_rec, charity, poor, rich, \
                                           moved_tracker) in money_of_agent.items() if money > m_r \
                                           and win == True]
    
    # calculate charity contributions and collect charity revenue
    if len(rich_agents) > 0 and len(poor_agents) > 0:
        for agent_id in rich_agents:
            # Set the agent as rich
            money_of_agent[agent_id][8] = True

            if R < charity_probability:  # agent is rich and has won and donates
                money_of_agent[agent_id][1] -= m_c  # deduct charity contribution from the agent's money
                money_of_agent[agent_id][6] = -m_c  # log charity amount
                total_charity_revenue += m_c  # add to total charity pool
        
        if total_charity_revenue > 0:
            # Redistribute charity revenue equally among poor agents
            charity_redistribution = total_charity_revenue / len(poor_agents)

            for agent_id in poor_agents:
                money_of_agent[agent_id][7] = True
                money_of_agent[agent_id][1] += charity_redistribution
                money_of_agent[agent_id][6] = charity_redistribution

    # ensure no agent has negative money
    for agent_id in money_of_agent:
        money_of_agent[agent_id][1] = max(0, money_of_agent[agent_id][1])

    return money_of_agent


def time_step_randwalk(grid, probablility_move, showmovements=False):
    """Perform a time step where the values move randomly without merging.
    
    Parameters:
        grid (np.array): The grid representing agent positions.
        probablility_move (float): The probability of an agent moving.
        showmovements (bool): Whether to print the movements.
        
    Returns:
        np.array: The updated grid after the time step.
    """
    height, width = grid.shape
    new_grid = np.zeros_like(grid)  # initialize a new grid for the updated state
    
    movements = []  # list to track movements
    occupied_count = 0  # counter for money_of_agent
    visited = set()  # set to track cells that are already occupied in the new grid

    # First pass: mark all existing positions as visited
    for m in range(height):
        for n in range(width):
            if grid[m, n] == 1:
                visited.add((m, n))
    
    # Loop over all cells
    for m in range(height):
        for n in range(width):
            if grid[m, n] == 1:  # if there is a person in a cell
                occupied_count += 1  # increment occupied cell count
                loc = [m, n]
                agent_id = next((key for key, value in money_of_agent.items() if value[0] == loc), None)

                # random movement
                if random.random() < probablility_move:  # move with some probability
                    m_new, n_new = globals()['move'](m, n, height, width)  # explicitly use the global move() function
                    money_of_agent[agent_id][9] = True  # log movement

                    # add the person to the grid if the cell is empty and not visited
                    if new_grid[m_new, n_new] == 0 and (m_new, n_new) not in visited:
                        new_loc = [m_new, n_new]
                        new_grid[m_new, n_new] = 1
                        visited.add((m_new, n_new))  # mark the cell as visited
                        movements.append(((m, n), (m_new, n_new)))  # log movement
                        money_of_agent[agent_id][0] = new_loc  # update the agent's location
                    else:
                        new_grid[m, n] = 1  # stay in place if target is occupied
                        movements.append(((m, n), (m, n)))  # log no movement due to target occupied
                else:
                    new_grid[m, n] = 1  # stay in place
                    movements.append(((m, n), (m, n)))  # log no movement
    
    # Print the number of occupied cells and movements
    if showmovements:
        print(f"money_of_agent: {occupied_count}")
        print("Movements:")
        for move in movements:
            print(f"{move[0]} -> {move[1]}")

    return new_grid

def get_shades_of_green(n):
    """Generate n shades of green to represent the money distribution."""
    start = np.array([144, 238, 144]) / 255  # lightgreen
    end = np.array([0, 100, 0]) / 255  # darkgreen
    return [(start + (end - start) * i / (n - 1)).tolist() for i in range(n)]

def animate_CA(initial_grid, steps,showmovements,show_animation, interval, probablility_move, m_0, delta_m, p_t, p_i, psi_max, omega, m_r, m_p, m_c, m_tax, charity_probability, run_num, param_list):
    """
    Animate the cellular automata, updating time step and cell values.
    
    Parameters:
        initial_grid (np.array): The initial grid representing agent positions.
        steps (int): The number of time steps to simulate.
        showmovements (bool): Whether to print the movements.
        show_animation (bool): Whether to animate the simulation.
        interval (int): The interval between frames in the animation.
        probablility_move (float): The probability of an agent moving.
        m_0 (int): The initial amount of money each agent has.
        delta_m (float): The amount exchanged in a transaction.
        p_t (float): The probability of transaction.
        p_i (float): The inequality parameter.
        psi_max (float): The maximum tax rate.
        omega (float): The empirical parameter for tax calculation.
        m_r (int): The rich threshold.
        m_p (int): The poverty threshold.
        m_c (float): The charity donation amount.
        m_tax (float): The critical threshold for taxation.
        charity_probability (float): The probability of donating to charity.
        run_num (int): The run number.
        param_list (str): The list of parameters used in the simulation.
        
    Returns:
        list: A list of averages.
        list: A list of total transactions per step.
        list: A list of total transaction counts.
        list: A list of all transaction amounts.
    """
    averages = []
    total_transactions_per_step = []  
    total_transaction_counts = []
    all_transaction_amounts = []  

    global money_of_agent  

    if show_animation:
        # colourmap
        cmap = plt.cm.coolwarm  # blue to red
        cmap.set_bad(color='white')  # empty spaces white
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_xticks(np.arange(-0.5, initial_grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, initial_grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)  # grid lines
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Fortune Automata")
        grid = np.copy(initial_grid)
        new_grid = np.full_like(grid, np.nan, dtype=float) 
        
        # determine wealth range for color scale
        min_wealth = 0  # the lowest possible wealth
        max_wealth = max(agent[1] for agent in money_of_agent.values()) * 2  # maximum wealth growth over time
        matrix = ax.matshow(new_grid, cmap=cmap, vmin=min_wealth, vmax=max_wealth)  # from poor to rich values
        def update(frames):
            global money_of_agent
            nonlocal grid
            grid = time_step_randwalk(grid, probablility_move, showmovements)
            # economic transaction and track interesting variables
            money_of_agent, total_transaction, transaction_count, transaction_amounts = \
                economic_transaction(grid, money_of_agent, delta_m, p_t, p_i)
            total_transactions_per_step.append(total_transaction)  
            total_transaction_counts.append(transaction_count)
            all_transaction_amounts.extend(transaction_amounts)
            
            # Apply tax and charity
            money_of_agent = tax(money_of_agent, delta_m, psi_max, omega, m_tax)
            money_of_agent = charity(money_of_agent, m_r, m_p, m_c, charity_probability)
            
            # Update the grid
            new_grid[:] = np.nan  # white background
            for agent_id, (location, money, win, transactions, tax_paid, tax_rec, \
                           charity_rec, poor, rich, moved_tracker) in money_of_agent.items():
                m, n = location
                new_grid[m, n] = money 
            matrix.set_array(new_grid)
            return [matrix]
        
        # Create the animation
        ani = FuncAnimation(fig, update, frames=steps - 1, interval=interval, blit=False, repeat=False)
        plt.colorbar(matrix, ax=ax, label="Wealth")  # color scale for wealth
        plt.show()

    else:
        # Run the cellular automata in the background, updating time step and cell values.
        grid = np.copy(initial_grid)

        # Lists to collect data
        total_agents_list = []

        # Run the simulation for the specified steps
        for step in range(steps):
            # Perform a time-step
            grid = time_step_randwalk(grid, probablility_move, showmovements=False)

            #eceonomic transaction and track interesting variables
            money_of_agent,total_transaction,transaction_count,transaction_amounts = economic_transaction(grid, money_of_agent, delta_m, p_t, p_i)
            total_transactions_per_step.append(total_transaction)  
            total_transaction_counts.append(transaction_count)
            all_transaction_amounts.extend(transaction_amounts)

            money_of_agent = tax(money_of_agent, delta_m, psi_max, omega, m_tax)
            money_of_agent = charity(money_of_agent, m_r, m_p, m_c, charity_probability)

            record_transaction_data(step) # Record transaction data at this step

            # Collect data at this step if needed
            average_money = np.mean([money for _, (_, money, _, _, _, _, _, _, _, _) in money_of_agent.items()])  
            total_agents = len(money_of_agent)
            total_agents_list.append(total_agents)

    # Save recorded data to CSV
    filepath = f"data/data_unsorted/transaction_log_run_{run_num}.csv"

    with open(filepath, "w") as file:
        pd.DataFrame(data_log).to_csv(filepath, index=False) # write df

    # append param_list to the beginning of the file
    with open(filepath, 'a') as file:
        file.write("\n" + param_list)

    return averages, total_transactions_per_step, total_transaction_counts, all_transaction_amounts

if __name__ == '__main__':
    # input parameters
    height = 20
    width = 20
    probablility_move = 0.8     # chance of movement of indiviudual
    steps = 10000               # timesteps
    density = 0.2

    # animation and logging
    showmovements = False
    show_animation = False

    m0 = 100                    # starting money   
    delta_m = m0/100            # amount exchanged in transaction
    p_t = 0.7                   # probability of transaction
    p_i = 0.0574                # inequality parameter

    m_tax = m0 / 2              # critical threshold for taxation
    psi_max = 0.5               # maximum tax rate (adjustable)
    omega = 1.0                 # empirical parameter for tax calculation

    m_p = 0.7 * m0              # poverty threshold (can receive charity)
    m_r = 1.5 * m0              # rich threshold (can give to charity)
    m_c = delta_m * 0.5         # charity donation amount
    charity_probability = 0.5   # probability of donating to charity

    run_num = 1

    param_list = f"m0: {m0}; delta_m: {delta_m}; p_t: {p_t}; p_i: {p_i}; m_tax: {m_tax}; psi_max: {psi_max}; omega: {omega}; " \
                 f"m_p: {m_p}; m_r: {m_r}; m_c: {m_c}; charity_probability: {charity_probability}"

    # Set up + initialize the grid
    grid = initialize_grid(height, width, density, m0)

    # start animation, any data of interest can be returned from animate_CA
    averages, total_money_transacted_per_timestep, total_transaction_counts, all_transaction_amounts = \
        animate_CA(grid, 
                   steps, 
                   showmovements, 
                   show_animation, 
                   interval=100, 
                   probablility_move=probablility_move, 
                   m_0=m0, 
                   delta_m=delta_m, 
                   p_t=p_t, 
                   p_i=p_i, 
                   psi_max=psi_max, 
                   omega=omega, 
                   m_r=m_r, 
                   m_p=m_p, 
                   m_c=m_c, 
                   m_tax=m_tax, 
                   charity_probability=charity_probability, 
                   run_num=run_num, 
                   param_list=param_list)

    # plot distribution of total money transacted each timestep
    # uncomment to plot
    # plot_transaction_distribution(total_money_transacted_per_timestep)
    # plot_transaction_counts(total_transaction_counts)




