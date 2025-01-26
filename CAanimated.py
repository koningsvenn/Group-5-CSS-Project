import numpy as np 
import numpy.random as random
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd



# track money and position of each agent
money_of_agent = {}

"""set up the grid"""
def initialize_grid(height, width, fall_heigth, density, m0):
    """Create a height x width grid with zeros representing empty cells or integers 
    to represent person size"""

    global money_of_agent
    money_of_agent.clear()
    #empty grid
    grid = np.zeros((height, width))  
    no_of_agents = int(height * width * density) 
    
    #select random coordinates for each person
    agents = 0
    while agents < no_of_agents:
        n = random.randint(0, width - 1)
        m = random.randint(0, height - 1)

        if grid[m, n] == 0:
            grid[m, n] = 1 
            # add agent info
            location = [m, n]
            money = m0
            money_of_agent[agents] = [location, money, False]
            # money_of_agent[(m, n)] = 2
        agents += 1
    print(money_of_agent)
    return grid 


def move(m, n, height, width):
    """Move the persons randomly."""
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
    Args:
        grid: The grid representing agent positions.
        money_of_agent: A dictionary mapping agent IDs to their location and money.
        delta_m: Fixed amount of money transferred in a transaction.
        p_t: Probability of a transaction occurring between neighbors.
        p_i: Inequality parameter to control transaction probabilities.
    Returns:
        Updated money_of_agent dictionary.
    """
    total_transaction = 0
    transaction_count = 0 

    height, width = grid.shape
    visited = set()

    # iterate over all agents
    for agent_id, (location, money, win) in money_of_agent.items():
        m, n = location
        neighbors = []

        # identify 8 neighbors (D2N8 model)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                m_new, n_new = (m + i) % height, (n + j) % width
                if grid[m_new, n_new] == 1:  # check if a neighbor is present
                    for neighbor_id, (neighbor_loc, neighbor_money, win) in money_of_agent.items():
                        if neighbor_loc == [m_new, n_new]:
                            neighbors.append((neighbor_id, neighbor_loc, neighbor_money))
                            break

        # perform transactions with neighbors
        for neighbor_id, neighbor_loc, neighbor_money in neighbors:
            if (agent_id, neighbor_id) in visited or (neighbor_id, agent_id) in visited:
                continue  # avoid duplicate transactions
            visited.add((agent_id, neighbor_id))

            R = random.random()  # random probability
            # transaction logic for the four cases
            if money == 0 and neighbor_money == 0:  # case 1
                continue
            elif money == 0 and neighbor_money > 0:  # case 1
                if R < p_t:  # agent Ai wins money
                    money_of_agent[agent_id][1] += delta_m
                    money_of_agent[neighbor_id][1] -= delta_m
                    money_of_agent[agent_id][2] = True
                    total_transaction += delta_m
                    transaction_count += 1
            elif money >= neighbor_money > 0:  # case 3
                if R < (p_t / 2 + p_i):  # agent Ai wins money
                    money_of_agent[agent_id][1] += delta_m
                    money_of_agent[neighbor_id][1] -= delta_m
                    money_of_agent[agent_id][2] = True
                    total_transaction += delta_m
                    transaction_count += 1
                elif R < (p_t/2 -p_i):  # agent Ai loses money
                    money_of_agent[agent_id][1] -= delta_m
                    money_of_agent[neighbor_id][1] += delta_m
                    money_of_agent[neighbor_id][2] = True
                    total_transaction += delta_m
                    transaction_count += 1
            elif neighbor_money > money > 0:  # case 4
                if R < (p_t / 2 - p_i):  # agent Ai wins money
                    money_of_agent[agent_id][1] += delta_m
                    money_of_agent[neighbor_id][1] -= delta_m
                    money_of_agent[agent_id][2] = True
                    total_transaction += delta_m
                    transaction_count += 1
                elif R < (p_t/2 + p_i):  # agent Ai loses money
                    money_of_agent[agent_id][1] -= delta_m
                    money_of_agent[neighbor_id][1] += delta_m
                    money_of_agent[neighbor_id][2] = True
                    total_transaction += delta_m
                    transaction_count += 1

            # ensure money remains non-negative
            money_of_agent[agent_id][1] = max(0, money_of_agent[agent_id][1])
            money_of_agent[neighbor_id][1] = max(0, money_of_agent[neighbor_id][1])

    return money_of_agent,total_transaction,transaction_count

def tax(money_of_agent, delta_m, psi_max, omega, m_tax):
    """
    Apply income tax and redistribute the revenue equally among agents.

    Args:
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

    winner_agents = [agent_id for agent_id, (location, money, win) in money_of_agent.items() if win == True]
    # calculate tax liability and collect tax revenue
    for agent_id in winner_agents:
        money = money_of_agent[agent_id][1]
        #update win to false
        money_of_agent[agent_id][2] = False
        if money > m_tax:
            psi_i = ((money / m_max) ** omega) * psi_max  # calculate average tax rate 
            tax_liability = psi_i * delta_m  # calculate 
            money_of_agent[agent_id][1] -= tax_liability  # deduct tax liability from the agent's money
            total_tax_revenue += tax_liability  # add to total tax revenue


    # redistribute tax revenue equally
    redistribution = total_tax_revenue / len(money_of_agent)

    for agent_id in money_of_agent:
        money_of_agent[agent_id][1] += redistribution  # add redistributed amount to each agent's money

    # ensure no agent has negative money
    for agent_id in money_of_agent:
        money_of_agent[agent_id][1] = max(0, money_of_agent[agent_id][1])

    return money_of_agent

def charity(money_of_agent, m_r, m_p, m_c, charity_probability):
    """
    Collect charity donations and redistribute the revenue equally among agents.

    Args:
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
    poor_agents = [agent_id for agent_id, (location, money, win) in money_of_agent.items() if money < m_p]
    rich_agents = [agent_id for agent_id, (location, money, win) in money_of_agent.items() if money > m_r]

    # calculate charity contributions and collect charity revenue
    if len(rich_agents) > 0 and len(poor_agents) > 0:
        for agent_id in rich_agents:
            if R < charity_probability:  # agent is rich and donates
                money_of_agent[agent_id][1] -= m_c  # deduct charity contribution from the agent's money
                total_charity_revenue += m_c  # add to total charity pool
        
        if total_charity_revenue > 0:
            charity_redistribution = total_charity_revenue / len(poor_agents)
            for agent_id in poor_agents:
                money_of_agent[agent_id][1] += charity_redistribution

    # ensure no agent has negative money
    for agent_id in money_of_agent:
        money_of_agent[agent_id][1] = max(0, money_of_agent[agent_id][1])

    return money_of_agent

def time_step_randwalk(grid, probablility_move,showmovements):
    """Perform a time step where the values move randomly without merging."""
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
    # loop over all cells
    for m in range(height):
        for n in range(width):
            if grid[m, n] == 1:  # if there is a person in a cell
                occupied_count += 1  # increment occupied cell count
                loc = [m, n]
                agent_id = next((key for key, value in money_of_agent.items() if value[0] == loc), None)

                # random movement
                if random.random() < probablility_move:  # move with some probability
                    m_new, n_new = globals()['move'](m, n, height, width)  # explicitly use the global move function


                    # add the person to the grid if the cell is empty and not visited
                    if new_grid[m_new, n_new] == 0 and (m_new, n_new) not in visited:
                        new_loc = [m_new, n_new]
                        new_grid[m_new, n_new] = 1
                        visited.add((m_new, n_new))  # mark the cell as visited
                        movements.append(((m, n), (m_new, n_new)))  # log movement
                        # TODO: implement transactions and change later
                        money_of_agent[agent_id][0] = new_loc  # update the agent's location
                    else:
                        new_grid[m, n] = 1  # stay in place if target is occupied
                        movements.append(((m, n), (m, n)))  # log no movement due to target occupied
                else:
                    new_grid[m, n] = 1  # stay in place
                    movements.append(((m, n), (m, n)))  # log no movement
    
    
    
    if showmovements:
        print(f"money_of_agent: {occupied_count}")
        print("Movements:")
        for move in movements:
            print(f"{move[0]} -> {move[1]}")

    return new_grid

"""Assign a color based on wealth"""
def get_shades_of_green(n):
    """Generate n shades of green."""
    start = np.array([144, 238, 144]) / 255  # lightgreen
    end = np.array([0, 100, 0]) / 255  # darkgreen
    return [(start + (end - start) * i / (n - 1)).tolist() for i in range(n)]

"""Animate the CA to visualize what is happening"""
def animate_CA(initial_grid, steps,showmovements,show_animation, interval, probablility_move, delta_m, p_t, p_i, psi_max, omega, mr, mp, mc, pc,m_tax):
    averages = []
    total_transactions_per_step = []  
    total_transaction_counts = []
    global money_of_agent  # Access the global variable

    if show_animation:
        """Animate the cellular automata, updating time step and cell values."""
        #set up colors and figure for the animation
        colors = get_shades_of_green(20) 
        cmap = LinearSegmentedColormap.from_list("custom_blue", colors, N=256)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_xticks(np.arange(-.5, initial_grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, initial_grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])

        grid = np.copy(initial_grid)
        matrix = ax.matshow(grid, cmap=cmap)

        # create text objects for each cell
        text = [[ax.text(j, i, '', ha='center', va='center', color='black') for j in range(grid.shape[1])] for i in range(grid.shape[0])]

        #lists to collect data
        total_agents_list = []
        
        def update(frames):
            global money_of_agent  # access the global variable
            nonlocal grid

            # Perform a time-step
            grid = time_step_randwalk(grid, probablility_move, showmovements)

            #transact and track interesting variables
            money_of_agent,total_transaction,transaction_count = economic_transaction(grid, money_of_agent, delta_m, p_t, p_i)
            total_transactions_per_step.append(total_transaction)  
            total_transaction_counts.append(transaction_count)

            money_of_agent = tax(money_of_agent, delta_m, psi_max, omega, m_tax)

            money_of_agent = charity(money_of_agent, mr, mp, mc, charity_probability)

            # print(money_of_agent)

            # update the grid display
            matrix.set_array(grid)

            # clear all text from the grid initially
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    text[i][j].set_text('')  # clear previous text

        # display money values for agents
        for agent_id, (location, money, win) in money_of_agent.items():
            m, n = location  # agent's location in the grid
            text[m][n].set_text(f'{int(money)}')  # display agent's money
            text[m][n].set_color('white' if money > 2 else 'black')  # adjust text color for better contrast
            # display money values for agents
            for agent_id, (location, money) in money_of_agent.items():
                m, n = location  # agent's location in the grid
                text[m][n].set_text(f'{int(money)}')  # display agent's money
                text[m][n].set_color('white' if money > 2 else 'black')  # adjust text color for better contrast

            # update title and return all drawable elements
            ax.set_title(f"Economy Automata")
            return [matrix] + [txt for row in text for txt in row]


        ani = FuncAnimation(fig, update, frames=steps-1, interval=interval, blit=False, repeat=False) #average step -1 because the first frame is a step 
        plt.show()
    else:
        """Run the cellular automata in the background, updating time step and cell values."""

        grid = np.copy(initial_grid)

        # Lists to collect data
        total_agents_list = []

        # Run the simulation for the specified steps
        for step in range(steps):
            # Perform a time-step
            grid = time_step_randwalk(grid, probablility_move, showmovements=False)

            #eceonomic transaction and track interesting variables
            money_of_agent,total_transaction,transaction_count = economic_transaction(grid, money_of_agent, delta_m, p_t, p_i)
            total_transactions_per_step.append(total_transaction)  
            total_transaction_counts.append(transaction_count)

            money_of_agent = tax(money_of_agent, delta_m, psi_max, omega, m_tax)
            money_of_agent = charity(money_of_agent, mr, mp, mc, charity_probability)

            # Collect data at this step if needed
            average_money = np.mean([money for _, (_, money, _) in money_of_agent.items()])  
            total_agents = len(money_of_agent)
            total_agents_list.append(total_agents)


    return averages,total_transactions_per_step,total_transaction_counts #,return any data of interest from this function



if __name__ == '__main__':
    """input parameters"""
    height = 20
    width = 20
    probablility_move = 0.8  # chance of movement of indiviudual
    steps = 200  # timesteps
    density = 0.2

    showmovements = False
    show_animation = False

    m0 = 100
    delta_m = m0/100
    p_t = 0.7
    p_i = 0.0574
    mr = 1.0
    mp = 1.0
    mc = 1.0
    pc = 1.0

    m_tax = m0 / 2  # critical threshold for taxation
    psi_max = 0.5   # maximum tax rate (adjustable)
    omega = 1.0     # empirical parameter for tax calculation

    m_p = 0.7 * m0 # poverty line (if agent has less than this level of income they are eligible to receive donations)
    m_r = 1.5 * m0 # rich line (if agent has more than this level of income they are eligible to give donations)
    m_c = delta_m * 0.5 # charity donation amount
    charity_probability = 0.5  # probability of donating to charity

    """set up grid"""
    grid = initialize_grid(height, width, 0, density,m0)  # init. the grid

    """start animation, any data of interest can be returned from animate_CA"""
    averages,total_money_transacted_per_timestep,total_transaction_counts = animate_CA(grid, steps,showmovements,show_animation, interval=100, probablility_move=probablility_move, 
                          delta_m=delta_m, p_t=p_t, p_i=p_i, psi_max=psi_max, omega=omega, mr=mr, mp=mp, mc=mc, pc=pc,m_tax=m_tax)

    
    
    print(total_transaction_counts)
    print(total_money_transacted_per_timestep)



