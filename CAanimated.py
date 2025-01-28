import numpy as np 
import numpy.random as random
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from plot_utils import plot_transaction_distribution,plot_transaction_counts



# track money and position of each agent
money_of_agent = {}
data_log = []  # To store the transaction details per timestep

"""set up the grid"""
def initialize_grid(height, width, density, m0):
    """Create a height x width grid with zeros representing empty cells or integers 
    to represent person size"""

    global money_of_agent
    money_of_agent.clear()
    # empty grid
    grid = np.zeros((height, width))  
    no_of_agents = int(height * width * density) 
    
    # select random coordinates for each person
    agents = 0
    while agents < no_of_agents:
        n = random.randint(0, width - 1)
        m = random.randint(0, height - 1)

        if grid[m, n] == 0:
            grid[m, n] = 1 
            # add agent info
            location = [m, n]
            money = m0
            transactions = []
            tax_amt_paid = 0
            tax_amt_received = 0
            charity_amt = 0
            poor = False
            rich = False
            money_of_agent[agents] = [location, money, False, transactions, tax_amt_paid, tax_amt_received, charity_amt, poor, rich]
            # money_of_agent[(m, n)] = 2
        agents += 1
    print(money_of_agent)
    return grid 


def record_transaction_data(timestep):
    global data_log

    for agent_id, (location, money, moved, transactions, tax_amt_paid, tax_amt_received, charity_amt, poor, rich) in money_of_agent.items():
        num_neighbors = len([neighbor_id for neighbor_id, (neighbor_loc, _, _, _, _, _, _, _, _) in money_of_agent.items() if np.linalg.norm(np.array(location) - np.array(neighbor_loc)) <= np.sqrt(2)])
        transacted = any(t != 0 for t in transactions)
        location_string = f"{location[0]},{location[1]}"

        data_log.append({
            "ID": agent_id,
            "Time step": timestep,
            "Position": location_string,
            "Number of neighbors": num_neighbors,
            "Moved": moved, # not updated, TODO: update this
            "Transacted": transacted,
            "Poor": poor,
            "Rich": rich,
            "Amount of income gained/lost": sum(transactions),
            "Amount of tax paid": tax_amt_paid,
            "Amount of tax received": tax_amt_received,
            "Amount of charity given": charity_amt if charity_amt < 0 else 0,
            "Amount of charity received": charity_amt if charity_amt > 0 else 0
        })

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
    transaction_amounts = []  # List for tracking transaction amounts

    height, width = grid.shape
    visited = set()

    # calculate total money before transactions
    total_money_before = sum(agent[1] for agent in money_of_agent.values())

    # iterate over all agents
    for agent_id, (location, money, win, transactions, tax_paid, tax_rec, charity, poor, rich) in money_of_agent.items():
        m, n = location
        neighbors = []

        # identify 8 neighbors (D2N8 model)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                m_new, n_new = (m + i) % height, (n + j) % width
                if grid[m_new, n_new] == 1:  # check if a neighbor is present
                    for neighbor_id, (neighbor_loc, neighbor_money, n_win, n_transactions, n_tax_paid, n_tax_received, n_charity, n_poor, n_rich) in money_of_agent.items():
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
                    money_of_agent[agent_id][3].append(delta_m) # log transaction
                    money_of_agent[neighbor_id][1] -= delta_m
                    money_of_agent[neighbor_id][3].append(-delta_m) # log transaction
                    money_of_agent[agent_id][2] = True
                    total_transaction += delta_m
                    transaction_count += 1
                    transaction_amounts.append(delta_m)
                    
            elif money >= neighbor_money > 0:  # case 3
                if R < (p_t / 2 + p_i):  # agent Ai wins money
                    money_of_agent[agent_id][1] += delta_m
                    money_of_agent[agent_id][3].append(delta_m)
                    money_of_agent[neighbor_id][1] -= delta_m
                    money_of_agent[neighbor_id][3].append(-delta_m)
                    money_of_agent[agent_id][2] = True
                    total_transaction += delta_m
                    transaction_count += 1
                    transaction_amounts.append(delta_m)
                    

                elif R < (p_t/2 -p_i):  # agent Ai loses money
                    money_of_agent[agent_id][1] -= delta_m
                    money_of_agent[agent_id][3].append(-delta_m)
                    money_of_agent[neighbor_id][1] += delta_m
                    money_of_agent[neighbor_id][3].append(delta_m)
                    money_of_agent[neighbor_id][2] = True
                    total_transaction += delta_m
                    transaction_count += 1
                    transaction_amounts.append(delta_m)
                    
            elif neighbor_money > money > 0:  # case 4
                if R < (p_t / 2 - p_i):  # agent Ai wins money
                    money_of_agent[agent_id][1] += delta_m
                    money_of_agent[agent_id][3].append(delta_m)
                    money_of_agent[neighbor_id][1] -= delta_m
                    money_of_agent[neighbor_id][3].append(-delta_m)
                    money_of_agent[agent_id][2] = True
                    total_transaction += delta_m
                    transaction_count += 1
                    transaction_amounts.append(delta_m)
                    
                elif R < (p_t/2 + p_i):  # agent Ai loses money
                    money_of_agent[agent_id][1] -= delta_m
                    money_of_agent[agent_id][3].append(-delta_m)
                    money_of_agent[neighbor_id][1] += delta_m
                    money_of_agent[neighbor_id][3].append(delta_m)
                    money_of_agent[neighbor_id][2] = True
                    total_transaction += delta_m
                    transaction_count += 1
                    transaction_amounts.append(delta_m)
                    

            # ensure money remains non-negative
            money_of_agent[agent_id][1] = max(0, money_of_agent[agent_id][1])
            money_of_agent[neighbor_id][1] = max(0, money_of_agent[neighbor_id][1])

    # calculate total money after transactions
    total_money_after = sum(agent[1] for agent in money_of_agent.values())

    # check if total money is conserved
    # assert np.floor(total_money_before) == np.floor(total_money_after), \
    # f"Total money before ({total_money_before:.2f}) and after ({total_money_after:.2f}) transactions does not match!"

    return money_of_agent,total_transaction,transaction_count,transaction_amounts

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

    winner_agents = [agent_id for agent_id, (location, money, win, transaction, tax_paid, tax_rec, charity, poor, rich) in money_of_agent.items() if win == True]
    # calculate tax liability and collect tax revenue
    for agent_id in winner_agents:
        money = money_of_agent[agent_id][1]
        #update win to false
        money_of_agent[agent_id][2] = False
        if money > m_tax:
            psi_i = ((money / m_max) ** omega) * psi_max  # calculate average tax rate 
            tax_liability = psi_i * delta_m  # calculate 
            money_of_agent[agent_id][1] -= tax_liability  # deduct tax liability from the agent's money
            money_of_agent[agent_id][4] += tax_liability  # log tax amount
            total_tax_revenue += tax_liability  # add to total tax revenue


    # redistribute tax revenue equally
    redistribution = total_tax_revenue / len(money_of_agent)

    for agent_id in money_of_agent:
        money_of_agent[agent_id][1] += redistribution  # add redistributed amount to each agent's money
        money_of_agent[agent_id][5] += redistribution  # log tax amount received

    # ensure no agent has negative money
    for agent_id in money_of_agent:
        money_of_agent[agent_id][1] = max(0, money_of_agent[agent_id][1])

    # check if total tax revenue matches total distributed amount
    total_distributed = redistribution * len(money_of_agent)
    assert round(total_tax_revenue, 5) == round(total_distributed, 5), \
        f"Total tax collected ({total_tax_revenue:.5f}) does not match total distributed ({total_distributed:.5f})!"

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
    poor_agents = [agent_id for agent_id, (location, money, win, transactions, tax_paid, tax_rec, charity, poor, rich) in money_of_agent.items() if money < m_p]
    rich_agents = [agent_id for agent_id, (location, money, win, transactions, tax_paid, tax_rec, charity, poor, rich) in money_of_agent.items() if money > m_r]

    # calculate charity contributions and collect charity revenue
    if len(rich_agents) > 0 and len(poor_agents) > 0:
        for agent_id in rich_agents:
            money_of_agent[agent_id][8] = True
            if R < charity_probability:  # agent is rich and donates
                money_of_agent[agent_id][1] -= m_c  # deduct charity contribution from the agent's money
                money_of_agent[agent_id][6] += m_c  # log charity amount
                total_charity_revenue += m_c  # add to total charity pool
        
        if total_charity_revenue > 0:
            charity_redistribution = total_charity_revenue / len(poor_agents)
            for agent_id in poor_agents:
                money_of_agent[agent_id][7] = True
                money_of_agent[agent_id][1] += charity_redistribution
                money_of_agent[agent_id][6] += charity_redistribution

    # ensure no agent has negative money
    for agent_id in money_of_agent:
        money_of_agent[agent_id][1] = max(0, money_of_agent[agent_id][1])

    # check if total charity revenue matches total redistributed amount
    total_redistributed = total_charity_revenue if len(poor_agents) > 0 else 0
    # assert round(total_charity_revenue, 5) == round(total_redistributed, 5), \
    assert int(total_charity_revenue) == int(total_redistributed), \
            f"Total charity collected ({total_charity_revenue:.5f}) does not match total redistributed ({total_redistributed:.5f})!"

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
def animate_CA(initial_grid, steps,showmovements,show_animation, interval, probablility_move, delta_m, p_t, p_i, psi_max, omega, mr, mp, mc, pc,m_tax, charity_probability, run_num, param_list):
    averages = []
    total_transactions_per_step = []  
    total_transaction_counts = []
    all_transaction_amounts = []  

    global money_of_agent  

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
            global money_of_agent  # Access the global variable
            nonlocal grid

            # Perform a time-step
            grid = time_step_randwalk(grid, probablility_move, showmovements)

            # Transact and track interesting variables
            money_of_agent, total_transaction, transaction_count,transaction_amounts = economic_transaction(grid, money_of_agent, delta_m, p_t, p_i)
            total_transactions_per_step.append(total_transaction)  
            total_transaction_counts.append(transaction_count)
            all_transaction_amounts.extend(transaction_amounts)
            
            
            money_of_agent = tax(money_of_agent, delta_m, psi_max, omega, m_tax)
            money_of_agent = charity(money_of_agent, mr, mp, mc, charity_probability)
            
            # record_transaction_data(step) # Record transaction data at this step

            # Update the grid display
            matrix.set_array(grid)

            # Clear all text from the grid initially
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    text[i][j].set_text('')  # Clear previous text

            # Display money values for agents
            for agent_id, (location, money, win, transactions, tax_paid, tax_rec, charity_rec, poor, rich) in money_of_agent.items():
                m, n = location  # Agent's location in the grid
                text[m][n].set_text(f'{int(money)}')  # Display agent's money
                text[m][n].set_color('white' if money > 2 else 'black')  # Adjust text color for better contrast

            # Update title and return all drawable elements
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
            money_of_agent,total_transaction,transaction_count,transaction_amounts = economic_transaction(grid, money_of_agent, delta_m, p_t, p_i)
            total_transactions_per_step.append(total_transaction)  
            total_transaction_counts.append(transaction_count)
            all_transaction_amounts.extend(transaction_amounts)

            money_of_agent = tax(money_of_agent, delta_m, psi_max, omega, m_tax)
            money_of_agent = charity(money_of_agent, mr, mp, mc, charity_probability)

            record_transaction_data(step) # Record transaction data at this step

            # Collect data at this step if needed
            
            average_money = np.mean([money for _, (_, money, _, _, _, _, _, _, _) in money_of_agent.items()])  
            total_agents = len(money_of_agent)
            total_agents_list.append(total_agents)

    # Save recorded data to CSV
    filepath = f"data/transaction_log_run_{run_num}.csv"

    with open(filepath, "w") as file:
        pd.DataFrame(data_log).to_csv(filepath, index=False) # write df

    # append param_list to the beginning of the file
    with open(filepath, 'a') as file:
        file.write("\n" + param_list + "\n")

    return averages,total_transactions_per_step,total_transaction_counts,all_transaction_amounts #,return any data of interest from this function



if __name__ == '__main__':
    """input parameters"""
    height = 20
    width = 20
    probablility_move = 0.8  # chance of movement of indiviudual
    steps = 200  # timesteps
    density = 0.2

    # animation and logging
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

    run_num = 1 # run number for saving data
    param_list = f"m0: {m0}; delta_m: {delta_m}; p_t: {p_t}; p_i: {p_i}; mr: {mr}; mp: {mp}; mc: {mc}; pc: {pc}; m_tax: {m_tax}; psi_max: {psi_max}; omega: {omega}; m_p: {m_p}; m_r:{m_r}; m_c:{m_c}; charity_probability: {charity_probability}"

    # set up + initialize the grid
    grid = initialize_grid(height, width, density, m0)

    # start animation, any data of interest can be returned from animate_CA
    averages,total_money_transacted_per_timestep,total_transaction_counts,all_transaction_amounts = animate_CA(grid, steps,showmovements,show_animation, interval=100, probablility_move=probablility_move, 
                          delta_m=delta_m, p_t=p_t, p_i=p_i, psi_max=psi_max, omega=omega, mr=mr, mp=mp, mc=mc, pc=pc,m_tax=m_tax, charity_probability=charity_probability, run_num=run_num, param_list=param_list)

    # plot distribution of total money transacted each timestep
    plot_transaction_distribution(total_money_transacted_per_timestep)
    plot_transaction_counts(total_transaction_counts)




