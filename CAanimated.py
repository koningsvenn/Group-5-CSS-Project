import numpy as np 
import numpy.random as random
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd



# track money and position of each agent
money_of_agent = {}

"""set up the grid"""
def initialize_grid(height, width, fall_heigth, density):
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
            money = 2
            money_of_agent[agents] = [location, money]
            # money_of_agent[(m, n)] = 2
        agents += 1
    print(money_of_agent)
    return grid 



#--------------------------- NOT USED FUNCTIONS ATM ---------------------------
def transaction_rule(grid, delta_m, p_t, p_l):
    """Apply transaction rules between agents."""
    height, width = grid.shape

    for _ in range(height * width):
        x, y = np.random.randint(0, height), np.random.randint(0, width)
        dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
        nx, ny = (x + dx) % height, (y + dy) % width

        if dx == 0 and dy == 0:
            continue

        m_i, m_j = grid[x, y], grid[nx, ny]

        if m_i + m_j == 0:
            continue

        R = np.random.random()

        if m_i == 0 and R < p_t / 2:
            grid[x, y] += delta_m
            grid[nx, ny] -= delta_m
        elif m_j == 0 and R < p_t / 2:
            grid[x, y] -= delta_m
            grid[nx, ny] += delta_m
        elif m_i > m_j:
            if R < p_t / 2 + p_l:
                grid[x, y] += delta_m
                grid[nx, ny] -= delta_m
            elif R < p_t:
                grid[x, y] -= delta_m
                grid[nx, ny] += delta_m
        elif m_i <= m_j:
            if R < p_t / 2 - p_l:
                grid[x, y] += delta_m
                grid[nx, ny] -= delta_m
            elif R < p_t:
                grid[x, y] -= delta_m
                grid[nx, ny] += delta_m

    return grid

def tax(grid, psi_max, omega, delta_m):
    """Apply tax rules."""
    m_max = np.max(grid)
    
    # Handle the case where m_max is zero
    if m_max == 0:
        return grid  # No taxation if there is no wealth
    
    # Calculate tax liabilities
    psi_i = ((grid / m_max) ** omega) * psi_max
    tax_liabilities = psi_i * delta_m

    # Avoid NaN or negative wealth
    tax_liabilities = np.nan_to_num(tax_liabilities, nan=0.0, posinf=0.0, neginf=0.0)
    grid -= tax_liabilities

    total_tax_revenue = np.sum(tax_liabilities)

    # Redistribute tax revenue equally
    redistribution = total_tax_revenue / grid.size
    grid += redistribution

    # Ensure no NaN or negative values remain in the grid
    grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    grid[grid < 0] = 0  # Prevent negative wealth

    return grid

def charity(grid, mr, mp, mc, pc):
    """Apply charity rules."""
    rich_agents = np.where(grid > mr)
    total_charity = 0

    for x, y in zip(rich_agents[0], rich_agents[1]):
        if np.random.random() < pc:
            donation = min(mc, grid[x, y])
            grid[x, y] -= donation
            total_charity += donation

    poor_agents = np.where(grid < mp)
    if len(poor_agents[0]) > 0:
        redistribution = total_charity / len(poor_agents[0])
        for x, y in zip(poor_agents[0], poor_agents[1]):
            grid[x, y] += redistribution

    return grid
#--------------------------- NOT USED FUNCTIONS ATM ------------------------




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
    Perform economic transactions between neighboring agents.
    Args:
        grid: The grid representing agent positions.
        money_of_agent: A dictionary mapping agent IDs to their location and money.
        delta_m: Fixed amount of money transferred in a transaction.
        p_t: Probability of a transaction occurring between neighbors.
        p_i: Inequality parameter to control transaction probabilities.
    Returns:
        Updated money_of_agent dictionary.
    """
    height, width = grid.shape
    visited = set()

    # iterate over all agents
    for agent_id, (location, money) in money_of_agent.items():
        m, n = location
        neighbors = []

        # identify 8 neighbors (D2N8 model)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                m_new, n_new = (m + i) % height, (n + j) % width
                if grid[m_new, n_new] == 1:  # check if a neighbor is present
                    for neighbor_id, (neighbor_loc, neighbor_money) in money_of_agent.items():
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
            elif money >= neighbor_money > 0:  # case 3
                if R < (p_t / 2 + p_i):  # agent Ai wins money
                    money_of_agent[agent_id][1] += delta_m
                    money_of_agent[neighbor_id][1] -= delta_m
                elif R < p_t:  # agent Ai loses money
                    money_of_agent[agent_id][1] -= delta_m
                    money_of_agent[neighbor_id][1] += delta_m
            elif neighbor_money > money > 0:  # case 4
                if R < (p_t / 2 - p_i):  # agent Ai wins money
                    money_of_agent[agent_id][1] += delta_m
                    money_of_agent[neighbor_id][1] -= delta_m
                elif R < p_t:  # agent Ai loses money
                    money_of_agent[agent_id][1] -= delta_m
                    money_of_agent[neighbor_id][1] += delta_m

            # ensure money remains non-negative
            money_of_agent[agent_id][1] = max(0, money_of_agent[agent_id][1])
            money_of_agent[neighbor_id][1] = max(0, money_of_agent[neighbor_id][1])

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
def animate_CA(initial_grid, steps,showmovements, interval, probablility_move, delta_m, p_t, p_l, psi_max, omega, mr, mp, mc, pc):

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
    averages = []
    raind_count_list = []
    total_agents_list = []
    
    def update(frames):
        global money_of_agent  # access the global variable
        nonlocal grid

        # Perform a time-step
        grid = time_step_randwalk(grid, probablility_move, showmovements)
        money_of_agent = economic_transaction(grid, money_of_agent, delta_m, p_t, p_l)
        # print(money_of_agent)

        # update the grid display
        matrix.set_array(grid)

        # clear all text from the grid initially
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                text[i][j].set_text('')  # clear previous text

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
    return averages #,return any data of interest from this function



if __name__ == '__main__':
    """input parameters"""
    height = 15
    width = 15
    probablility_move = 0.3  # chance of movement of indiviudual
    steps = 100  # timesteps
    density = 0.2
    showmovements = False
    delta_m = 1.0
    p_t = 1.0
    p_l = 1.0
    psi_max = 1.0
    omega = 1.0
    mr = 1.0
    mp = 1.0
    mc = 1.0
    pc = 1.0
    
    """set up grid"""
    grid = initialize_grid(height, width, 0, density)  # init. the grid

    """start animation, any data of interest can be returned from animate_CA"""
    averages = animate_CA(grid, steps,showmovements, interval=100, probablility_move=probablility_move, 
                          delta_m=delta_m, p_t=p_t, p_l=p_l, psi_max=psi_max, omega=omega, mr=mr, mp=mp, mc=mc, pc=pc)




