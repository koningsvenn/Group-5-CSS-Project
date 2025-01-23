import numpy as np 
import numpy.random as random
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import datetime
import os


"""set up the grid"""

money_of_agent = {}

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

# def print_money_of_agent():
#     """
#     Prints the global dictionary of money_of_agent.
#     """
#     global money_of_agent
#     print("money_of_agent:")
#     for coord, value in money_of_agent.items():
#         print(f"ID: {coord}, Value: {value}")

grid = initialize_grid(15, 15, 0, density=0.5)
print("Initial grid:")
print(grid)
print("\nmoney_of_agent:")
print_money_of_agent()



def move(m, n, height, width):
    """Move the persons randomly."""
    # direction options
    directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])  # right, left, bottom, top
    direction = directions[random.randint(0, directions.shape[0] - 1)]  # choose a random row
    
    # calculate new coordinates based on direction
    m_new = (m + direction[0]) % height  # keep within boundary
    n_new = (n + direction[1]) % width   # keep within boundary
    
    return m_new, n_new

def transaction_rule(grid, delta_m, p_t, p_l):
    """Apply transaction rules between the agents."""
    for _ in range(height * width):
        x, y = np.random.randint(0, height), np.random.randint(0, width)
        dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
        nx, ny = (x + dx) % height, (y + dy) % width
        if dx == 0 and dy == 0:
            continue

        m_i, m_j = grid[x, y], grid[nx, ny]

        if m_i + m_j ==0:
            continue

        R = np.random.random()

        if m_i == 0:
            if R < p_t / 2:
                grid[x, y] += delta_m
                grid[nx, ny] -= delta_m

        elif m_j == 0:
            if R < p_t / 2:
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
    """Apply tax rule."""
    m_max = np.max(grid)
    psi_i = ((grid / m_max) ** omega) * psi_max
    tax_liabilities = psi_i * delta_m
    grid -= tax_liabilities
    total_tax_revenue = np.sum(tax_liabilities)
    num_agents = grid.size
    redistribution = total_tax_revenue / num_agents
    grid += redistribution
    return grid

def charity(grid, mr, mp, mc, pc):
    """Apply charity rule."""
    rich_agents = np.where(grid > mr)
    total_charity = 0
    for x, y in zip(rich_agents[0], rich_agents[1]):
        if np.random.random() < pc:
            donation = min(mc, grid[x, y])
            grid[x, y] -= donation
            total_charity += donation

    poor_agents = np.where(grid < mp)
    num_poor_agents = len(poor_agents[0])
    if num_poor_agents > 0:
        redistribution = total_charity / num_poor_agents
        for x, y in zip(poor_agents[0], poor_agents[1]):
            grid[x, y] += redistribution

    return grid

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
                agent_id = money_of_agent.keys[money_of_agent()[0]==loc]

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

def neighbour_figure_outer(new_grid, money_of_agents):
    # Return the neighbours of each agent in the grid as a dictionary.
    neighbours = {}
    for agent_id, value in money_of_agents.items():
        m, n = value[0]
        neighbours[agent_id] = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                m_new = (m + i) % new_grid.shape[0]
                n_new = (n + j) % new_grid.shape[1]
                if new_grid[m_new, n_new] == 1:
                    # add agent_id value based on m_new and n_new
                    neighbours[agent_id].append((m_new, n_new))
    

"""Assign a color based on wealth"""
def get_shades_of_green(n):
    """Generate n shades of green."""
    start = np.array([144, 238, 144]) / 255  # lightgreen
    end = np.array([0, 100, 0]) / 255  # darkgreen
    return [(start + (end - start) * i / (n - 1)).tolist() for i in range(n)]

"""Animate the CA to visualize what is happening"""
def animate_CA(initial_grid, steps,showmovements, interval, probablility_move,):
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
        nonlocal grid

        #perform a time-step
        grid = time_step_randwalk(grid, probablility_move,showmovements)  
        matrix.set_array(grid)
        # update text for each cell
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                text_color = 'white' if val > 2 else 'black'  # make values of person wealth white or black for contrast
                text[i][j].set_text(f'{int(val)}' if val else '')
                text[i][j].set_color(text_color)
                text[i][j].set_visible(bool(val))  # show text only for non-zero values

        #add interesting data alculations here based on the grid
        non_zero_elements = np.count_nonzero(grid)
        average_size = np.sum(grid) / non_zero_elements if non_zero_elements else 0
        averages.append(int(average_size))

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
    density = 0.5
    showmovements = False

    """set up grid"""
    grid = initialize_grid(height, width, 0, density)  # init. the grid

    """start animation, any data of interest can be returned from animate_CA"""
    averages = animate_CA(grid, steps,showmovements, interval=100, probablility_move=probablility_move)




