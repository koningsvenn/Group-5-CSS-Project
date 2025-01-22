import numpy as np 
import numpy.random as random
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import datetime
import os


"""set up the grid"""
def initialize_grid(height, width, fall_heigth, density):
    """create a height x width grid with zeros representing empty cells or integers 
    to represent person size"""
    #empty grid
    grid = np.zeros((height, width))  
    drops_amount = int(height * width * density) 
    
    #select random coordinates for each person
    drops = 0
    while drops < drops_amount:
        n = random.randint(0, width - 1)
        m = random.randint(0, height - 1)

        if grid[m, n] == 0:
            grid[m, n] = 1  
            drops += 1
        
    return grid 



"""Time steps and movement"""
def time_step_randwalk(grid, probablility_move):
    """Perform a time step where the values move randomly without merging."""
    height, width = grid.shape
    new_grid = np.zeros_like(grid)  # initialize a new grid for the updated state

    # loop over all cells
    for m in range(height):
        for n in range(width):
            if grid[m, n] == 1:  # if there is a person in a cell
                # random movement
                if random.rand() < probablility_move:  # move with some prob.
                    m_new, n_new = move(m, n, height, width)

                    # add the person to the grid if  cell is empty
                    if new_grid[m_new, n_new] == 0:
                        new_grid[m_new, n_new] = 1
                    else:
                        
                        new_grid[m, n] = 1
                else:
                    new_grid[m, n] = 1  

    return new_grid


def move(m, n, height, width):
    """Move the persons randomly."""
    # direction options
    directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])  # right, left, bottom, top
    direction = directions[random.randint(0, directions.shape[0] - 1)]  # choose a random row
    
    # calculate new coordinates based on direction
    m_new = (m + direction[0]) % height  # keep within boundary
    n_new = (n + direction[1]) % width   # keep within boundary
    
    return m_new, n_new


"""animation and plotting"""
def get_shades_of_blue(n):
    """Generate n shades of blue."""
    start = np.array([173, 216, 230]) / 255  # lightblue
    end = np.array([25, 25, 112]) / 255  # midnightblue
    return [(start + (end - start) * i / (n - 1)).tolist() for i in range(n)]


def animate_CA(initial_grid, steps, interval, probablility_move):
    """Animate the cellular automata, updating time step and cell values."""
    #set up colors and figure for the animation
    colors = get_shades_of_blue(20) 
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
    total_drops_list = []
    
    def update(frames):
        nonlocal grid

        #perform a time-step
        grid = time_step_randwalk(grid, probablility_move)  
        matrix.set_array(grid)
        # Update text for each cell
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                text_color = 'white' if val > 2 else 'black'  # Make values of dropletsize white or black for contrast
                text[i][j].set_text(f'{int(val)}' if val else '')
                text[i][j].set_color(text_color)
                text[i][j].set_visible(bool(val))  # Show text only for non-zero values

        #update data lists
        non_zero_elements = np.count_nonzero(grid)
        average_size = np.sum(grid) / non_zero_elements if non_zero_elements else 0
        averages.append(int(average_size))
        raind_count_list.append(np.count_nonzero(grid) - non_zero_elements)
        total_drops_list.append(np.count_nonzero(grid))

        ax.set_title(f"Economy Automata")
        return [matrix] + [txt for row in text for txt in row]

    ani = FuncAnimation(fig, update, frames=steps-1, interval=interval, blit=False, repeat=False) #Average step -1 because the first frame is a step and thus average dropletsize
    plt.show()
    return averages, raind_count_list,total_drops_list



"""run experiments and collect data"""
def run_simulation(initial_grid, steps, probablility_move):
    """run a cloud simulation without animation"""
    #set up the grid
    grid = np.copy(initial_grid)

    #lists to collect data
    averages = []
    raind_count_list = []
    total_drops_list = []

    #run steps amount of time_steps
    for i in range(steps):
        #perform a time-step
        grid = time_step_randwalk(grid, probablility_move)
        #average dropsize
        non_zero_elements = np.count_nonzero(grid)
        average_size = np.sum(grid) / non_zero_elements if non_zero_elements else 0
        #add data to lists
        averages.append(int(average_size))
        raind_count_list.append(np.count_nonzero(grid) - non_zero_elements)
        total_drops_list.append(np.count_nonzero(grid))
    
    return averages, raind_count_list,total_drops_list

def run_experiment(height,width,humidity,steps,probablility_move):
    """run a few simulations and take the averages"""
    #set up the grid
    grid = initialize_grid(height, width, 0, humidity)

    #lists to collect averages
    rain_mean_list = []
    size_mean_list = []
    total_mean_list = []
    max_drop_mean_list = []

    #run 40 simulations for each humidity
    n = 40
    for i in range(n):
        #one simulation
        averages,rain_count_list,total_drops_list = run_simulation(grid,steps,probablility_move)
        #add average outcomes to lists
        rain_mean_list.append(np.mean(rain_count_list))
        size_mean_list.append(np.mean(averages))
        total_mean_list.append(np.mean(total_drops_list))

    #average and standard deviation of the 40 simulations
    rain_mean = np.mean(rain_mean_list)
    rain_std = np.std(rain_mean_list)
    size_mean = np.mean(size_mean_list)
    size_std = np.std(size_mean_list)
    total_mean = np.mean(total_mean_list)
    total_std = np.std(total_mean_list)
    max_drop_mean = np.mean(max_drop_mean_list)
    max_drop_std = np.std(max_drop_mean_list)
    
    return rain_mean, rain_std, size_mean, size_std, total_mean, total_std, max_drop_mean, max_drop_std


if __name__ == '__main__':
    """input parameters"""
    height = 15
    width = 15
    probablility_move = 0.5  # chance of movement of indiviudual
    steps = 100  # timesteps
    density = 0.5

    """set up grid"""
    grid = initialize_grid(height, width, 0, density)  # init. the grid

    """start de animatie"""
    averages, raind_count_list, total_drops_list = animate_CA(grid, steps, interval=1000, probablility_move=probablility_move)




