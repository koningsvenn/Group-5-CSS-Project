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
    """Create a height x width grid with zeros representing empty cells or integers 
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


def move(m, n, height, width):
    """Move the persons randomly."""
    # direction options
    directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])  # right, left, bottom, top
    direction = directions[random.randint(0, directions.shape[0] - 1)]  # choose a random row
    
    # calculate new coordinates based on direction
    m_new = (m + direction[0]) % height  # keep within boundary
    n_new = (n + direction[1]) % width   # keep within boundary
    
    return m_new, n_new

def time_step_randwalk(grid, probablility_move,showmovements):
    """Perform a time step where the values move randomly without merging."""
    height, width = grid.shape
    new_grid = np.zeros_like(grid)  # initialize a new grid for the updated state
    
    movements = []  # list to track movements
    occupied_count = 0  # counter for occupied cells
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

                # random movement
                if random.random() < probablility_move:  # move with some probability
                    m_new, n_new = globals()['move'](m, n, height, width)  # explicitly use the global move function

                    # add the person to the grid if the cell is empty and not visited
                    if new_grid[m_new, n_new] == 0 and (m_new, n_new) not in visited:
                        new_grid[m_new, n_new] = 1
                        visited.add((m_new, n_new))  # mark the cell as visited
                        movements.append(((m, n), (m_new, n_new)))  # log movement
                    else:
                        new_grid[m, n] = 1  # stay in place if target is occupied
                        movements.append(((m, n), (m, n)))  # log no movement due to target occupied
                else:
                    new_grid[m, n] = 1  # stay in place
                    movements.append(((m, n), (m, n)))  # log no movement

    if showmovements:
        print(f"Occupied Cells: {occupied_count}")
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
    total_drops_list = []
    
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




