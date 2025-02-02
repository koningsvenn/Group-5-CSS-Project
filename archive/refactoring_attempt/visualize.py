import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from archive.refactoring_attempt.grid import time_step_randwalk
from archive.refactoring_attempt.agent import economic_transaction, tax, charity

def get_shades_of_green(n):
    """Generate n shades of green."""
    start = np.array([144, 238, 144]) / 255
    end = np.array([0, 100, 0]) / 255
    return [(start + (end - start) * i / (n - 1)).tolist() for i in range(n)]

def animate_CA(grid, steps, interval, prob_move, delta_m, p_t, p_i, psi_max, omega, m_r, m_p, m_c, m_tax, charity_probability, money_of_agent):
    """Animate cellular automata for economic simulation."""
    colors = get_shades_of_green(20)
    cmap = LinearSegmentedColormap.from_list("custom_green", colors, N=256)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xticks([])
    ax.set_yticks([])
    matrix = ax.matshow(grid, cmap=cmap)
    
    def update(frame):
        nonlocal grid
        grid = time_step_randwalk(grid, prob_move, money_of_agent)
        money_of_agent, _, _, _ = economic_transaction(grid, money_of_agent, delta_m, p_t, p_i)
        money_of_agent = tax(money_of_agent, delta_m, psi_max, omega, m_tax)
        money_of_agent = charity(money_of_agent, m_r, m_p, m_c, charity_probability)
        matrix.set_array(grid)
        return [matrix]
    
    ani = FuncAnimation(fig, update, frames=steps, interval=interval, blit=False, repeat=False)
    plt.show()