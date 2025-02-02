# Cellular Automata Simulation

This Python script simulates Lattice-Gas Cellular Automata (CA) environment where agents interact economically, move randomly, and engage in taxation and charity processes. The simulation can optionally display animations or run in the background for heavier computations.

## Features
- Simulates economic transactions between agents.
- Implements taxation and charity redistribution processes.
- Agents move randomly within the grid.
- Tracks various interesting variables such as:
  - **Average wealth of agents over time**.
  - **Total money transacted per timestep**.
  - **Transaction count per timestep**.
- Option to display movements and animation.

## Folders
- `./code` contains the main simulation code, plotting scripts, file sorting scripts (if needed) and the scripts to test different parameters using Snellius. ***Note***: only use `./code/sorting_scripts` when there are many files in `data/data_unsorted` that need to be sorted to run certain plotting scripts.
- `./plots` contains the plots made from gathered data.

## Files
- `simulation.py` => Main file that runs the simulations and gathers data. Saves a data file in `data/data_unsorted`.
- `plotting_scripts/plot_distribution.py` => Animated plot of distribution wealth of agents in each time step in a single simulation run. Data from the `data/data_unsorted` folder required.
- `plotting_scripts/plot_finite_size_scaling.py` => Generates plots of fraction of rich agents in the system in the last time step (when equilibrium has been reached) versus varying values of tax (psi_max) or charity (m_c) contributions.
- `plotting_scripts/phase_transition.py` => Creates phase transition plots for all available files in `data/data_sorted` to visually find if the number of rich agents changes dramatically when varying parameters like tax (psi_max), charity (m_c) or grid size.
- `plotting_scripts/plot_utils.py` => Contains the functions to plot transaction distribution and transaction counts of agents over time.
- `plotting_scripts/plot_other.py` => Plots to look at wealth distribution over all time steps and plot if number of rich agents stabilizes in the final time steps (actual number of time steps determined by data in `data/*` files).

## Parameters
The simulation behavior can be customized using the following parameters in the initialization section (`__main__` block):
- **Grid Dimensions**:
  - `height`: Number of rows in the grid.
  - `width`: Number of columns in the grid.
- **Agent Density**:
  - `density`: Proportion of grid cells occupied by agents (value between 0 and 1).
- **Movement Probability**:
  - `probability_move`: Likelihood of an agent moving during a timestep.
- **Economic Parameters**:
  - `m0`: Initial money assigned to each agent.
  - `delta_m`: Unit amount of money involved in transactions.
  - `p_t`: Probability of a transaction occurring.
  - `p_i`: Inequality parameter affecting transaction probabilities.
  - `m_tax`: Money threshold for taxation.
  - `psi_max`: Maximum tax rate.
  - `omega`: Parameter affecting tax rate calculation.
  - `m_p`: Poverty line for charity redistribution.
  - `m_r`: Wealth threshold for charity contribution.
  - `m_c`: Fixed charity contribution amount.
  - `charity_probability`: Likelihood of an agent donating to charity.
  - `run_num`: Current number of simulation (example: set as 2 if running `simulation.py` for the second time to generate a data file.)

## Usage

### How to run?
Edit parameters in the 'main' section of `code/simulation.py`. Use Python 3.12+ to run the simulations. The code should work for some older versions of Python (3.8+). Before running the code, install all required packages from `requirements.txt`. You can use pip package manager to do so. Install using: 

```pip install -r requirements.txt```

### Key Options
- **Display Movements** in code/simulation.py:
  - Set `showmovements = True` to log agent movements in the terminal.
  - Set `showmovements = False` to disable movement logging.
- **Show Animation** in code/simulation.py:
  - Set `show_animation = True` to visualize the simulation in an animated grid. **Note:** This does not generate a data file! Set the boolean to False to generate a data file in `data/data_unsorted`.
  - Set `show_animation = False` for faster processing, especially for long simulations or heavy computations.

### Example Customization
```python
height = 20              # Grid height
width = 20               # Grid width
density = 0.2            # Agent density (20% of cells occupied by agents)
probability_move = 0.8   # Movement probability
steps = 200              # Number of timesteps
showmovements = False    # Disable movement logging
show_animation = False   # Disable animation for faster processing
