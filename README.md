# Cellular Automata Simulation

This Python script simulates a Cellular Automata (CA) environment where agents interact economically, move randomly, and engage in taxation and charity processes. The simulation can optionally display animations or run in the background for heavier computations.

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
- ./results contains the plots made from gathered data

## Files
- CAanimated.py => main file that runs the simulations and gathers data
- plot_utils.py => file that contains the functions to plot the gathered data from
CAanimated.py and saves them

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

## Functions
- `animate_CA`: The main function that runs the simulation. Returns:
  - `averages`: Average wealth of agents at each timestep.
  - `total_transactions_per_step`: Total money transacted per timestep.
  - `total_transaction_counts`: Total transaction count per timestep.

## Usage
### Key Options
- **Display Movements**:
  - Set `showmovements = True` to log agent movements in the terminal.
  - Set `showmovements = False` to disable movement logging.
- **Show Animation**:
  - Set `show_animation = True` to visualize the simulation in an animated grid.
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
