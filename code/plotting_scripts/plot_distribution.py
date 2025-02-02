import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# load the data
run_number = 1
# for run_number in run_numbers:
filepath = f"data/data_unsorted/transaction_log_run_{run_number}.csv"

# Get the parameters used in the run
with open(filepath, 'r') as file:
    lines = file.readlines()
    parameter_string = lines[-1].split("; ")
    parameters = {param.split(": ")[0]: float(param.split(": ")[1]) for param in parameter_string}

# Convert data to pandas df
data = pd.read_csv(filepath)

# Get unique time steps
time_steps = sorted(data['Time step'].unique())

# Get axis + limits
x_min, x_max = data['Total wealth'].min(), data['Total wealth'].max()
y_max = data.groupby('Time step')['Total wealth'].apply(lambda x: x.value_counts().max()).max() + 5

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Function to update frame
def update(frame):
    ax.clear()
    current_data = data[data['Time step'] == time_steps[frame]]
    ax.hist(current_data['Total wealth'], bins=20, alpha=0.75, color='blue', edgecolor='black')
    ax.set_xlabel("Total Wealth")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Income Distribution at Timestep {time_steps[frame]}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 25)
    ax.grid(True)

# Create animation
ani = FuncAnimation(fig, update, frames=len(time_steps), repeat=True, interval=200)

plt.show()
