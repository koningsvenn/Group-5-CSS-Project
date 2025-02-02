# Plot the relations between the features and the target variable
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
run_number = 3
# read in multiple files
# run_numbers = [1]

# for run_number in run_numbers:
filepath = f"data/transaction_log_run_{run_number}.csv"

# Get the parameters used in the run
with open(filepath, 'r') as file:
    lines = file.readlines()
    parameter_string = lines[-1].split("; ")
    parameters = {param.split(": ")[0]: float(param.split(": ")[1]) for param in parameter_string}

# Convert data to pandas df
data = pd.read_csv(filepath)
# print(parameters)
# print(data.columns)
# Plot ----
# 1. p_m against 'income distribution' 
# plot frequency of a certain "total wealth" value
data['income distribution'] = (data['Total wealth']-parameters['m0']) / parameters['m0']

# group agents by id
# Assuming 'df' is your dataset
# print(data['Total wealth'])
# data['Total wealth'] = data['Total wealth'] - parameters['m0']
# print(data['Total wealth'])
data_grp_timestep = data.groupby('Time step')['Total wealth'].sum().reset_index()

# rich agents 
data_rich_agents = data[data['Rich'] == 1]
data_grp_timestep_rich = data_rich_agents.groupby('Time step')['Total wealth'].sum().reset_index()

# poor agents
data_poor_agents = data[data['Poor'] == 1]
data_grp_timestep_poor = data_poor_agents.groupby('Time step')['Total wealth'].sum().reset_index()
# data_grp_timestep = data.groupby('Time step')['Amount of income gained/lost'].mean().reset_index()

# Plotting ----
plt.figure(figsize=(8, 5))

plt.plot(data_grp_timestep['Time step'], data_grp_timestep['Total wealth'], marker='', linestyle='-')
plt.plot(data_grp_timestep_rich['Time step'], data_grp_timestep_rich['Total wealth'], marker='', linestyle='-')
plt.plot(data_grp_timestep_poor['Time step'], data_grp_timestep_poor['Total wealth'], marker='', linestyle='-')

plt.xlabel("Timestep")
plt.ylabel("Total Wealth")
plt.title(f"Total Wealth Over Time (P_I = {parameters['p_i']}, m_c = {parameters['m_c']}, psi_max = {parameters['psi_max']}, m_0 = {parameters['m0']})")
plt.grid(True)
plt.legend(['Total wealth', 'Rich agents', 'Poor agents'])
plt.savefig(f"plots/total_wealth_run_{run_number}.png")
plt.show()

