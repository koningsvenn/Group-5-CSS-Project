from matplotlib import pyplot as plt

def plot_wealth_distribution_across_time(data, parameters, run_number):
    """
    Plots the wealth distribution across time for all agents, rich agents, and poor agents.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the simulation data with columns 'Total wealth', 'Time step', 'Rich', and 'Poor'.
    parameters (dict): Dictionary containing the parameters of the simulation, including 'm0', 'p_i', 'm_c', and 'psi_max'.
    run_number (int): The run number of the simulation, used for saving the plot.
    
    Returns:
    None: The function saves the plot as a PNG file and displays it.
    """

    # Find income distribution as (wealth of agent / initial wealth)
    data['income distribution'] = (data['Total wealth']-parameters['m0']) / parameters['m0']

    # print(data['Total wealth'])
    data['Total wealth'] = data['Total wealth'] - parameters['m0']
    data_grp_timestep = data.groupby('Time step')['Total wealth'].sum().reset_index()

    # Rich agents 
    data_rich_agents = data[data['Rich'] == 1]
    data_grp_timestep_rich = data_rich_agents.groupby('Time step')['Total wealth'].sum().reset_index()

    # Poor agents
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

def plot_rich_agents(data, parameters):
    """
    Plots the number of rich agents over time and computes the rate of change in the number of rich agents.

    Parameters:
    data (pd.DataFrame): DataFrame containing the simulation data with columns 'Total wealth', 'Time step', 'Rich', and 'Poor'.
    parameters (dict): Dictionary containing the parameters of the simulation, including 'm0', 'p_i', 'm_c', and 'psi_max'.

    Returns:
    None: The function saves the plot as a PNG file and displays it.
    """

    # Rich agents 
    data_rich_agents = data[data['Rich'] == 1]
    data_grp_timestep_rich = data_rich_agents.groupby('Time step')['Rich'].sum()

    # Compute rate of change in the number of rich agents
    rich_agents_change_rate = data_grp_timestep_rich.diff()

    # Plot number of rich agents over time
    plt.figure(figsize=(8, 5))
    plt.plot(data_grp_timestep_rich.index, data_grp_timestep_rich.values, marker='', linestyle='-')
    # Check if trend is stabilizing (uncomment to plot)
    # plt.xscale('log')
    # Check differentials (uncomment to plot)
    # plt.plot(data_grp_timestep_rich.index[1:], data_grp_timestep_rich.values[1:]-data_grp_timestep_rich.values[:-1] )
    plt.xlabel("Timestep")
    plt.ylabel("Number of Rich Agents")
    plt.title(f"Number of Rich Agents Over Time ($P_I$ = {parameters['p_i']}, m_c = {parameters['m_c']}, psi_max = {parameters['psi_max']}, m_0 = {parameters['m0']})")
    plt.grid(True)
    # plt.savefig(f"plots/rich_agents.png")
    plt.show()
    
    return None