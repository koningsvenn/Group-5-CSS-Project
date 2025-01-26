import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_transaction_distribution(total_money_transacted_per_timestep, bins=30, alpha=0.7, color='blue'):
    """
    Plots a histogram for the distribution of total money transacted per timestep.

    Parameters:
        total_money_transacted_per_timestep (list or array): The data for total money transacted per timestep.
        bins (int): Number of bins for the histogram. Default is 30.
        alpha (float): Transparency level for the bars. Default is 0.7.
        color (str): Color of the bars. Default is 'blue'.

    Returns:
       None: Saves and shows plot
    """

    #save plot in results folder with date and time
    os.makedirs("results", exist_ok=True)
    title = "tx_ammount_distribution"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{title.replace(' ', '_')}_{timestamp}.png"
    filepath = os.path.join("results", filename)
    
    plt.figure(figsize=(10, 6))
    plt.hist(total_money_transacted_per_timestep, bins=bins, alpha=alpha, color=color)
    plt.title('Distribution of Total Money Transacted Each Timestep')
    plt.xlabel('Total Transaction Amount')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()  
    plt.show()

def plot_transaction_counts(transaction_counts, bins=30, alpha=0.7, color='green'):
    """
    Plots a histogram for the distribution of transaction counts per timestep.

    Parameters:
        transaction_counts (list or array): The data for transaction counts per timestep.
        bins (int): Number of bins for the histogram. Default is 30.
        alpha (float): Transparency level for the bars. Default is 0.7.
        color (str): Color of the bars. Default is 'green'.

    Returns:
        None: Saves and shows plot
    """

        #save plot in results folder with date and time
    os.makedirs("results", exist_ok=True)
    title = "tx_count_distribution"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{title.replace(' ', '_')}_{timestamp}.png"
    filepath = os.path.join("results", filename)
    

    plt.figure(figsize=(10, 6))
    plt.hist(transaction_counts, bins=bins, alpha=alpha, color=color)
    plt.title('Distribution of transaction counts per timestep')
    plt.xlabel('Transaction counts')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()  
    plt.show()
