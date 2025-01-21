import numpy as np

def transaction_rule(grid, delta_m, p_t, p_l):
    height, width = grid.shape

    for _ in range(height * width):  # Every agent on the grid can interact with its neighbours
        # Select random agent and random neighbour
        x, y = np.random.randint(0, height), np.random.randint(0, width)
        dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])    
        nx, ny = (x + dx) % height, (y + dy) % width    

        if dx == 0 and dy == 0:
            continue  # No self-interactions

        m_i, m_j = grid[x, y], grid[nx, ny]

        if m_i +m_j == 0:
            continue # No transaction if both agents have zero wealth

        R = np.random.random()  # Generate random number between 0 and 1

        # Conditions from figure 3
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
    
    # Calculate maximum income
    m_max = np.max(grid)
    
    # Calculate individual tax liabilities
    psi_i = ((grid / m_max) ** omega) * psi_max  
    tax_liabilities = psi_i * delta_m

    # Deduct taxes from agents
    grid -= tax_liabilities

    # Calculate total tax revenue
    total_tax_revenue = np.sum(tax_liabilities)

    # Redistribute tax revenue equal
    num_agents = grid.size
    redistribution = total_tax_revenue / num_agents
    grid += redistribution

    return grid

def charity(grid, mr, mp, mc, pc):
    
    # Identify rich agents (m_i > mr)
    rich_agents = np.where(grid > mr)
    
    # Total charity pool
    total_charity = 0
    
    # Iterate over rich agents to collect charity contributions
    for x, y in zip(rich_agents[0], rich_agents[1]):
        if np.random.random() < pc:  # With probability Pc, rich agent donates
            donation = min(mc, grid[x, y])  # Prevent over-donation
            grid[x, y] -= donation
            total_charity += donation

    # Identify poor agents (m_i < mp)
    poor_agents = np.where(grid < mp)
    num_poor_agents = len(poor_agents[0])
    
    # Redistribute charity to poor agents
    if num_poor_agents > 0:
        redistribution = total_charity / num_poor_agents
        for x, y in zip(poor_agents[0], poor_agents[1]):
            grid[x, y] += redistribution

    return grid
