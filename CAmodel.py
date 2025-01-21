def transaction_rule(grid, delta_m, p_t):
    height, width = grid.shape

    for _ in range(height * width):  # Every agent on the grid can interact with its neighbours
        x, y = np.random.randint(0, height), np.random.randint(0, width)    # Random agent
        dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])    
        nx, ny = (x + dx) % height, (y + dy) % width    # Random neighbour

        if dx == 0 and dy == 0:
            continue  # Skip self-interactions

        m_i, m_j = grid[x, y], grid[nx, ny]

        if m_i +m_j == 0:
            continue

        R = np.random.random()  # Generate random number between 0 and 1

        if m_i == 0:
            if R < p_t / 2:
                grid[x, y] += delta_m
                grid[nx, ny] -= delta_m

        elif m_j == 0:
            if R < p_t / 2:
                grid[x, y] -= delta_m
                grid[nx, ny] += delta_m

        elif m_i > m_j:
            if R < p_t / 2 + p_t:
                grid[x, y] += delta_m
                grid[nx, ny] -= delta_m
            elif R < p_t:
                grid[x, y] -= delta_m
                grid[nx, ny] += delta_m

        elif m_i <= m_j:
            if R < p_t / 2:
                grid[x, y] += delta_m
                grid[nx, ny] -= delta_m
            elif R < p_t:
                grid[x, y] -= delta_m
                grid[nx, ny] += delta_m

    return grid
