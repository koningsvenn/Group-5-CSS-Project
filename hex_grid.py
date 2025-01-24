import numpy as np
# import pandas as pd # export test data to csv

# # build a hexagonal grid
# ratio = 1/np.sqrt(2)            # vertical distance between hexagons
# n = 121                         # number of spots (ideally a square number)
# n_x = int(np.sqrt(n)/ratio)     # number of spots in x direction
# n_y = n//n_x                    # number of spots in y direction

# # meshgrid
# xv, yv = np.meshgrid(np.arange(n_x), np.arange(n_y), sparse=False, indexing='xy')

# xv = xv * ratio         # stretch the grid
# xv[::2, :] += ratio/2   # interlace every two lines by half the width

# # plot the grid
# plt.figure(figsize=(8, 8))
# plt.scatter(xv, yv, s=1)
# plt.axis('equal')
# plt.show()

class HexagonalGrid:
    def __init__(self, size):
        """Initialize the hexagonal grid."""
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)

    def random_initialize(self, num_agents):
        """Randomly distribute agents on the grid."""
        positions = np.random.choice(self.size * self.size, num_agents, replace=False)
        for pos in positions:
            x, y = divmod(pos, self.size)
            self.grid[x, y] = 1

    def find_neighbors(self, x, y):
        """Find hexagonal neighbors for a given cell."""
        offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, 1) if y % 2 == 0 else (-1, -1),
            (1, 1) if y % 2 == 0 else (1, -1)
        ]
        neighbors = []
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        return neighbors
