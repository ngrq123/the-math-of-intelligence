import numpy as np
from matplotlib import pyplot as plt

# Define data
X = np.array([
    [-2, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
    ])

# Output labels (-1 for '_', 1 for '+')
y = np.array([-1, -1, 1, 1, 1])

for index, datapoint in enumerate(X):
    if index < 2:
        # Negative samples
        plt.scatter(datapoint[0], datapoint[1], s=120, marker="_", linewidths = 2)
    else:
        # Positive samples
        plt.scatter(datapoint[0], datapoint[1], s=120, marker="+", linewidths = 2)

# Print possible hyperplane based on guess
plt.plot([-2, 6], [6, 0.5])

plt.show()
