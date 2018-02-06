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

# Plot possible hyperplane based on guess
# plt.plot([-2, 6], [6, 0.5])

# Define possible loss function (error, to minimise) and objective function (to optimise)
def svm_sgd_plot(X, Y):
    # Perform stochastic gradient descent to learn separating hyperplane between the two variables

    # Initialise SWM vector with zeroes
    w = np.zeros(len(X[0]))
    # Learning rate
    eta = 1
    # Number of iterations
    epochs = 100000
    # Store misclassified data points to plot how they change over iterations
    errors = []

    # Perform gradient descent
    for epoch in range(1, epochs):
        error = 0
        for index, datapoint in enumerate(X):
            # Check if misclassified
            # Misclassifiction formula
            if (Y[index]*np.dot(X[index], w)) < 1:
                # Misclassified update of weight with gradients of both terms
                w = w + eta * ((X[index] * Y[index]) + (-2 * (1/epoch) * w))
                error = 1
            else:
                # Correctly classified, update weight by gradient of reguliser
                w = w + eta * (-2 * (1/epoch) * w)
            errors.append(error)

    plt.plot(errors, "|")
    plt.ylim(0.5, 1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel("Epoch")
    plt.ylabel("Misclassified")
    plt.show()

    return w

w = svm_sgd_plot(X, y)

for index, datapoint in enumerate(X):
    if index < 2:
        # Negative samples
        plt.scatter(datapoint[0], datapoint[1], s=120, marker="_", linewidths = 2)
    else:
        # Positive samples
        plt.scatter(datapoint[0], datapoint[1], s=120, marker="+", linewidths = 2)

# Add test samples
plt.scatter(2, 2, s=120, marker="_", linewidths = 2, color="yellow")
plt.scatter(4, 3, s=120, marker="+", linewidths = 2, color="blue")

# Print hyperplane calculated by svm_sgd_plot(X, Y)
x2 = [w[0], w[1], -w[1], w[0]]
x3 = [w[0], w[1], w[1], -w[0]]

x2x3 = np.array([x2, x3])
X, Y, U, V = zip(*x2x3)
# gca = get current axes
ax = plt.gca()
# Plot a quiver plot
ax.quiver(X, Y, U, V, scale=1, color="blue")

plt.show()
