
# Single Neuron:
import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
    dot_product = np.dot(features,weights)
    z = bias +  dot_product
    probabilities = 1/(1+np.exp(-z))
    mse = np.square(np.subtract(labels,probabilities)).mean()
    probabilities = np.round(probabilities,decimals=4)
    mse = np.round(mse,decimals=4)
    return probabilities, mse
# test run:
print('Single Neuron Model:\n',single_neuron_model([[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], [0, 1, 0], [0.7, -0.4], -0.1))
#expected output:
#([0.4626, 0.4134, 0.6682], 0.3349)


# Single Neuron with Backpropagation
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    mse_values = []
    weights = initial_weights.copy()
    bias = initial_bias
    for _ in range(epochs):
        z = bias +  np.dot(features ,weights)
        probabilities = 1/ (1 + np.exp(-z))
        mse = np.mean((probabilities - labels) ** 2)
        mse_values.append(round(mse, 4))

        # derivativ of sigmoid: σ'(x) = σ(x) * (1 - σ(x)) (probabilities * (1-probabilities))
        error = probabilities - labels

        gradient_of_mse_weights = (2 / len(labels)) * np.dot(features.T, error * probabilities * (1 - probabilities))
        gradient_of_mse_bias = (2 / len(labels)) * np.sum(error * probabilities * (1 - probabilities))

        weights -= learning_rate * gradient_of_mse_weights
        bias -= learning_rate * gradient_of_mse_bias

    weights = np.round(weights, 4)
    bias = np.round(bias, 4)
    mse_values = np.round(mse_values, 4)
    return weights, bias, mse_values

# test run:
print('Single Neuron with Backpropagation:\n',train_neuron(np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]),
                   np.array([1, 0, 0]), np.array([0.1, -0.2]),
                  0.0, 0.1, 2))
# expected output:
#updated_weights = ([0.1036, -0.1425], -0.0167, [0.3033, 0.2942])



# Notes:
# The learning rate is a scalar value that determines how much the weights change
# in each iteration of backpropagation
# What is an epoch in backpropagation?
# Iteration over batches: An epoch consists of iterating over all batches in the
# dataset, performing the forward pass, loss calculation, backpropagation, and
# weight update for each batch


