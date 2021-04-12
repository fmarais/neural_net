# Activation function
import numpy as np


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


def to_single_line(input):
    return str(input).replace('\r', '').replace('\n', '')


def load_weights(weights_name, weights_number):
    return np.load("{}_weights{}.npy".format(weights_name, weights_number))


def save_weights(weights_name, weights_number, weights):
    np.save("{}_weights{}.npy".format(weights_name, weights_number), weights)
