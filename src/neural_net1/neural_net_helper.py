# Activation function
import numpy as np


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


def to_single_line(input):
    return str(input).replace('\r', '').replace('\n', '')


def load_weights(name, number):
    return np.load("{}_weights{}.npy".format(name, number))


def save_weights(name, number, data):
    np.save("{}_weights{}.npy".format(name, number), data)
