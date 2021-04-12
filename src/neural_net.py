import numpy as np

from src.neural_net_helper import load_weights, sigmoid_derivative, sigmoid


class NeuralNetwork:
    def __init__(self, _x, _y, weights_name):
        # settings
        # hidden_layer_node_count = 4
        self.hidden_layer_node_count = 10
        # learn_rate = 2
        self.learn_rate = 2

        self.init_network(_x, weights_name)
        self.init_output(_y)

    def init_network(self, _x, saved_weights_name):
        self.input = _x
        # [[0.  0.  1.]
        #  [0.  1.  1.]
        #  [1.  0.  1.]
        #  [1.  1.  1.]]

        # width - self.input.shape[0] - number of iterations - eg. 4
        # height - self.input.shape[1] - number of inputs - eg. 5
        # hidden layers (weights)

        # try to load saved weights
        try:
            self.weights1 = load_weights(saved_weights_name, "1")
            self.weights2 = load_weights(saved_weights_name, "2")
            # we have loaded weights, freeze the learning rate
            learn_rate = 0
        except FileNotFoundError:
            # generate random weights, we also have a learning rate set to change the weights
            # hidden layer weights
            self.weights1 = np.random.rand(self.input.shape[1], self.hidden_layer_node_count)
            # output layer (final weight, single value)
            self.weights2 = np.random.rand(self.hidden_layer_node_count, 1)

    def init_output(self, y):
        # creates an output the size of the expected result with 0 values
        # eg [1,2,3,4] > [0,0,0,0]
        self.y = y
        self.output = np.zeros(y.shape)

    # sends values through network from input to output
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    # updates the weights based on how accurate the prediction was
    # using activation function sigmoid
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T,
                            self.learn_rate * (self.y - self.output) * sigmoid_derivative(
                                self.output))
        d_weights1 = np.dot(self.input.T,
                            np.dot(self.learn_rate * (self.y - self.output) * sigmoid_derivative(
                                self.output),
                                   self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    # do init_output() first
    def run(self, x):
        self.input = x
        self.init_output(self.y)
        # run.py the input through the network
        self.output = self.feedforward()
        self.backprop()
