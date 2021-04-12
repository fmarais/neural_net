# run.py after training
import numpy as np

from src.neural_net import NeuralNetwork

x = np.array(([1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0]), dtype=float)

y = np.array(([1, 0]), dtype=float)

NN = NeuralNetwork(x, y, "net1", False)
NN.run(x, y)
output_x = NN.feedforward()

print("input:" + str(x[0]) + " > output:" + str(y[0]) + " - loss(" + str(
    np.mean(np.square(y - NN.feedforward()))) + ")")
