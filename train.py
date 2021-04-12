import numpy as np

from src.neural_net import NeuralNetwork
from src.neural_net_helper import save_weights

# we can use an input matrix here eg.
# currently only using a single feature
# input > person    [smoking,   obesity,    excersize,  diabetic]
# ------------------------------------------------------------------
#         person1   [1,         0,          1,          0]
#         person2   [0,         1,          1,          1]
# output > healthy
# ------------------------------------------------------------------
#       person1     [1]
#       person2     [0]

# INPUT
# input examples for training
# x = np.array(([0, 0, 1],
#               [0, 1, 1],
#               [1, 0, 1],
#               [1, 1, 1]), dtype=float)
x = np.array(([1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0]), dtype=float)

# OUTPUT
# expected output for training
y = np.array(([1],
              [0]), dtype=float)

# init network
NN = NeuralNetwork(x, y, "net1")

# train with expected output x times
for i in range(500):
    # print every 100th result
    if i % 100 == 0:
        # print("for iteration # " + str(i) + "\n")
        # print("Input : \n" + str(x))
        # print("Actual Output: \n" + str(y))
        # print("Predicted Output: \n" + str(NN.feedforward()))
        # print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss

        print("input:" + str(x[0]) + " > output:" + str(y[0]) + " loss(" + str(
            np.mean(np.square(y - NN.feedforward()))) + ")")
        print("\n")

    NN.run(x)

# training complete, save weights
save_weights("net1", "1", NN.weights1)
save_weights("net1", "2", NN.weights2)
