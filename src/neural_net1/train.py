import numpy as np

from src.neural_net1.neural_net import NeuralNetwork
from src.neural_net1.neural_net_helper import save_weights

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
y = np.array(([1], [0]), dtype=float)

# init network
NN = NeuralNetwork(x, y, "net1", True)

# training exmaples
examples = 5000

# train with expected output x times
for i in range(examples):
    # generate random input
    x2 = np.random.randint(low=0, high=1 + 1, size=(5,))
    x = np.array((x2,
                  [0, 0, 0, 0, 0]), dtype=float)

    # output condition
    # if the input has 0's the output will be 0
    # if the input is all 1's the output will be 1
    # we dont use the 2nd output, that is left blank
    if 0 in x2:
        y = np.array(([0], [0]), dtype=float)
    else:
        y = np.array(([1], [0]), dtype=float)

    # print every 100th result
    if i % 10 == 0:
        # print("for iteration # " + str(i) + "\n")
        # print("Input : \n" + str(x))
        # print("Actual Output: \n" + str(y))
        # print("Predicted Output: \n" + str(NN.feedforward()))
        # print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss

        print("#" + str(i) + " - " + "input: " + str(x[0]) + " > output:" + str(
            y[0]) + "loss(" + str(
            np.mean(np.square(y - NN.feedforward()))) + ")")

    NN.run(x, y)

# training complete, save weights
save_weights("net1", "1", NN.weights1)
save_weights("net1", "2", NN.weights2)
