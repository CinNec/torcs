import numpy as np
from Normalize_clean import Normalize
import pickle

# Set bias value
BIAS = 1

# Create layer with all weights randomly initialized between -1 and 1
class Layer:
    def __init__(self, neurons, inputs_per_neuron):
        self.weights = 0.01 * (2.0 * np.random.random((inputs_per_neuron, neurons)) - 1.0)
        self.bias = 0.01 * (2.0 * np.random.random((neurons)) - 1.0)
        self.activation = np.zeros(neurons)

class NeuralNetworkClassifier:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    # Calculate the output using the current weights
    def forward_propagation(self, input):
        # Activations of hidden layers (sigmoid)
        for layer in self.layers:
            layer.activation = self.sigmoid(input.dot(layer.weights) + layer.bias * BIAS)
            input = layer.activation
        return input

    # Error function using mean squared error
#    def error(self, y, t, N):
#        return 1/N * np.sum(np.square(y - t))

    def error(self, y, t, N):
        return -np.mean((t * np.log(y) + (1 - t)* np.log(1 - y)))

    # The derivative of the error function
    def derivative_error(self, y, t):
        return y - t

    # Propagate error backwards to find gradients
    def backward_propagation(self, x, y, t, L_rate):
        deltas = []
        layer_delta = self.derivative_error(y, t)
        deltas.insert(0, layer_delta)

        # Iterate backwards through the hidden layers
        for i in reversed(range(len(self.layers) - 1)):
            layer_error = layer_delta.dot(self.layers[i + 1].weights.T)
            layer_delta = layer_error * self.sigmoid_derivative(self.layers[i].activation)
            deltas.insert(0, layer_delta)

        for i in range(len(self.layers)):
             # Separate code for the first hidden layer after input, uses x
            if i == 0:
                self.layers[i].weights -= L_rate * x.T.dot(deltas[i])
                self.layers[i].bias -= L_rate * np.sum(deltas[i], axis = 0)
                continue

            self.layers[i].weights -= L_rate * self.layers[i - 1].activation.T.dot(deltas[i])
            self.layers[i].bias -= L_rate * np.sum(deltas[i], axis = 0)
        return

# Implement mini-batch training function that uses forward and back propagation to train the function
    def train(self, X, Y, L_rate, epochs):
        # Shuffle dataset
        data = np.concatenate((X, Y), axis=1)
        np.random.shuffle(data)
        X = data[:, 0:X.shape[1]]
        Y = data[:, X.shape[1]:]

        Position = 0
        PositionEnd = epochs

        while(PositionEnd < len(X)):
            XBatch = X[Position:PositionEnd]
            YBatch = Y[Position:PositionEnd]
            pred_yBatch = self.forward_propagation(XBatch)

            self.backward_propagation(XBatch, pred_yBatch, YBatch, L_rate)

            Position += epochs
            PositionEnd += epochs
        XBatch = X[Position:]
        YBatch = Y[Position:]
        pred_yBatch = self.forward_propagation(XBatch)

        self.backward_propagation(XBatch, pred_yBatch, YBatch, L_rate)
        return


#%%

class NeuralNetworkRegressor(NeuralNetworkClassifier):
    # Calculate the output using the current weights
    def forward_propagation(self, input):
        # Activations of hidden layers (sigmoid)
        for layer in self.layers[0:-1]:
            layer.activation = self.sigmoid(input.dot(layer.weights) + layer.bias * BIAS)
            input = layer.activation

        # Activation of output layer (identity activation)
        self.layers[-1].activation = input.dot(self.layers[-1].weights) + self.layers[-1].bias * BIAS
        return self.layers[-1].activation

    # Error function using mean squared error
    def error(self, y, t, N):
        return 1 / N * np.sum(np.square(y - t))

    # The derivative of the error function
    def derivative_error(self, y, t):
        return y - t


#%%

# Simple adaptive learning rate to speed up, # Gets stuck in "error went up" sometimes
def adapt_L_rate(L_rate, pre_error, post_error):
#    print("difference:", post_error - pre_error)
    # Increase learning rate by a small amount if cost went down
    if post_error < pre_error:
#        print("Error went down")
        L_rate *= 1.007
    # Decrease learning rate by a large amount if cost went up
    if post_error >= pre_error:
#        print("Error went up")
        L_rate *= 0.990
    return L_rate


#%%


def train_accbrk():
    Ndata = Normalize()
    data = Ndata.data

#    a = [0, 1, 2, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#    X = np.swapaxes([data[:, i] for i in a], 0, 1)
    X = np.array(data[:, 0 : 22])

    Y = np.array(data[:, [22, 23]])

    # Set up layers for neural network
    # Create layers(number of neurons, number of inputs)
    layer1 = Layer(24, 22)
    layer2 = Layer(12, 24)
    layer3 = Layer(2, 12)

    ## Add the layers

    nn1 = NeuralNetworkClassifier()
    nn1.add_layer(layer1)
    nn1.add_layer(layer2)
    nn1.add_layer(layer3)

    # Set parameters
    maxError = 0.0001
    error = 1000000 # Just initialization for adaptable L_rate
    L_rate = 0.005
    epochs = 3000
    batch_size = 64

    for i in range(epochs):
        nn1.train(X,Y,L_rate,batch_size)
        previous_error = error
        pred_y = nn1.forward_propagation(X)
        error = nn1.error(pred_y, Y, len(X))

        if error < maxError:
            print("Converged after %d iterations" % i)
            print("Predicted Y:", pred_y)
            break
        
        correct = np.equal(np.around(pred_y), Y)
        accuracy = np.mean(correct)
        
        L_rate = adapt_L_rate(L_rate, previous_error, error)
        
        if i % 10 == 0:
            print('\nEpoch:', i, "\nLearning rate:", L_rate, "\nTotal error:", error, "\nAccuracy:", accuracy, "\n")

    with open("pickled_nn_accbrk.txt", "wb") as pickle_file:
        pickle.dump(nn1, pickle_file)

    with open('stats_np_accbrk.txt', "w+") as file:
        print("\nEpoch:", epochs, "\nLearning rate:", L_rate, "\nFinal error:", error, "\nAccuracy", accuracy, "\n", file=file)


def train_steer():
    Ndata = Normalize()
    data = Ndata.data
    
    X = np.array(data[:, 0 : 22])
    Y = np.array([data[:, 24]])
    Y.shape = (Y.shape[1], 1)
#
#    a = [0, 1, 2, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#    X = np.swapaxes([data[:, i] for i in a], 0, 1)

    # Create layers(number of neurons, number of inputs)
    layer4 = Layer(24, 22)
    layer5 = Layer(12, 24)
    layer6 = Layer(1, 12)

    ## Add the layers
    nn2 = NeuralNetworkRegressor()
    nn2.add_layer(layer4)
    nn2.add_layer(layer5)
    nn2.add_layer(layer6)

    # Set parameters
    maxError = 0.0001
    error = 1000000 # Just initialization for adaptable L_rate
    L_rate = 0.005
    epochs = 3000
    batch_size = 64

    for i in range(epochs):
        nn2.train(X,Y,L_rate,batch_size)
        previous_error = error
        pred_y = nn2.forward_propagation(X)
        error = nn2.error(pred_y, Y, len(X))

        if error < maxError:
            print("Converged after %d iterations" % i)
            print("Predicted Y:", pred_y)
            break

        average_error = np.mean(np.abs(pred_y - Y))
        
        L_rate = adapt_L_rate(L_rate, previous_error, error)
        
        if i % 10 == 0:
            print('\nEpoch:', i, "\nLearning rate:", L_rate, "\nTotal error:", error, "\nAverage error:", average_error, "\n")

    with open("pickled_nn_steering.txt", "wb") as pickle_file:
        pickle.dump(nn2, pickle_file)

    with open('stats_np_steer.txt', "w+") as file:
        print('\nEpoch:', epochs, "\nLearning rate:", L_rate, "\nFinal error:", error, "\nAverage error:", average_error, "\n", file=file)



#%%
#train_steer()
#train_accbrk()
