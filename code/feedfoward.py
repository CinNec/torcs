import numpy as np
from Normalize import Normalize

# Build model creates a neural network with the specified number of hidden layers and neurons per layer

# Create a weight matrix for a hidden layer 

BIAS = 0

class Layer:
    def __init__(self, neurons, inputs_per_neuron):
        # add 1 for bias 
        self.weights = 2 * np.random.random((neurons, inputs_per_neuron)) - 1
        self.bias = 2 * np.random.random((1, inputs_per_neuron)) - 1
#        self.activation = 
        
class NeuralNetwork: 
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # Calculate the output using the weights
    def forward_propagation(self, input):
        for layer in self.layers:
            print("input:", input)
            print("layer.weights:", layer.weights)
#            input = self.sigmoid(input.dot(layer.weights) + layer.bias * BIAS)
            input = self.sigmoid(input.dot(layer.weights))
            print("input after dot:", input)
        output = input
        return output
    
    # Calculate the error using cross-entropy error
    def error(self, t, y):
        return np.sum(np.square(t - y))
        return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))
    
    # The derivative of the error function
    def derivative_error(self, t, y):
        return y - t
        
    # Propagate error backwards to find gradients
    def backward_propagation(self, t, y):
        # Find gradients for output layer weights
        gradient_output = derivative_error(y - t)
        layer_delta = gradient_output * self.sigmoid_derivative(y)
        
        for hidden_layer in self.layers[0:-1]
            layer_error = layer_delta.dot(hidden_layer.weights)
        return
    
    # Update the weights using batch SGD
    

# decides on an activation function
#def activation(type)
#    if type == sigmoid:
#        return 
    
layer1 = Layer(2, 2)
layer2 = Layer(1, 2)

layer1.weights = np.array([[0.4, 0.2], [0.65, 0.1]])
layer2.weights = np.array([0.8, 0.15])

nn = NeuralNetwork()
nn.add_layer(layer1)
nn.add_layer(layer2)

x = np.array([0.1, 0.35])


y = nn.forward_propagation(x)
print("y:", y)

t = np.array([0.9, 0.35])
y = np.array([0.8, 0.5])
error = nn.error(t, y)
print("error:", error)

# data processing
#data = np.array(Normalize().data)


