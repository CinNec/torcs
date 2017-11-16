import numpy as np
from Normalize import Normalize
import pickle

# Build model creates a neural network with the specified number of hidden layers and neurons per layer

# Create a weight matrix for a hidden layer 


# Set bias value
BIAS = 1

# Create layer with all weights randomly initialized between -1 and 1
class Layer:
    def __init__(self, neurons, inputs_per_neuron):
        self.weights = 2 * np.random.random((inputs_per_neuron, neurons)) - 1
        self.bias = 2 * np.random.random((neurons)) - 1
        self.activation = np.zeros(neurons)
        
class NeuralNetwork: 
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # Calculate the output using the current weights
    def forward_propagation(self, input):
        # Activations of hidden layers (sigmoid)
        for layer in self.layers[0:-1]:
            layer.activation = self.sigmoid(input.dot(layer.weights) + layer.bias * BIAS)
#            layer.activation = self.sigmoid(input.dot(layer.weights))
            input = layer.activation
        
        # Activation of output layer (identity activation)
        self.layers[-1].activation = input.dot(self.layers[-1].weights) + self.layers[-1].bias * BIAS
        output = self.layers[-1].activation
        return output
    
    # Error function using mean squared error
    def error(self, y, t, N):
        return 1/N * np.sum(np.square(y - t))
    
    # The derivative of the error function
    def derivative_error(self, y, t):
        return y - t
        
    # Propagate error backwards to find gradients
    def backward_propagation(self, x, y, t, L_rate):
        # Find delta values for output layer weights
        output_error = self.derivative_error(y, t)
        output_delta = output_error
        
        # Set output delta as the initial delta for the loop
        layer_delta = output_delta
        
        # Iterate backwards through the hidden layers
        for i in reversed(range(len(self.layers) - 1)): 
            layer_error = layer_delta.dot(self.layers[i + 1].weights.T)
            layer_delta = layer_error * self.sigmoid_derivative(self.layers[i].activation)
#           
            # Separate code for the first hidden layer after input, uses x
            # to calculate weight gradients, exit loop after.
            if i == 0:
                self.layers[i].weights -= L_rate * x.T.dot(layer_delta)
                self.layers[i].bias -= L_rate * np.sum(layer_delta, axis = 0)
                break
            
            # Use delta to update weights and bias
            self.layers[i].weights -= L_rate * self.layers[i - 1].activation.T.dot(layer_delta)
            self.layers[i].bias -= L_rate * np.sum(layer_delta, axis = 0)

        
        # Update weights and bias of output layer
        self.layers[-1].weights -= L_rate * self.layers[-2].activation.T.dot(output_delta)
        self.layers[-1].bias -= L_rate * np.sum(output_delta, axis = 0)
        return
    
# Implement mini-batch training function that uses forward and back propagation to train the function      
    def train(self, X, Y, L_rate, epochs):
        Position = 0
        PositionEnd = epochs
        while(PositionEnd < len(X)):
            XBatch = X[Position:PositionEnd]
            YBatch = Y[Position:PositionEnd]
            pred_yBatch = nn.forward_propagation(XBatch)

            nn.backward_propagation(XBatch, pred_yBatch, YBatch, L_rate)
            
            error = nn.error(pred_yBatch, YBatch, len(XBatch))
    
            #print("error:", error)
            if error < maxError:
                print("Converged after %d iterations" % i)
                print("Predicted Y:", pred_yBatch)
                return True
            
            Position += epochs
            PositionEnd += epochs
        XBatch = X[Position:-1]
        YBatch = Y[Position:-1]
        pred_yBatch = nn.forward_propagation(XBatch)

        nn.backward_propagation(XBatch, pred_yBatch, YBatch, L_rate)
        
        error = nn.error(pred_yBatch, YBatch, len(XBatch))
#        print("error:", error)
        
        
        #print("error:", error)
        if error < maxError:
            print("Converged after %d iterations" % i)
            print("Predicted Y:", pred_yBatch)
            return True
        return False
            

    
    
        
#%%   

############# TEST DATASET ###############
# from sklearn import preprocessing

# Set inputs
# Each row is (x1, x2)
Ndata = Normalize()
X = Ndata.inputdata

# Normalize the inputs

#X = preprocessing.scale(X)
# X = preprocessing.scale(X)


# Set goals
# Each row is (y1)
Y = Ndata.outputdata

#Y = np.array([
#                [0],
#                [1],
#                [0],
#                [1],
#                [1],
#                [0],
#                [0],
#                [1],
#                [0],
#                [1]
#                ])

#%%

# Grabbing the actual dataset


#%%

# Simple adaptive learning rate to speed up, # Gets stuck in "error went up" sometimes
def adapt_L_rate(L_rate, pre_error, post_error):
#    print("difference:", post_error - pre_error)
    # Increase learning rate by a small amount if cost went down
    if post_error < pre_error:
#        print("Error went down")
        L_rate *= 1.04
    # Decrease learning rate by a large amount if cost went up
    if post_error >= pre_error:
#        print("Error went up")
        L_rate *= 0.93
    return L_rate
 

# Create layers(number of neurons, number of inputs)
# Three hidden layer network
#layer1 = Layer(20, 21)
#layer2 = Layer(15, 20)
#layer3 = Layer(10, 15)
#layer4 = Layer(3, 10)
##
### Add the layers
###
#nn = NeuralNetwork()
#nn.add_layer(layer1)
#nn.add_layer(layer2)
#nn.add_layer(layer3)
#nn.add_layer(layer4)

def main():
    # One hidden layer
    # Create layers(number of neurons, number of inputs)
    layer1 = Layer(30, 21)
    layer2 = Layer(3, 30)

    global nn
    nn = NeuralNetwork()
    nn.add_layer(layer1)
    nn.add_layer(layer2)



    # Make random numbers predictable
    #np.random.seed(1)
     
    global maxError
    maxError = 0.0001
    error = 1000000
    L_rate = 0.01

    # Quick function to train a neural network until maxError is reached.
    for i in range(1000):
        
        print("\nIteration:", i)
        if(nn.train(X,Y,L_rate,64)):
            break
        previous_error = error
        pred_y = nn.forward_propagation(X)
        error = nn.error(pred_y, Y, len(X))
        print("error:", error)
        L_rate = adapt_L_rate(L_rate, previous_error, error)
        print("L_rate:", L_rate)

    error = nn.error(pred_y, Y, len(X))
    print("error:", error)

    with open("pickled_nn.txt", "wb") as pickle_file:
        pickle.dump(nn, pickle_file)


    

