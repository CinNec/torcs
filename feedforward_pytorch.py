from Normalize import Normalize
import pickle
import tensorflow as tf

# Build model creates a neural network with the specified number of hidden layers and neurons per layer

# Create a weight matrix for a hidden layer 


# Set bias value
BIAS = 1

datype = tf.float64 # Uncomment this to run on GPU

# Create layer with all weights randomly initialized between -1 and 1
class Layer:
    def __init__(self, neurons, inputs_per_neuron):
        self.weights = 2 * tf.cast(tf.random_uniform((inputs_per_neuron, neurons)), dtype=datype)-1
        self.bias = 2 * tf.cast(tf.random_uniform((1,neurons)),dtype=datype) - 1
        self.activation = tf.cast(tf.zeros(neurons),dtype=datype)
        
        
class NeuralNetwork: 
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def sigmoid(self, x):
        return 1 / (1 + tf.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # Calculate the output using the current weights
    def forward_propagation(self, input):
        # Activations of hidden layers (sigmoid)
        for layer in self.layers[0:-1]:
            layer.activation = self.sigmoid(tf.tensordot(input,layer.weights,1) + layer.bias * BIAS)
#            layer.activation = self.sigmoid(input.dot(layer.weights))
        
        # Activation of output layer (identity activation)
        self.layers[-1].activation = tf.tensordot(layer.activation,self.layers[-1].weights,1) + self.layers[-1].bias * BIAS
        return (self.layers[-1].activation)
        
    
    # Error function using mean squared error
    def error(self, y, t, N):
        return (1/N * tf.reduce_sum(tf.square(y - t)).eval())
    
    # The derivative of the error function
    def derivative_error(self, y, t):
        return (tf.subtract(y,t))
        
    # Propagate error backwards to find gradients
    def backward_propagation(self, x, y, t, L_rate):
        # Find delta values for output layer weights
        layer_delta = self.derivative_error(y, t)
        
        # Set output delta as the initial delta for the loop
        output_delta = layer_delta
        # Iterate backwards through the hidden layers
        for i in reversed(range(len(self.layers) - 1)): 
            layer_error = tf.tensordot(layer_delta,tf.transpose(self.layers[i + 1].weights),1)
            layer_delta = layer_error * self.sigmoid_derivative(self.layers[i].activation)
#           
            # Separate code for the first hidden layer after input, uses x
            # to calculate weight gradients, exit loop after.
            if i == 0:
                self.layers[i].weights -= L_rate * tf.tensordot(tf.transpose(x),layer_delta,1)
                self.layers[i].bias -= L_rate * tf.reduce_sum(layer_delta,0)
                break
            
            # Use delta to update weights and bias
            self.layers[i].weights -= L_rate * tf.tensordot(tf.transpose(self.layers[i - 1].activation),layer_delta,1)
            self.layers[i].bias -= L_rate * tf.reduce_sum(layer_delta,0)
            
        
        # Update weights and bias of output layer
        self.layers[-1].weights -= L_rate * tf.tensordot(tf.transpose(self.layers[-2].activation),output_delta,1)
        self.layers[-1].bias -= L_rate * tf.reduce_sum(output_delta,0)
        
        return
    
# Implement mini-batch training function that uses forward and back propagation to train the function      
    def train(self, X, Y, L_rate, epochs):
        Position = 0
        PositionEnd = epochs
        self.sess = tf.Session()
        while(PositionEnd < X.get_shape().as_list()[0]):
            print(Position)
            self.XBatch = X[Position:PositionEnd]
            self.YBatch = Y[Position:PositionEnd]
            self.pred_yBatch = nn.forward_propagation(self.XBatch)
            
            nn.backward_propagation(self.XBatch, self.pred_yBatch, self.YBatch, L_rate)

            Position += epochs
            PositionEnd += epochs
        XBatch = X[Position:]
        YBatch = Y[Position:]
        pred_yBatch = nn.forward_propagation(XBatch)

        nn.backward_propagation(XBatch, pred_yBatch, YBatch, L_rate)
        self.pred_y = nn.forward_propagation(X)
        error = nn.error(self.pred_y, Y, X.get_shape()[0].value)
        self.sess.close()
#        print("error:", error)
        
        
        #print("error:", error)
        if( error < maxError):
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
X = tf.convert_to_tensor(X, datype)

# Normalize the inputs

#X = preprocessing.scale(X)
# X = preprocessing.scale(X)


# Set goals
# Each row is (y1)
Y = Ndata.outputdata
Y = tf.convert_to_tensor(Y, datype)

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
        L_rate *= 1.005
    # Decrease learning rate by a large amount if cost went up
    if post_error >= pre_error:
#        print("Error went up")
        L_rate *= 0.85
    return L_rate
 



def main():
#    np.random.seed(12)

#    # Create layers(number of neurons, number of inputs)
#    # Three hidden layer network
#    layer1 = Layer(30, 21)
#    layer2 = Layer(30, 30)
#    layer3 = Layer(3, 30)
#    #
#    ## Add the layers
#    ##
#    global nn
#    nn = NeuralNetwork()
#    nn.add_layer(layer1)
#    nn.add_layer(layer2)
#    nn.add_layer(layer3)

#
#     One hidden layer
#     Create layers(number of neurons, number of inputs)
    layer1 = Layer(14, 21)
    layer2 = Layer(3, 14)
    
    global nn
    with tf.Graph().as_default():
        nn = NeuralNetwork()
    nn.add_layer(layer1)
    nn.add_layer(layer2)

    global maxError
    maxError = 0.0001
    error = 1000000
    L_rate = 0.005
    # Quick function to train a neural network until maxError is reached.
    for i in range(500):
        
        print("\nIteration:", i)        
        if(nn.train(X,Y,L_rate,128)):
            break
                
        previous_error = error
        pred_y = nn.forward_propagation(X)
        error = nn.error(pred_y, Y, X.get_shape()[0].value)
        print("error:", error)
        L_rate = adapt_L_rate(L_rate, previous_error, error)
        print("L_rate:", L_rate)
    error = nn.error(pred_y, Y, X.get_shape()[0].value)
    print("error:", error)
    with open("pickled_nn.txt", "wb") as pickle_file:
        pickle.dump(nn, pickle_file)


    

