from pytocl.driver import Driver
from pytocl.car import State, Command
import pickle
# from feedforward import NeuralNetwork, Layer
import feedforward
import numpy as np
from feedforward import Ndata


# class NeuralNetwork:
#     def __init__():
#         print('thing')

# print(feedforward.NeuralNetwork)
with open("pickled_nn.txt", "rb") as pickle_file:
    nn = pickle.load(pickle_file)

class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    def drive(self, carstate: State) -> Command:
        command = Command()
        nn_input = np.array([carstate.speed_x, carstate.distance_from_center, carstate.angle]+list(carstate.distances_from_edge)[0:-1])
        i=0
        while(i <= 20):
            nn_input[i] = nn_input[i] = (nn_input[i] - Ndata.minarray[i])/(Ndata.maxarray[i]-Ndata.minarray[i])
            i += 1
        nn_output = nn.forward_propagation(nn_input)
        print(nn_output[1])
        command.accelerator= round(nn_output[1])
        command.brake = round(nn_output[0])
        command.steering = nn_output[2]
        command.gear = carstate.gear +1
        return command