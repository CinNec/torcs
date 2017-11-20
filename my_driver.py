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

last_dfc = 0

class MyDriver(Driver):

    # Override the `drive` method to create your own driver
    def drive(self, carstate: State) -> Command:
        print (carstate.angle)
        command = Command()
        nn_input = np.array([carstate.speed_x, carstate.distance_from_center, carstate.angle]+list(carstate.distances_from_edge)[0:-1])
        i=0
        while(i <= 20):
            nn_input[i] = nn_input[i] = (nn_input[i] - Ndata.minarray[i])/(Ndata.maxarray[i]-Ndata.minarray[i])
            i += 1
        nn_output = nn.forward_propagation(nn_input)

        command.accelerator= round(nn_output[0])
        command.brake = round(nn_output[1])
        command.steering = nn_output[2]

        acceleration = command.accelerator

        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1:
                # off track, reduced grip:
                acceleration = min(0.2, acceleration)

            command.accelerator = min(acceleration, 1)

        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1


        if carstate.rpm < 4000:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        # if abs(carstate.distance_from_center) >= 1:
        #     command.brake = 0
        #     if abs(carstate.distance_from_center) >= abs(last_dfc):
        #         command.steering = -1
        #     else:
        #         command.steering = 0
        # last_dfc = carstate.distance_from_center

        return command
