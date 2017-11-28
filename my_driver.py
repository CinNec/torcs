from pytocl.driver import Driver
from pytocl.car import State, Command
import pickle
# from feedforward import NeuralNetwork, Layer
import feedforward
import numpy as np
from feedforward import Ndata
#import sklearn
#from sklearn.neural_network import MLPRegressor
import feedforward_split
from ea import EvoAlg

with open("pickled_nn.txt", "rb") as pickle_file:
    nn = pickle.load(pickle_file)

#with open("sklearn_nn.txt", "rb") as pickle_file:
#    mlp = pickle.load(pickle_file)

with open("pickled_nn_accbrk.txt", "rb") as pickle_file:
    nn1 = pickle.load(pickle_file)

with open("pickled_nn_steering.txt", "rb") as pickle_file:
    nn2 = pickle.load(pickle_file)


class MyDriver(Driver):
    # ...
    def __init__(self):
        self.drive_step = 0
        self.steering = 0

    # Override the `drive` method to create your own driver
    def drive(self, carstate: State) -> Command:

        command = Command()
        nn_input = np.array([carstate.speed_x, carstate.distance_from_center, carstate.angle]+list(carstate.distances_from_edge)[0:-1])
        i=0
        while(i <= 20):
            nn_input[i] = (nn_input[i] - Ndata.minarray[i])/(Ndata.maxarray[i]-Ndata.minarray[i])
            i += 1

        # nn_output = nn.forward_propagation(nn_input)
        # command.accelerator= round(nn_output[0])
        # command.brake = round(nn_output[1])
        # command.steering = nn_output[2]

        # mlp_output = mlp.predict([nn_input])[0]
        # # print(mlp_output)
        # command.accelerator= round(mlp_output[0])
        # command.brake = round(mlp_output[1]) if mlp_output[1] > 0.95 else 0
        # command.steering = mlp_output[2]

<<<<<<< HEAD
        nn1_out = nn1.forward_propagation(nn_input)
        rounded_out = np.array([round(nn1_out[0]), round(nn1_out[1])])
#        a = [0, 1, 2, 11, 12, 13, 14, 10]
        nn_input = np.array([1.7 if x> 1.7 else x if x>0 else  0  for x in nn_input])
        print(nn_input)
        nn2_out = nn2.forward_propagation(nn_input)
        command.accelerator= round(nn1_out[0])
        command.brake = round(nn1_out[1])
        command.steering = nn2_out[0]

=======
        # nn1_out = nn1.forward_propagation(nn_input)
        # rounded_out = np.array([round(nn1_out[0]), round(nn1_out[1])])
        # nn_input = np.append(nn_input, rounded_out)
        # nn2_out = nn2.forward_propagation(nn_input)
        # command.accelerator= round(nn1_out[0])
        # command.brake = round(nn1_out[1])
        # command.steering = nn2_out

        alg = EvoAlg()
        ea_input = [nn_input[0], nn_input[1], nn_input[2], nn_input[12], self.steering]
        print(nn_input[0])
        print(nn_input[1])
        print(nn_input[2])
        print(nn_input[12])
        ea_output = alg.ea_output(ea_input)
        self.steering = ea_output[2]
        command.accelerator= ea_output[0]
        command.brake = ea_output[1]
        command.steering = ea_output[2]
>>>>>>> d606518759797e1e26ddd51c31668041ddd61b02

        # GEAR HANDLER

        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        if carstate.rpm < 3500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1
        # OFFTRACK HANDLER

        # reduce acceleration if offtrack
        acceleration = command.accelerator
        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1 and carstate.distances_from_edge[0] == -1:
                # off track, reduced grip:
<<<<<<< HEAD
                acceleration = min(0.4, acceleration)
            command.accelerator = min(acceleration, 1)#
=======
                acceleration = min(0.3, acceleration)
            command.accelerator = min(acceleration, 1)
>>>>>>> d606518759797e1e26ddd51c31668041ddd61b02

        # manually adjust angle
        if carstate.angle > 45:
            command.accelerator = 0.4
            command.steering = 1
        if carstate.angle < -45:
            command.accelerator = 0.4
            command.steering = -1
<<<<<<< HEAD
       # the car is offtrack on the right
        if carstate.distance_from_center < -1:
            if carstate.angle >= -90 and carstate.angle <= -30:
=======

        # the car is offtrack on the right
        if carstate.distance_from_center < -1 and carstate.distances_from_edge[0] == -1:
            if carstate.angle >= -90 and carstate.angle <= -10:
>>>>>>> d606518759797e1e26ddd51c31668041ddd61b02
                command.steering = 0
            elif carstate.angle > -10 and carstate.angle <= 90:
                # steer left
                command.steering = 1
            elif carstate.angle > 90 or carstate.angle < -90:
                # steer right
                command.steering = -1

        # the car is offtrack on the left
        if carstate.distance_from_center > 1:
            if carstate.angle >= 10 and carstate.angle <= 90:
                command.steering = 0
            elif carstate.angle < 10 and carstate.angle >= -90:
                # steer right
                command.steering = -1
            elif carstate.angle > 90 or carstate.angle < -90:
                # steer left
                command.steering = 1

<<<<<<< HEAD
=======
        self.drive_step += 1

>>>>>>> d606518759797e1e26ddd51c31668041ddd61b02
        return command

