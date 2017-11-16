from pytocl.driver import Driver
from pytocl.car import State, Command
import pickle
from feedforward import NeuralNetwork

with open("pickled_nn.txt", "rb") as pickle_file:
    nn = pickle.load(pickle_file)

class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    def drive(self, carstate: State) -> Command:
        command = Command()
        nn_input = [carstate.speed_x, carstate.distance_from_center, carstate.angle]+list(carstate.distances_from_edge)
        nn_output = nn.forward_propagation(nn_input)
        command.acceleration = nn_output[0]
        command.brake = nn_output[1]
        command.steering = nn_output[2]
        return command
