from pytocl.driver import Driver
from pytocl.car import State, Command


class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    ...
    # print (list(carstate.distances_from_edge))
    # def drive(self, carstate: State) -> Command:
    #     command = Command(...)
    	# nn_input = [
    	# command.accelerator,
    	# command.brake,
    	# command.steering,
    	# carstate.speed_x,
    	# carstate.distance_from_center,
    	# carstate.angle
    	# ] + list(carstate.distances_from_edge) + [
    	# command.acceleration = nn_output[0],
    	# command.brake = nn_output[1],
    	# command.steering = nn_output[2]
    	# ]
    #     return command
    	
