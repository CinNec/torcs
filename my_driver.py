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
        self.stuck_step = 0
        self.stuck_counter = 0
        self.stuck_recovery = 200
        self.stuck_period = 300
        self.stuck = False

        # fixed EA variables
        self.tests = 0
        self.speeds = []
        self.sensors = []
        self.steerings = []
        self.drivers = []
        self.driver = 0
        self.test_step = 0
        self.drive_test = False
        self.min_speed_change = 0.1
        # changeable EA variables
        self.pop_size = 10 # must be 10 or more
        self.test_length = 20
        self.test_best = False
        self.generations = 0

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

        # nn1_out = nn1.forward_propagation(nn_input)
        # rounded_out = np.array([round(nn1_out[0]), round(nn1_out[1])])
        # a = [0, 1, 2, 11, 12, 13, 14, 10]
        # nn_input = np.array([1 if x> 1 else x if x>0 else  0  for x in nn_input])
        # print(nn_input)
        # nn2_out = nn2.forward_propagation(nn_input)
        # command.accelerator= round(nn1_out[0])
        # command.brake = round(nn1_out[1])
        # command.steering = nn2_out[0]

        # nn1_out = nn1.forward_propagation(nn_input)
        # rounded_out = np.array([round(nn1_out[0]), round(nn1_out[1])])
        # nn_input = np.append(nn_input, rounded_out)
        # nn2_out = nn2.forward_propagation(nn_input)
        # command.accelerator= round(nn1_out[0])
        # command.brake = round(nn1_out[1])
        # command.steering = nn2_out

        # EVOLUTIONARY ALGORITHM

        EA = EvoAlg()

        ea_input = {}
        ea_input['speed'] = nn_input[0]
        ea_input['distance'] = nn_input[1]
        ea_input['angle'] = nn_input[2] / float(180)
        ea_input['sensor_ahead'] = nn_input[12]
        ea_input['steering'] = self.steering
        # 0 means out of the track or against a wall and it's set to 1
        if ea_input['sensor_ahead'] == 0:
            ea_input['sensor_ahead'] = 1

        # if self.drive_step % 100 == 0:
        #     print(carstate.opponents)

        if self.drive_step == 0 or (self.tests % self.pop_size == 0 and not self.test_best):
            self.drivers = EA.load_drivers()
            if len(self.drivers) != self.pop_size:
                self.drivers = EA.create_population(self.pop_size)
                print('population created')

        if ea_input['speed'] > self.min_speed_change and self.test_step == 0 and not self.test_best:
            self.drive_test = True

        if self.drive_test:
            driver = self.drivers[self.driver]
            self.speeds.append(ea_input['speed'])
            self.sensors.append(ea_input['sensor_ahead'])
            self.steerings.append(self.steering)
            self.test_step += 1
            if self.test_step == self.test_length:
                evaluation = EA.evaluate(self.speeds, self.sensors, self.steerings)
                print(evaluation)
                self.drivers[self.driver]['evaluation'] = evaluation
                self.speeds = []
                self.sensors = []
                self.steerings = []
                self.drive_test = False
                self.driver = (self.driver + 1) % len(self.drivers)
                self.test_step = 0
                self.tests += 1
                if self.tests % self.pop_size == 0:
                    EA.save_drivers(self.drivers)
                    print('drivers saved')
                    if self.tests <= self.pop_size * self.generations:
                        self.drivers = EA.next_gen()
        elif not self.test_best:
            driver = {}
        else:
            driver = sorted(self.drivers, key=lambda x: x['evaluation'], reverse=True)[0]
            if self.drive_step == 0:
                print('min_speed_divisor: ' + str(driver['min_speed_divisor']))
                print('very_min_speed: ' + str(driver['very_min_speed']))
                print('speed_sensor_divisor: ' + str(driver['speed_sensor_divisor']))
                print('breaking_speed_parameter: ' + str(driver['breaking_speed_parameter']))
                print('angle_stop_breaking: ' + str(driver['angle_stop_breaking']))
                print('distance_from_center: ' + str(driver['distance_from_center']))
                print('max_angle: ' + str(driver['max_angle']))
                print('steer_step: ' + str(driver['steer_step']))
                print('evaluation: ' + str(driver['evaluation']))

        ea_output = EA.ea_output(ea_input, driver)
        self.steering = ea_output[2]
        command.accelerator= ea_output[0]
        command.brake = ea_output[1]
        command.steering = ea_output[2]

        # GEAR HANDLER

        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        if carstate.rpm < 3500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1

        # # manually adjust angle
        # if carstate.angle > 45:
        #     command.accelerator = 0.6
        #     command.steering = 0.5
        # if carstate.angle < -45:
        #     command.accelerator = 0.6
        #     command.steering = -0.5

        # OFFTRACK HANDLER
        # reduce acceleration if offtrack
        acceleration = command.accelerator
        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1 and carstate.distances_from_edge[0] == -1:
                # off track, reduced grip:
                acceleration = min(0.4, acceleration)
            command.accelerator = min(acceleration, 1)

        # the car is offtrack on the right
        if carstate.distance_from_center < -0.5 and carstate.distances_from_edge[0] == -1:
            if carstate.angle >= -30 and carstate.angle <= -15:
                command.steering = 0
            elif carstate.angle > -15 and carstate.angle <= 120:
                # steer left
                command.steering = 0.8
            elif carstate.angle > 120 or carstate.angle < -30:
                # steer right
                command.steering = -0.8

        # the car is offtrack on the left
        if carstate.distance_from_center > 1.5 and carstate.distances_from_edge[0] == -1:
            if carstate.angle >= 15 and carstate.angle <= 30:
                command.steering = 0
            elif carstate.angle >= -120 and carstate.angle < 15:
                # steer right
                command.steering = -0.8
            elif carstate.angle > 30 or carstate.angle < -120:
                # steer left
                command.steering = 0.8

        # STUCK CAR HANDLER
        if (nn_input[0] < 0.001 and nn_input[0] > -0.001 and command.accelerator > 0.05 and command.gear != -1 and not self.stuck):
            self.stuck_step += 1
            if self.stuck_step > self.stuck_period:
                self.stuck = True
        else:
            self.stuck_step = 0
        if self.stuck:
            print ('Stuck')
            command.gear = -1
            command.steering = -command.steering
            self.stuck_counter += 1
            if self.stuck_counter == self.stuck_recovery:
                self.stuck = False
                self.stuck_counter = 0
                command.gear = 1

        self.drive_step += 1
        return command


