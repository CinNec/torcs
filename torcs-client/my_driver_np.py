from pytocl.driver import Driver
from pytocl.car import State, Command
import pickle
import numpy as np
from Normalize_clean import Normalize
import feedforward_split

Ndata = Normalize()

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
        self.stuck_period = 400
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
        self.test_length = 100
        self.test_best = False
        self.generations = 1

    # Override the `drive` method to create your own driver
    def drive(self, carstate: State) -> Command:
        command = Command()
        nn_input = np.array([carstate.speed_x, carstate.distance_from_center, carstate.angle]+list(carstate.distances_from_edge)[0:])
        i=0
        while(i <= 21):
            nn_input[i] = (nn_input[i] - Ndata.minarray[i])/(Ndata.maxarray[i]-Ndata.minarray[i])
            i += 1
        
        nn_input = np.array([1 if x> 1 else x if x>0 else  0  for x in nn_input])
        
        nn1_out = nn1.forward_propagation(nn_input)
        nn2_out = nn2.forward_propagation(nn_input)
        
        command.accelerator= round(nn1_out[0])
        command.brake = round(nn1_out[1])
        if np.abs(nn2_out[0]) > 0.4:
            command.steering = nn2_out[0] * 1.5
        else:
            command.steering = nn2_out[0]

        #aggressive swarm    
        if min([carstate.opponents[i] for i in [1,-1]]) <50:
            #print("B")
            command.accelerator = 1
        if carstate.opponents[-18] <10:
            print("R17",command.steering)
            command.steering -= 0.01
        elif carstate.opponents[-17] <10:
            print("R16",command.steering)
            command.steering -= 0.01
        elif carstate.opponents[-16] <10:
            print("R15",command.steering)
            command.steering -= 0.01
        elif carstate.opponents[-15] <10:
            print("R14",command.steering)
            command.steering += 0.05
        elif carstate.opponents[-14] <10:
            print("R13",command.steering)
            command.steering += 0.05
        elif carstate.opponents[-13] <10:
            print("R12",command.steering)
            command.steering += 0.1
        elif carstate.opponents[-12] <10:
            print("R11",command.steering)
            command.steering += 0.1
        elif carstate.opponents[-11] <10:
            print("R10",command.steering)
            command.steering -= 0.1
        elif carstate.opponents[-10] <10:
            print("R9",command.steering)
            command.steering -= 0.1
        elif carstate.opponents[-9] <10:
            print("R8",command.steering)
            command.steering -= 0.1
        elif carstate.opponents[-8] <25:
            print("R7",command.steering)
            command.steering -= 0.1
        elif carstate.opponents[-7] <25:
            print("R6",command.steering)
            command.steering -= 0.1
        elif carstate.opponents[-6] <25:
            print("R5",command.steering)
            command.steering -= 0.1
        elif carstate.opponents[-5] <25:
            print("R4",command.steering)
            command.steering += 0.05
        elif carstate.opponents[-4] <25:
            print("R3",command.steering)
            command.steering -= 0.1
            command.accelerator = 1
        elif carstate.opponents[-3] <25:
            print("R2",command.steering)
            command.steering -= 0.1
            command.accelerator = 1
        elif carstate.opponents[-2] <25:
            print("R1",command.steering)
            command.accelerator = 1
        if carstate.opponents[18] <10:
            print("L17",command.steering)            
            command.steering += 0.01
        elif carstate.opponents[17] <10:
            print("L16",command.steering)            
            command.steering += 0.01
        elif carstate.opponents[16] <10:
            print("L15",command.steering)            
            command.steering += 0.01
        elif carstate.opponents[15] <10:
            print("L14",command.steering)            
            command.steering -= 0.05
        elif carstate.opponents[14] <10:
            print("L13",command.steering)            
            command.steering -= 0.05
        elif carstate.opponents[13] <10:
            print("L12",command.steering)            
            command.steering -= 0.1 
        elif carstate.opponents[12] <10:
            print("L11",command.steering)            
            command.steering -= 0.1 
        elif carstate.opponents[11] <10:
            print("L10",command.steering)            
            command.steering += 0.1 
        elif carstate.opponents[10] <10:
            print("L9",command.steering)            
            command.steering += 0.1 
        elif carstate.opponents[9] <10:
            print("L8",command.steering)            
            command.steering += 0.1
        elif carstate.opponents[8] <25:
            print("L7",command.steering)            
            command.steering += 0.05       
        elif carstate.opponents[7] <25:
            print("L6",command.steering)            
            command.steering += 0.1
        elif carstate.opponents[6] <25:
            print("L5",command.steering)
            command.steering += 0.1
        elif carstate.opponents[5] <25:
            print("L4",command.steering)
            command.steering -= 0.05
        elif carstate.opponents[4] <25:
            print("L3",command.steering)
            command.steering += 0.1
            command.accelerator = 1
        elif carstate.opponents[3] <25:
            print("L2",command.steering)
            command.steering += 0.1
            command.accelerator = 1
        elif carstate.opponents[2] <25:
            print("L1",command.steering)
            command.accelerator = 1
            command.steering += 0.1

        # GEAR HANDLER        
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        if carstate.rpm < 3500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1

        # manually adjust angle
        if carstate.angle > 70:
            command.accelerator = 0.6
            command.steering = 0.5
        if carstate.angle < -70:
            command.accelerator = 0.6
            command.steering = -0.5

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
        if (carstate.speed_x < 5 and carstate.speed_x > -5 and command.accelerator > 0.05 and command.gear != -1 and not self.stuck):
            self.stuck_step += 1
            if self.stuck_step > self.stuck_period:
                self.stuck = True
                print ('Stuck')
        else:
            self.stuck_step = 0
        if self.stuck:
            command.gear = -1
            command.steering = -command.steering
            self.stuck_counter += 1
            if self.stuck_counter == self.stuck_recovery:
                self.stuck = False
                self.stuck_counter = 0
                command.gear = 1

        self.drive_step += 1
        return command


