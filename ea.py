import json

class EvoAlg():
    def __init__(self):
        pop = []

    def ea_output(self, carstate):
        steering = 0
        command = []
        c_dist = 0.5
        angle = 0.01
        carstate[2] = carstate[2] / float(180)
        print (carstate[0])
        # print (carstate[1])
        # print (carstate[2])
        # print (carstate[3])
        # print (carstate[4])
        if carstate[3] == 0:
            carstate[3] = 1
        if carstate[0] < 0.04 or carstate[3] == 1:
            command.append(0.8)
        else: 
            command.append(0)
        if carstate[0] > 0.04 and carstate[3] < 1:
        # if carstate[3] < 1:
            command.append(1)
        else: command.append(0)
        # if (carstate[1] < -c_dist or (carstate[2] > angle and carstate[1] <= c_dist and carstate[4] >= 0)) and carstate[4] < 1:
        # if (carstate[1] > c_dist or (carstate[2] < -angle and carstate[1] >= -c_dist and carstate[4] <= 0)) and carstate[4] > -1:
        if carstate[2] > angle and carstate[4] < 1:
            if carstate[4] < 0:
                carstate[4] = 0
            steering = carstate[4] + 0.008
        if carstate[2] < -angle and carstate[4] > -1:
            if carstate[4] > 0:
                carstate[4] = 0
            steering = carstate[4] - 0.008
        if carstate[1] < -c_dist and carstate[4] < 1:
            if carstate[4] < 0:
                carstate[4] = 0
            steering = carstate[4] + 0.008
        if carstate[1] > c_dist and carstate[4] > -1:
            if carstate[4] > 0:
                carstate[4] = 0
            steering = carstate[4] - 0.008
        command.append(steering)
        # print (command)
        return command
