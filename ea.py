import json

class EvoAlg():
    def __init__(self):
        pop = []

    def ea_output(self, carstate, driver):
        steering = 0
        command = []
        carstate[2] = carstate[2] / float(180)

        c_dist = 0.1
        angle = 0.01
        min_speed = carstate[3]/4
        steer_amount = 0.007
        angle_stop_steering = 10
        # print (carstate[0])
        # print (carstate[1])
        print (carstate[2])
        # print (carstate[3])
        # print (carstate[4])
        if carstate[3] == 0:
            carstate[3] = 1
        if carstate[0] < min_speed or carstate[3] >= carstate[0]/0.2:
            command.append(1)
        else: 
            command.append(0)
        if carstate[0] > min_speed and carstate[3] < carstate[0]/0.2 and abs(carstate[4]) < angle_stop_steering * steer_amount:
        # if carstate[3] < 1:
            command.append(carstate[0]/0.2)
        else: command.append(0)
        # if (carstate[1] < -c_dist or (carstate[2] > angle and carstate[1] <= c_dist and carstate[4] >= 0)) and carstate[4] < 1:
        # if (carstate[1] > c_dist or (carstate[2] < -angle and carstate[1] >= -c_dist and carstate[4] <= 0)) and carstate[4] > -1:
        if carstate[1] < 0.5 - c_dist and carstate[2] >= -angle and carstate[4] < 1:
            if carstate[4] < 0:
                carstate[4] = 0
            steering = carstate[4] + steer_amount
        if carstate[1] > 0.5 + c_dist and carstate[2] <= angle and carstate[4] > -1:
            if carstate[4] > 0:
                carstate[4] = 0
            steering = carstate[4] - steer_amount
        if carstate[2] > angle and carstate[4] < 1:
            if carstate[4] < 0:
                carstate[4] = 0
            steering = carstate[4] + steer_amount
        if carstate[2] < -angle and carstate[4] > -1:
            if carstate[4] > 0:
                carstate[4] = 0
            steering = carstate[4] - steer_amount
        command.append(steering)
        # print (command)
        return command
