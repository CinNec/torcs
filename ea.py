import json

class EvoAlg():
    def __init__(self):
        pop = []

    def ea_output(self, carstate, driver):
        steering = 0
        command = []
        carstate['angle'] = carstate['angle'] / float(180)

        c_dist = 0.1
        angle = 0.01
        min_speed_divisor = 1.8
        steer_amount = 0.007
        angle_stop_steering = 10
        very_min_speed = 0.07
        min_speed = max(very_min_speed, carstate['sensor_ahead'] / min_speed_divisor)
        # print (carstate['speed'])
        # print (carstate['distance'])
        # print (carstate['angle'])
        # print (carstate['sensor_ahead'])
        # print (carstate['steering'])
        if carstate['sensor_ahead'] == 0:
            carstate['sensor_ahead'] = 1
        if carstate['speed'] < min_speed or carstate['sensor_ahead'] >= carstate['speed']/0.2:
            command.append(1)
        else: 
            command.append(0)
        if carstate['speed'] > min_speed and carstate['sensor_ahead'] < carstate['speed']/0.2 and abs(carstate['steering']) < angle_stop_steering * steer_amount:
        # if carstate['sensor_ahead'] < 1:
            command.append(carstate['speed']/0.2)
        else: command.append(0)
        # if (carstate['distance'] < -c_dist or (carstate['angle'] > angle and carstate['distance'] <= c_dist and carstate['steering'] >= 0)) and carstate['steering'] < 1:
        # if (carstate['distance'] > c_dist or (carstate['angle'] < -angle and carstate['distance'] >= -c_dist and carstate['steering'] <= 0)) and carstate['steering'] > -1:
        if carstate['distance'] < 0.5 - c_dist and carstate['angle'] >= -angle and carstate['steering'] < 1:
            if carstate['steering'] < 0:
                carstate['steering'] = 0
            steering = carstate['steering'] + steer_amount
        if carstate['distance'] > 0.5 + c_dist and carstate['angle'] <= angle and carstate['steering'] > -1:
            if carstate['steering'] > 0:
                carstate['steering'] = 0
            steering = carstate['steering'] - steer_amount
        if carstate['angle'] > angle and carstate['steering'] < 1:
            if carstate['steering'] < 0:
                carstate['steering'] = 0
            steering = carstate['steering'] + steer_amount
        if carstate['angle'] < -angle and carstate['steering'] > -1:
            if carstate['steering'] > 0:
                carstate['steering'] = 0
            steering = carstate['steering'] - steer_amount
        command.append(steering)
        # print (command)
        return command
