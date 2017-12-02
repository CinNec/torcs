import json

class EvoAlg():
    def __init__(self):
        pop = []

    def ea_output(self, carstate, driver):
        steering = 0
        command = []
        carstate['angle'] = carstate['angle'] / float(180)

        min_speed_divisor = 1.8
        very_min_speed = 0.07
        speed_sensor_divisor = 0.2
        angle_stop_breaking = 10
        c_dist = 0.1
        angle = 0.01
        steer_amount = 0.007

        min_speed = max(very_min_speed, carstate['sensor_ahead'] / min_speed_divisor)

        # print (carstate['speed'])
        # print (carstate['distance'])
        # print (carstate['angle'])
        # print (carstate['sensor_ahead'])
        # print (carstate['steering'])

        # 0 means out of the track or against a wall and it's set to 1
        if carstate['sensor_ahead'] == 0:
            carstate['sensor_ahead'] = 1
        # the car accelerates if it's current speed is less than the minimal one or the turn is enough far w.r.t. the current speed
        if carstate['speed'] < min_speed or carstate['sensor_ahead'] >= carstate['speed'] / speed_sensor_divisor:
            command.append(1)
        else: 
            command.append(0)

        if carstate['speed'] > min_speed and carstate['sensor_ahead'] < carstate['speed'] / speed_sensor_divisor 
                                                                                            and abs(carstate['steering']) < angle_stop_breaking * steer_amount:
            command.append(carstate['speed']/0.2)
        else: command.append(0)
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
