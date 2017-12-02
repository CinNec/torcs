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
        breaking_speed_parameter = 0.2
        turn_flag = 0.25

        # print (carstate['speed'])
        # print (carstate['distance'])
        # print (carstate['angle'])
        # print (carstate['sensor_ahead'])
        # print (carstate['steering'])

        # 0 means out of the track or against a wall and it's set to 1
        if carstate['sensor_ahead'] == 0:
            carstate['sensor_ahead'] = 1
        # ACCELERATION
        # the car accelerates if it's current speed is less than the minimal one or the turn is enough far w.r.t. the current speed
        if carstate['speed'] < min_speed or carstate['sensor_ahead'] >= carstate['speed'] / speed_sensor_divisor:
            command.append(1)
        else: 
            command.append(0)
        # BREAKING
        # the car breaks if the current speed is greater than the minimal one, the turn is enough close w.r.t. the speed and the car is not steering too much
        if carstate['speed'] > min_speed and carstate['sensor_ahead'] < carstate['speed'] / speed_sensor_divisor and abs(carstate['steering']) < angle_stop_breaking * steer_amount:
        # the breaking is proportional to the speed
            command.append(carstate['speed'] / breaking_speed_parameter)
        else: command.append(0)
        # STEERING (the minimal steering is always -1 and the maximal is 1)
        # the car steers left if it's at the right of the center of the track by a certain amount and the car is not facing a turn
        if carstate['distance'] < 0.5 - c_dist and carstate['sensor_ahead'] > turn_flag and carstate['steering'] < 1:
            if carstate['steering'] < 0:
                carstate['steering'] = 0
            steering = carstate['steering'] + steer_amount
        # the car steers right if it's at the left of the center of the track by a certain amount and the car is not facing a turn
        if carstate['distance'] > 0.5 + c_dist and carstate['sensor_ahead'] > turn_flag and carstate['steering'] > -1:
            if carstate['steering'] > 0:
                carstate['steering'] = 0
            steering = carstate['steering'] - steer_amount
        #  the car steers left if the angle to the track axis is positive by a certain amount
        if carstate['angle'] > angle and carstate['steering'] < 1:
            if carstate['steering'] < 0:
                carstate['steering'] = 0
            steering = carstate['steering'] + steer_amount
        #  the car steers right if the angle to the track axis is negative by a certain amount
        if carstate['angle'] < -angle and carstate['steering'] > -1:
            if carstate['steering'] > 0:
                carstate['steering'] = 0
            steering = carstate['steering'] - steer_amount
        command.append(steering)

        return command
