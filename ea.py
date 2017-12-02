import json

class EvoAlg():
    def __init__(self):
        self.default_driver = {}
        self.default_driver['min_speed_divisor'] = 1.8
        self.default_driver['very_min_speed'] = 0.07
        self.default_driver['speed_sensor_divisor'] = 0.2
        self.default_driver['angle_stop_breaking'] = 8
        self.default_driver['distance_from_center'] = 0.1
        self.default_driver['max_angle'] = 0.01
        self.default_driver['steer_step'] = 0.007
        pop = []

    def ea_output(self, carstate, driver = {}):
        if len(driver) == 0:
            driver = self.default_driver
        steering = 0
        command = []
        carstate['angle'] = carstate['angle'] / float(180)

        min_speed_divisor = driver['min_speed_divisor']
        very_min_speed = driver['very_min_speed']
        speed_sensor_divisor = driver['speed_sensor_divisor']
        angle_stop_breaking = driver['angle_stop_breaking']
        c_dist = driver['distance_from_center']
        max_angle = driver['max_angle']
        steer_step = driver['steer_step']

        min_speed = max(very_min_speed, carstate['sensor_ahead'] / min_speed_divisor)
        breaking_speed_parameter = 0.2

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
        if carstate['speed'] > min_speed and carstate['sensor_ahead'] < carstate['speed'] / speed_sensor_divisor and abs(carstate['steering']) < angle_stop_breaking * steer_step:
        # the breaking is proportional to the speed
            command.append(carstate['speed'] / breaking_speed_parameter)
        else: command.append(0)
        # STEERING (the minimal steering is always -1 and the maximal is 1)
        # the car steers left if it's at the right of the center of the track by a certain amount
        if carstate['distance'] < 0.5 - c_dist and carstate['angle'] >= -max_angle and carstate['steering'] < 1:
            if carstate['steering'] < 0:
                carstate['steering'] = 0
            steering = carstate['steering'] + steer_step
        # the car steers right if it's at the left of the center of the track by a certain amount
        if carstate['distance'] > 0.5 + c_dist and carstate['angle'] <= max_angle and carstate['steering'] > -1:
            if carstate['steering'] > 0:
                carstate['steering'] = 0
            steering = carstate['steering'] - steer_step
        #  the car steers left if the angle to the track axis is positive by a certain amount
        if carstate['angle'] > max_angle and carstate['steering'] < 1:
            if carstate['steering'] < 0:
                carstate['steering'] = 0
            steering = carstate['steering'] + steer_step
        #  the car steers right if the angle to the track axis is negative by a certain amount
        if carstate['angle'] < -max_angle and carstate['steering'] > -1:
            if carstate['steering'] > 0:
                carstate['steering'] = 0
            steering = carstate['steering'] - steer_step
        command.append(steering)

        return command

    def evaluate(self, speeds, sensors):
        evaluation = 0
        for i, speed in enumerate(speeds):
            evaluation += speed * (15 / math.exp(sensors[i]))
        evaluation = evaluation / float(len(speeds))
        return evaluation

