import json
import math
import random

class EvoAlg():
    def __init__(self):
        self.default_driver = {}
        self.default_driver['min_speed_divisor'] = 1.8
        self.default_driver['very_min_speed'] = 0.07
        self.default_driver['speed_sensor_divisor'] = 0.2
        self.default_driver['breaking_speed_parameter'] = 0.2
        self.default_driver['angle_stop_breaking'] = 8
        self.default_driver['distance_from_center'] = 0.1
        self.default_driver['max_angle'] = 0.01
        self.default_driver['steer_step'] = 0.007

    def create_population(self, size):
        population = []
        for i in range(size):
            driver = {}
            driver['min_speed_divisor'] = random.uniform(1,3)
            driver['very_min_speed'] = random.uniform(0.05,0.15)
            driver['speed_sensor_divisor'] = random.uniform(0.15,0.25)
            driver['breaking_speed_parameter'] = random.uniform(0.15,0.25)
            driver['angle_stop_breaking'] = random.uniform(3,12)
            driver['distance_from_center'] = random.uniform(0.05,0.25)
            driver['max_angle'] = random.uniform(0.005,0.02)
            driver['steer_step'] = random.uniform(0.003,0.012)
            driver['evaluation'] = 0
            population.append(driver)
        return population


    def ea_output(self, carstate, driver = {}):
        if len(driver) == 0:
            driver = self.default_driver
        steering = 0

        command = []

        min_speed_divisor = driver['min_speed_divisor']
        very_min_speed = driver['very_min_speed']
        speed_sensor_divisor = driver['speed_sensor_divisor']
        breaking_speed_parameter = driver['breaking_speed_parameter']
        angle_stop_breaking = driver['angle_stop_breaking']
        c_dist = driver['distance_from_center']
        max_angle = driver['max_angle']
        steer_step = driver['steer_step']

        min_speed = max(very_min_speed, carstate['sensor_ahead'] / min_speed_divisor)

        # print (carstate['speed'])
        # print (carstate['distance'])
        # print (carstate['angle'])
        # print (carstate['sensor_ahead'])
        # print (carstate['steering'])

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

    def evaluate(self, speeds, sensors, steerings):
        evaluation = 0
        for i, speed in enumerate(speeds):
            # evaluation += speed * (15 / math.exp(sensors[i]))
            evaluation += 10 * speed * (-0.5 * sensors[i] + 1) - 0.2 * abs(steerings[i])
        if evaluation > 0:
            evaluation = evaluation / float(len(speeds))
        return evaluation

    def next_gen(self):
        drivers = load_drivers()

    def generate_offspring(self, drivers):
        k = len(drivers) / 3
        t_number = len(drivers) / 2
        remaining_drivers = drivers[:]
        mating_pool = []
        for t in range(t_number):
            tournament = []
            t_drivers = remaining_drivers[:]
            for i in range(k):
                tournament.append(random.choice(t_drivers))
                t_drivers.remove(tournament[i])
            tournament.sort(key=lambda x: x.evaluation, reverse=True)
            remaining_drivers.remove(tournament[0])
            mating_pool.append(tournament[0])
            tournament = []
        for driver in mating_pool:
            print(driver.evaluation)

    def load_drivers(self):
        return json.load(open('drivers.json','r'))

    def save_drivers(self, drivers):
        json.dump(drivers, open('drivers.json', 'w'))

