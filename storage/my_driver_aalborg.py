from pytocl.driver import Driver
from pytocl.car import State, Command
import pickle
# from feedforward import NeuralNetwork, Layer
import feedforward
from feedforward import Ndata
import csv
from os import listdir
import numpy as np



def output():
    data = []
    n = 0
    while(n <= 2 ):
        data.append([])
        n += 1

    files = [f for f in listdir("training_data")]
    for file in files:
        with open("training_data/" + file, newline='') as csvfile:
            reader = csv.DictReader(csvfile,delimiter = ",")
            for row in reader:
                data[0].append( float(row["ACCELERATION"]))
                data[1].append( float(row["BRAKE"]))
                data[2].append( float(row["STEERING"]))

    # To get a full data matrix
    output_data = np.array([
								data[0],
								data[1],
								data[2],
								])
    output_data = np.swapaxes(output_data,0,1)
    return output_data

output = output()

class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    
    i = 0
    
    def drive(self, carstate: State) -> Command:
        command = Command()
        
        command.accelerator= output[self.i][0]
        command.brake = output[self.i][1]
        command.steering = output[self.i][2]
     
        acceleration = command.accelerator 
        
        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1:
                # off track, reduced grip:
                acceleration = min(0.4, acceleration)

            command.accelerator = min(acceleration, 1)

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1



        if carstate.rpm < 4000:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        self.i += 1 
        return command


