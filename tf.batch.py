import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time

from Normalize import Normalize
import pickle

INPUT_SIZE    = 21      # 2 bits per timestep
OUTPUT_SIZE   = 1     # 1 bit per timestep

def generate_sequence(data, time_steps, pos):
    pos += 1
    sequence = data[pos - time_steps : pos]
    return sequence

# for 1 output
def generate_batch_1(data, batch_size, time_steps, position):
    batch = []
    start = position
    while position < start + batch_size:
        sequence = generate_sequence(data, time_steps, position)
        batch.append(sequence)
        position += 1
        
    batch = np.array(batch)
    x = batch[:, :, :INPUT_SIZE]
    y = batch[:, time_steps - 1,  21]
    y.shape = (len(y), 1)
    return x, y, position


def generate_batch_2(data, batch_size, time_steps, position):
    batch = []
    start = position
    while position < start + batch_size:
        sequence = generate_sequence(data, time_steps, position)
        batch.append(sequence)
        position += 1
        
    batch = np.array(batch)
    x = batch[:, :, :INPUT_SIZE]
    y = batch[:, 0,  [21 , 22]]
    return x, y, position


TIME_STEPS = 3
BATCH_SIZE = 10
POSITION = TIME_STEPS - 1
counter = 0

#data = np.random.rand(20,23)

Ndata = Normalize()
data = Ndata.data

start = time.time()
while POSITION - 1 + BATCH_SIZE < len(data):
    counter += 1
    x,y, POSITION = generate_batch_2(data, BATCH_SIZE, TIME_STEPS, position=POSITION)

end = time.time()
print("done")
print(end - start)
print("Counter:", counter)

X = data[:, :21]
X.shape = (X.shape[0], 1, X.shape[1])
Y = data[:, [21, 22]]


x,y, POSITION = generate_batch_2(data, BATCH_SIZE, TIME_STEPS, position=13)
#    
#a = np.random.rand(1,21)
#a.shape = (1, 1, a.shape[1])


#data2 = data
#x_train = Ndata.train_data_accbrk
#x_train.shape = (x_train.shape[0], 1, x_train.shape[1])
#
#y_train = Ndata.train_out_accbrk
#
#y_small = np.array([0])

#arr = np.arange(10)
#arr2 = arr[0:8]
#print(arr2)
#print(arr)
#np.random.shuffle(arr)
#print(arr)
#print(arr2)