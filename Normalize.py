import csv
from os import listdir
import numpy as np
import math


class Normalize:

    def __init__(self):
        data = []
        n = 0
        while(n <= 23 ):
            data.append([])
            n += 1
        i = 0
        j = 0
        files = [f for f in listdir("training_data")]
        for file in files:
            with open("training_data/" + file, newline='') as csvfile:
                reader = csv.DictReader(csvfile,delimiter = ",")
                for row in reader:
                    data[0].append( float(row["SPEED"]))
                    data[1].append( float(row["TRACK_POSITION"]))
                    data[2].append( float(row["ANGLE_TO_TRACK_AXIS"]))
                    data[3].append( float(row["TRACK_EDGE_0"]))
                    data[4].append( float(row["TRACK_EDGE_1"]))
                    data[5].append( float(row["TRACK_EDGE_2"]))
                    data[6].append( float(row["TRACK_EDGE_3"]))
                    data[7].append( float(row["TRACK_EDGE_4"]))
                    data[8].append( float(row["TRACK_EDGE_5"]))
                    data[9].append( float(row["TRACK_EDGE_6"]))
                    data[10].append( float(row["TRACK_EDGE_7"]))
                    data[11].append( float(row["TRACK_EDGE_8"]))
                    data[12].append( float(row["TRACK_EDGE_9"]))
                    data[13].append( float(row["TRACK_EDGE_10"]))
                    data[14].append( float(row["TRACK_EDGE_11"]))
                    data[15].append( float(row["TRACK_EDGE_12"]))
                    data[16].append( float(row["TRACK_EDGE_13"]))
                    data[17].append( float(row["TRACK_EDGE_14"]))
                    data[18].append( float(row["TRACK_EDGE_15"]))
                    data[19].append( float(row["TRACK_EDGE_16"]))
                    data[20].append( float(row["TRACK_EDGE_17"]))
                    data[21].append( float(row["ACCELERATION"]))
                    data[22].append( round(float(row["BRAKE"])))
                    data[23].append( float(row["STEERING"]))
        self.minarray = []
        self.maxarray = []
        while (i <= 20):
            maxv = max(data[i])
            self.maxarray.append(maxv)
            minv = min(data[i])
            self.minarray.append(minv)
            j = 0
            while(j < len(data[i])):
                data[i][j] = (data[i][j] - minv)/(maxv-minv)
                j += 1
            i += 1
        while(i < len(data[21])):
            if (data[21][i] == data[22][i] and data[21][i] != 0):
                data[22][i] = 0
            i += 1
#         To get a full data matrix
        npdata = np.array([
                                data[0],
                                data[1],
                                data[2],
                                data[3],
                                data[4],
                                data[5],
                                data[6],
                                data[7],
                                data[8],
                                data[9],
                                data[10],
                                data[11],
                                data[12],
                                data[13],
                                data[14],
                                data[15],
                                data[16],
                                data[17],
                                data[18],
                                data[19],
                                data[20],
                                data[21],
                                data[22],
                                data[23],
                                ])
        
        npdata = np.swapaxes(npdata,0,1)
        self.data = npdata
#        np.random.shuffle(npdata)
        cut = math.floor(0.9 * len(npdata))
        train_data = npdata[:cut]
        test_data = npdata[cut:]
#        train_data = np.swapaxes(train_data,0,1)
#        test_data = np.swapaxes(test_data,0,1)
#        npdata = np.swapaxes(npdata,0,1)
#
#        self.data = npdata
#        self.train_data = np.swapaxes(train_data[0:21],0,1)
#        self.train_out = np.swapaxes(train_data[21:],0,1)
#        self.test_data = np.swapaxes(test_data[0:21],0,1)
#        self.test_out = np.swapaxes(test_data[21:],0,1)
#        
#        self.train_data_accbrk = np.swapaxes(train_data[0:21],0,1)
#        self.train_out_accbrk  = np.swapaxes(np.array([train_data[21]]),0,1)
#        self.test_data_accbrk  = np.swapaxes(test_data[0:21],0,1)
#        self.test_out_accbrk  = np.swapaxes(np.array([test_data[21]]),0,1)
#        
#        self.train_data_steer = np.swapaxes(train_data[0:21],0,1)
#        self.train_out_steer  = np.swapaxes(np.array([train_data[23]]),0,1)
#        self.test_data_steer  = np.swapaxes(test_data[0:21],0,1)
#        self.test_out_steer  = np.swapaxes(np.array([test_data[23]]),0,1)
#        

Normalize()