import csv
from os import listdir


class Normalize:

    def __init__(self):
        
        """This code reads all the csv files located in training_data and lists the elements in lists."""
        data = []
        n = 0
        while(n <= 20 ):
            data.append([])
            n += 1
        i = 0
        j = 0
        files = [f for f in listdir("training_data")]
        for file in files:
            with open("training_data/" + file, newline='') as csvfile:
                reader = csv.DictReader(csvfile,delimiter = ",")
                for row in reader:
                    data[0].append(float(row["SPEED"]))
                    data[1].append(float(row["TRACK_POSITION"]))
                    data[2].append(float(row["ANGLE_TO_TRACK_AXIS"]))
                    data[3].append(float(row["TRACK_EDGE_0"]))
                    data[4].append(float(row["TRACK_EDGE_1"]))
                    data[5].append(float(row["TRACK_EDGE_2"]))
                    data[6].append(float(row["TRACK_EDGE_3"]))
                    data[7].append(float(row["TRACK_EDGE_4"]))
                    data[8].append(float(row["TRACK_EDGE_5"]))
                    data[9].append(float(row["TRACK_EDGE_6"]))
                    data[10].append(float(row["TRACK_EDGE_7"]))
                    data[11].append(float(row["TRACK_EDGE_8"]))
                    data[12].append(float(row["TRACK_EDGE_9"]))
                    data[13].append(float(row["TRACK_EDGE_10"]))
                    data[14].append(float(row["TRACK_EDGE_11"]))
                    data[15].append(float(row["TRACK_EDGE_12"]))
                    data[16].append(float(row["TRACK_EDGE_13"]))
                    data[17].append(float(row["TRACK_EDGE_14"]))
                    data[18].append(float(row["TRACK_EDGE_15"]))
                    data[19].append(float(row["TRACK_EDGE_16"]))
                    data[20].append(float(row["TRACK_EDGE_17"]))
                    
        while (i < len(data)):
            maxv = max(data[i])
            minv = min(data[i])
            j = 0
            while(j < len(data[i])):
                print(data[i][j])
                data[i][j] = (data[i][j] - minv)/(maxv-minv)
                j += 1
            i += 1
        print(data[0])