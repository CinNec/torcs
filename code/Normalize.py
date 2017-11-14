import csv
from os import listdir

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
					data[22].append( float(row["BRAKE"]))
					data[23].append( float(row["STEERING"]))

		while (i < len(data)):
			maxv = max(data[i])
			minv = min(data[i])
			j = 0
			while(j < len(data[i])):
				data[i][j] = (data[i][j] - minv)/(maxv-minv)
				j += 1
			i += 1

		# To get a full data matrix
		self.data = data

		self.speed = data[0]
		self.track_position = data[1]
		self.angle_to_track_axis = data[2]
		self.track_edge_0 = data[3]
		self.track_edge_1 = data[4]
		self.track_edge_2 = data[5]
		self.track_edge_3 = data[6]
		self.track_edge_4 = data[7]
		self.track_edge_5 = data[8]
		self.track_edge_6 = data[9]
		self.track_edge_7 = data[10]
		self.track_edge_8 = data[11]
		self.track_edge_9 = data[12]
		self.track_edge_10 = data[13]
		self.track_edge_11 = data[14]
		self.track_edge_12 = data[15]
		self.track_edge_13 = data[16]
		self.track_edge_14 = data[17]
		self.track_edge_15 = data[18]
		self.track_edge_16 = data[19]
		self.track_edge_16 = data[20]
