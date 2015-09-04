import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing

def processFile(file_name):
	new_file_name = "nor.arff"
	output = open(new_file_name, "w")
	output.write("@relation 'pzd'\n")
	for i in range(20):
		output.write("@attribute "+str(i)+" numeric\n")
	output.write("@data\n")
	with open(file_name) as f:
		for line in f:
			info = line.split(": ")
			output.write(info[1])
	output.close()

def clustering(file_name):
	with open(file_name) as f:
		matrix_x = []
		matrix_y = []
		for line in f:
			# info = line.split(": ")[1].split(" ")
			info = line.split(" ")
			tmp = []
			for num in info[:-1]:
				n = float(num)
				tmp.append(n)
			matrix_x.append(tmp)
			matrix_y.append(info[-1])
	# Preprocessing
	min_max_scaler = preprocessing.MinMaxScaler()
	train_x = min_max_scaler.fit_transform(matrix_x)
	# KMeans
	kmeans = KMeans(n_clusters=20)
	kmeans.fit(np.asarray(train_x))
	
	length = len(kmeans.labels_)
	for i in range(length):
		print kmeans.labels_[i], matrix_y[i],
			


def main():
	# processFile("./NoEG/p_z_d.txt")
	# clustering("./NoEG/p_z_d.txt")
	clustering("/Users/wyatt/Documents/Code/Gla/Final/Sources/plsa/n_pzd.txt")
	# clustering("/Users/wyatt/Documents/Code/Gla/Final/Sources/plsa/model/p_z_d.txt")

if __name__ == "__main__":
	main()