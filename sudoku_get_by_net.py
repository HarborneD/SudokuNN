from PIL import Image
import numpy as np
import os
from PyNet import NeuralNetwork
import sudoku

def produce_training_data(values_list,image_names):

	for image_index in range(0,len(values_list)):
		values = values_list[image_index]

		outputs = [0] * 81

		for i in range(0,81):
			outputs[i] = [0] * 10
			outputs[i][values[i]] = 1

		sudoku.output_to_file(image_names[image_index],270,"suduku_train.csv",outputs)

def train_network(iterations):
	NN = NeuralNetwork(training_data_path = "suduku_train.csv",init_weight_config =[900,360,10])
	NN.train_net(1,save_name="sudoku_number_weights.xml",status_interval=5)
	for i in range(0,iterations):
		NN = NeuralNetwork(training_data_path = "suduku_train.csv",weights_data_path ="sudoku_number_weights.xml")
		NN.train_net(1,save_name="sudoku_number_weights.xml",status_interval=5)


def test_network(file_path):
	sudoku.output_to_file(file_path,270,"testing_net_inputs.csv")

	NN = NeuralNetwork(input_data_path = "testing_net_inputs.csv",weights_data_path ="sudoku_number_weights.xml")

	NN.run_net()


	values = [0] * 9



	row_count = -1
	column_count = 9


	for i in range(0,81):
		output_list = NN.output_data[i][0,:].tolist()[0]
		if(column_count == 9):
			column_count = 1
			row_count += 1
			values[row_count] = []
		else:
			column_count += 1

		values[row_count].append(output_list.index(max(output_list)))
	

	print(np.matrix(values))


#weights
images = ["431.jpg","3453.jpg","3451.jpg"]
values = []
values.append([0,0,0,0,0,0,0,0,0,4,0,0,0,7,0,0,0,3,0,0,0,0,4,0,0,0,0,0,7,0,0,0,0,0,3,0,0,3,9,5,0,7,4,8,0,0,0,8,0,6,0,5,0,0,0,1,0,0,0,0,0,9,0,7,0,0,8,0,4,0,0,6,0,8,0,6,3,9,0,2,0])
values.append([0,0,0,0,0,3,0,5,0,0,0,7,5,0,0,0,0,1,0,0,9,0,1,0,0,0,7,0,9,3,0,5,1,0,0,0,0,0,0,7,0,0,0,9,0,0,8,2,0,4,6,0,0,0,0,0,5,0,8,0,0,0,2,0,0,8,2,0,0,0,0,4,0,0,0,0,0,4,0,6,0])
values.append([0,0,0,7,0,2,0,0,0,0,0,6,0,0,0,9,0,0,0,0,0,0,3,0,0,0,0,0,0,1,3,0,4,8,0,0,9,2,0,0,0,0,0,6,1,0,0,4,0,0,0,7,0,0,0,3,0,4,0,7,0,9,0,7,0,9,5,0,1,3,0,2,8,0,0,0,0,0,0,0,7])

#produce_training_data(values,images)

#train_network(100)

test_network("3454.jpg")
