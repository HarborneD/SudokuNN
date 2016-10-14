import numpy as np
import csv
import xml.etree.ElementTree as ET
import math
import random

class NeuralNetwork:

	def __init__(self,input_data_path ="",training_data_path = "", weights_data_path = "",init_weight_config =[]):

		if(training_data_path != ""):
			self.set_train_data(training_data_path)

		if(input_data_path != ""):
			self.set_input_data(input_data_path)

		if(weights_data_path != ""):
			self.set_weights_from_file(weights_data_path)

		if(len(init_weight_config) != 0 ):
			self.initialise_weights(init_weight_config)

	def set_train_data(self,training_data_path):
		self.train_inputs,self.train_outputs = self.import_data(training_data_path,True)
	
	def set_input_data(self,input_data_path):
		self.input_data = self.import_data(input_data_path,False)

	def set_weights_from_file(self,weights_data_path):
		self.layer_weights = self.get_weights_from_file(weights_data_path)

	def initialise_weights(self,init_weight_config):
		#init_weight_config a list of layer sizes in the form:
		# [input_layer_size,layer-1_size,layer-2_size ... layer-(n-1)_size,output_layer_size]

		weights = [0] * (len(init_weight_config)-1)

		for l in range(0,len(weights)):
			layer_list = []
			limit = 1/math.sqrt(init_weight_config[l]+1)
			for j in range(0,init_weight_config[l+1]):
				node_list = []
				for i in range(0,init_weight_config[l]+1):
					node_list.append(random.uniform(-limit, limit))			
				layer_list.append(node_list[:])
			weights[l] = np.matrix(layer_list)
		self.layer_weights = weights
		

	#activation functions
	def sigmoid(self,x):
		return 1/(1 + np.exp(-x))

	def Deriv_Sigmoid(self,x):
		sig_x = self.sigmoid(x)
		return sig_x*(1 - sig_x)


	#Neural Network Processing Functions
	def get_next_layer(self,in_var_array,weight_mat,vec_sigmoid_func,input_store):
		in_inc_1 = np.insert(in_var_array,0,1)

		transposed_mat = np.transpose(weight_mat)

		input_store.append(in_inc_1*transposed_mat)
		
		return(vec_sigmoid_func(input_store[-1]))


	def forward_pass(self,input_matrix,vecfunc):
		layer_outputs = []
		pre_trans_inputs = []

		layer_outputs.append(input_matrix)

		for i in range(0,len(self.layer_weights)):
			layer_outputs.append(self.get_next_layer(layer_outputs[i],self.layer_weights[i],vecfunc,pre_trans_inputs))

		return layer_outputs,pre_trans_inputs


	def train_back_prop(self,pre_trans_inputs,layer_outputs,target_outputs,train_rate,momentum,prev_deltas = []):
		error = 0

		final_output_index = (len(layer_outputs)-1) -1
		deltas = []
		for l in range(final_output_index,-1,-1):
			#within the loop to allow for later implementation of different trans functions across layers
			DerivSigVec = np.vectorize(self.Deriv_Sigmoid)
			if(l == final_output_index):

				#l+1 because inclusion of inital inputs
				deltas.append(layer_outputs[l+1] - target_outputs)
				
				error = deltas[-1]*np.transpose(deltas[-1])
				
				deltas[-1] = np.multiply(deltas[-1],DerivSigVec(pre_trans_inputs[l])) 
			else:
				sigma = deltas[-1]*self.layer_weights[l+1]
				
				sigma =np.multiply(np.delete(sigma,0,1),DerivSigVec(pre_trans_inputs[l]))

				deltas.append(sigma)
				
		deltas.reverse()
		
		for l in range(0,len(deltas)):

			if(len(prev_deltas) == len(deltas)):

				delta_weights = train_rate * (np.transpose(deltas[l]) * np.insert(layer_outputs[l],0,1)) + momentum * prev_deltas[l]
				
				prev_deltas.append(delta_weights)
			
			else:	
				
				delta_weights = train_rate * (np.transpose(deltas[l]) * np.insert(layer_outputs[l],0,1))
				
				prev_deltas.append(delta_weights)
			
			self.layer_weights[l] =  self.layer_weights[l]-delta_weights 
			
		return error


	#I/O functions
	def layer_weights_to_file(self,file_path):
		
		current_tab = 0

		output_string = "";
		output_string += '<?xml version="1.0"?>\n'
		output_string += '<NeuralNetwork>\n'
		
		current_tab+=1
		
		for l in range(0,len(self.layer_weights)):
			output_string += '\t'*current_tab
			output_string += '<Layer>\n'
			
			current_tab+=1
			for j in range(0,self.layer_weights[l].shape[0]):
				output_string += '\t'*current_tab
				output_string += '<Node>\n'
				
				current_tab+=1
				for i in range(0,self.layer_weights[l].shape[1]):
					if(i == 0):
						output_string += '\t'*current_tab
						output_string += '<Theta>'
						output_string += str(self.layer_weights[l].item(j,i))
						output_string += '</Theta>\n'
					else:		
						output_string += '\t'*current_tab
						output_string += '<Weight>'
						output_string += str(self.layer_weights[l].item(j,i))
						output_string += '</Weight>\n'
					
				current_tab-=1
				output_string += '\t'*current_tab
				output_string += '</Node>\n'
			
			current_tab-=1
			output_string += '\t'*current_tab
			output_string += '</Layer>\n'
		
		current_tab-=1
		output_string += '\t'*current_tab
		output_string += '</NeuralNetwork>\n'

		with open(file_path, 'w') as f:
			f.write(output_string)


	def get_weights_from_file(self,file_path):
		weights= []

		tree = ET.parse(file_path)
		root = tree.getroot()

		for layer in root:
			node_list = []
			for node in layer:
				weight_list = []
				for weight in node:
					weight_list.append(float(weight.text))
				node_list.append(weight_list[:])
			weights.append(np.matrix(node_list[:]))

		return weights


	def import_data(self,file_path,training=False):
		input_list = []
		output_list = []

		with open(file_path, 'r') as f:
			for line in f:
				line_array = line.rstrip().split(",||,")

				input_list.append([float(i) for i in (line_array[0].split(","))])

				if(training):
					output_list.append([float(i) for i in (line_array[1].split(","))])

		if(training):
			return np.matrix(input_list),np.matrix(output_list)
		else:
			return np.matrix(input_list)

	def save_results(self,save_name):
		with open(save_name,"w") as output_file:
			for x in range(0,len(self.input_data)):
				line = ""
				line += (",".join([ str(x_int) for x_int in self.input_data[x].tolist()])).replace("[","").replace("]","")
				line += ",||,"
				line += (",".join([ str(y_int) for y_int in self.output_data[x].tolist()])).replace("[","").replace("]","")
				line += "\n"
				output_file.write(line)

	#Utilisation Functions
	def train_net(self,train_iterations,train_rate = 0.15,momentum = 0.1,save_name="",status_interval=100):
		if(len(self.train_inputs) == 0 or len(self.train_outputs) == 0):
			raise ValueError('Training Data is not set up corrrectly')
		if(len(self.layer_weights) == 0):
			raise ValueError("The netowrk's weights are not set up")

		vecfunc = np.vectorize(self.sigmoid)

		deltas= []
		error = -1

		for epochs in range(0,train_iterations):
			
			if((status_interval!=0) and (epochs+1) % status_interval == 0):
					print("iteration: "+str(epochs+1))

			for x in range(0,len(self.train_inputs)):
				layer_outputs,pre_trans_inputs = self.forward_pass(self.train_inputs[x],vecfunc)

				error = self.train_back_prop(pre_trans_inputs,layer_outputs,self.train_outputs[x],train_rate,momentum,deltas)

				if((status_interval!=0) and (epochs+1) % status_interval == 0):
					print("output: " + str(layer_outputs[-1]))
					print("Error: " + str(error))

		if(save_name !=""):
			self.layer_weights_to_file(save_name)
		
		#self.layer_weights = layer_weights
		return(self.layer_weights,error)		


	def run_net(self,save_name=""):
		vecfunc = np.vectorize(self.sigmoid)

		self.output_data = []
		
		for x in range(0,len(self.input_data)):
			layer_outputs,activation_inputs = self.forward_pass(self.input_data[x],vecfunc)

			self.output_data.append(layer_outputs[-1])


		if(save_name !=""):
			self.save_results(save_name)
