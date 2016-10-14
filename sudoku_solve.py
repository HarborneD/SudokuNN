from PyNet import NeuralNetwork
import sudoku
import math
import numpy as np
from PIL import Image
import os

def determine_black_or_white(r,g,b):
	mean = (int(r)+int(g)+int(b))/3

	return (0 if (mean < 128) else 1)

def image_resize(file_path,width=180):
	
	outfile = "temp_img.jpg"
	try:
		with Image.open(file_path,"r") as im:
			current_width,current_height = im.size
			new_height = round((width/current_width) * current_height)
			size =  width, new_height
			im.thumbnail(size, Image.ANTIALIAS)
			im.save(outfile, "JPEG")
	except IOError:
		print ("cannot create thumbnail for '%s'" % file_path)

	return outfile

def create_tile_inputs_matrx(file_path,width_in_pix,width_in_tiles=9):
	with Image.open(image_resize(file_path,width_in_pix)) as im:

		rgb_list = list(im.getdata())

		r_index = 0
		c_index = 0

		tile_size = int(width_in_pix / width_in_tiles)


		pixel_list = [0]*(tile_size*tile_size)
		tile_list =  [0] * (width_in_tiles*width_in_tiles)

		for i in range(0,len(tile_list)):
			tile_list[i] = pixel_list[:]

		
		current_r = 1
		current_c = 0

		for pixel_index in range(0,len(rgb_list)):
			current_c += 1

			if(current_c > width_in_pix):
				current_c = 1
				current_r += 1

			
			tile_r_index = (current_r-1) // tile_size
			tile_c_index = (current_c-1) // tile_size
			
			t_index = (tile_c_index) + (width_in_tiles * tile_r_index)

			pix_r_index = (current_r - (tile_r_index * tile_size))-1
			pix_c_index = (current_c - (tile_c_index * tile_size))-1
			
		
			pix_index = pix_c_index  + (tile_size * pix_r_index)

			
			pix_colour = determine_black_or_white(rgb_list[pixel_index][0],rgb_list[pixel_index][1],rgb_list[pixel_index][2])
		
			tile_list[t_index][pix_index] = pix_colour

	return tile_list


def output_to_file(file_path,width,output_file,outputs=""):
	input_list = create_tile_inputs_matrx(file_path,width)

	if(outputs == ""):
		write_type = "w"
	else:
		write_type = "a"

	with open(output_file,write_type) as fileoutput:
		for i in range(0,81):
			output_line = ""
			output_line += ",".join([str(x) for x in input_list[i]]).replace("[","").replace("]","")
			if(outputs != ""):
				output_line += ",||,"
				output_line += str(outputs[i]).replace("[","").replace("]","")
			output_line += "\n"

			fileoutput.write(output_line)

def solve_sudoku(sudoku_image_path):
	output_to_file(sudoku_image_path,180,"temp_inputs.csv")

	NN = NeuralNetwork(input_data_path = "temp_inputs.csv",weights_data_path ="sudoku_number_weights.xml")

	NN.run_net()


	values = [0] *9

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

	sudoku.solve_sudoku_from_array(np.array(values))

solve_sudoku("3454.jpg")