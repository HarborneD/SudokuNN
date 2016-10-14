from PyNet import NeuralNetwork
import math
import numpy as np
from PIL import Image
import os
import uuid

class move_state:
	def __init__(self,row_i,column_i,value,further_moves,remaining_deductions,reason = ""):
		self.coord = (row_i,column_i)
		self.set_value = value
		self.further_moves_list = further_moves
		self.deductions_left = remaining_deductions
		self.current_move = 0
		self.reason = reason

	def __str__(self):
		return "("+str(self.coord)+": "+str(self.set_value)+" furthers: "+str(len(self.further_moves_list))+")"
	
	def __repr__(self):
		return "("+str(self.coord)+": "+str(self.set_value)+" furthers: "+str(len(self.further_moves_list))+")"
		
def make_possibles_matrix(row_length):
	all_nums_set = set(range(1,row_length+1))

	possibles_matrix = np.matrix([[all_nums_set] * row_length] * row_length)

	return possibles_matrix

def make_sqaures_sets_array(row_length):
	
	squares_sets = [set() for index in range(0,row_length)]

	return squares_sets

def determine_elements_square(x_cord,y_cord,row_length):
	square_length = int(math.sqrt(row_length))

	square_x = math.ceil((x_cord+1)/square_length) -1

	square_y = math.ceil((y_cord+1)/square_length)

	return (square_y + square_x * square_length) - 1

def determine_square_first_element_coord(square_index,row_length):
	square_length = int(math.sqrt(row_length))

	space_x = ((square_index) // square_length) * square_length
	space_y = ((square_index) % square_length) * square_length
	
	return space_x,space_y

def determine_square_element_respect_to_index(start_x,y_start,row_length,inner_index):
	square_length = int(math.sqrt(row_length))

	x_contrib = ((inner_index) % square_length)
	y_contrib = ((inner_index) // square_length)


	return start_x+x_contrib,y_start+y_contrib

def create_sets_arrays(square_mat,row_length):

	row_sets = []
	column_sets = []

	for i in range(0,row_length):
		row_sets.append(set(square_mat[i,:]))

		column_sets.append(set(square_mat[:,i]))

	return row_sets,column_sets

def initial_setup(square_mat,possibles_mat,squares_sets,row_sets,column_sets,row_size,remaining_deductions,verbose = False):

	for row_i in range (0,row_size):

		for column_i in range (0,row_size):
			
			square_val = square_mat[row_i,column_i]

			#if(verbose):
				#print("Square Value:")
				#print(square_val)
			
			if(square_val != 0):
				possibles_mat[row_i,column_i] = set([])
				remaining_deductions -= 1

				large_square_index = determine_elements_square(row_i,column_i,row_size)
				
				#if(verbose):
					#print("Large Square Index:")
					#print(large_square_index)
				#squares_sets[large_square_index].add(square_val)
				squares_sets[large_square_index].add(square_val)
			else:	
				possibles_mat[row_i,column_i] = possibles_mat[row_i,column_i].difference(row_sets[row_i])
				possibles_mat[row_i,column_i] = possibles_mat[row_i,column_i].difference(column_sets[column_i])

	for row_i in range (0,row_size):

		for column_i in range (0,row_size):
			
			square_val = square_mat[row_i,column_i]
			
			if(square_val == 0):
				large_square_index = determine_elements_square(row_i,column_i,row_size)

				possibles_mat[row_i,column_i] = possibles_mat[row_i,column_i].difference(squares_sets[large_square_index])

	return remaining_deductions

def look_for_square_based_moves(possibles_mat,row_size,moves_list,remaining_deductions,verbose = False):
	for row_i in range (0,row_size):

		for column_i in range (0,row_size):

			if(verbose):
				print("row_i,column_i:"+str(row_i)+","+str(column_i))

			square_index = determine_elements_square(row_i,column_i,row_size)
			square_space_start_x,square_space_y_start = determine_square_first_element_coord(square_index,row_size)  

			if(verbose):
				print("square_space_start_x,square_space_y_start:")
				print(square_space_start_x,square_space_y_start)
			

			current_space_set = possibles_mat[row_i,column_i]
			
			if(verbose):
				print("Starting Possibles:")
				print(current_space_set)
			
			for square_space_index in range(0,row_size):
				check_x,check_y = determine_square_element_respect_to_index(square_space_start_x,square_space_y_start,row_size,square_space_index)
				
				if(verbose):
					print("check_x,check_y:")
					print(check_x,check_y)
					print("possibles_mat[check_x,check_y]")
					print(possibles_mat[check_x,check_y])
					
				if(not((row_i == check_x) and (column_i == check_y))):	
					current_space_set = current_space_set.difference(possibles_mat[check_x,check_y])
					if(verbose):
						print("current_space_set:")
						print(current_space_set)
			if(verbose):
				print("final current_space_set:")
				print(current_space_set)
			
			if(len(current_space_set) == 1):
				moves_list.append(move_state(row_i,column_i,list(current_space_set)[0],[],remaining_deductions,"square,"+str(square_index+1)))




def look_for_moves(square_mat,possibles_mat,squares_sets,row_sets,column_sets,row_size,moves_list,remaining_deductions,verbose = False):
	
	for row_i in range (0,row_size):

		for column_i in range (0,row_size):
			current_set = possibles_mat[row_i,column_i].difference(squares_sets[determine_elements_square(row_i,column_i,row_size)])
			
			if(len(current_set) == 1):
				moves_list.append(move_state(row_i,column_i,list(current_set)[0],[],remaining_deductions,"space,"))
			else:

				for check_row_i in range (0,row_size):
					if(row_i != check_row_i):
						current_set = current_set.difference(possibles_mat[check_row_i,column_i]) 
				
				if(len(current_set) == 1):
					moves_list.append(move_state(row_i,column_i,list(current_set)[0],[],remaining_deductions,"column,"+str(column_i+1)))

				
				current_set = possibles_mat[row_i,column_i].difference(squares_sets[determine_elements_square(row_i,column_i,row_size)])
				for check_column_i in range (0,row_size):
					if(column_i != check_column_i):
						current_set = current_set.difference(possibles_mat[row_i,check_column_i])

				if(len(current_set) == 1):
					moves_list.append(move_state(row_i,column_i,list(current_set)[0],[],remaining_deductions,"row,"+str(row_i+1)))


def make_move(sudoku_square,possibles_mat,squares_sets,row_sets,column_sets,row_size,move,verbose = False):
	cord_x = move.coord[0]
	cord_y = move.coord[1]
	value = move.set_value

	if(verbose):
		reason_dict = {"space":"Because this was the only remaining possible value for this space.","column":"Because this space was the only one in column CN that could take this value.","row":"Because this space was the only one in row RN that could take this value.","square":"Because this space was the only one in square SN that could take this value."}

		reason_array = move.reason.split(",")
		reason_string = reason_dict[reason_array[0]].replace("CN",reason_array[1]).replace("RN",reason_array[1]).replace("SN",reason_array[1])
		if(verbose):
			print("Place Value '"+str(value)+"' at "+str(cord_x+1)+","+str(cord_y+1))
			print(reason_string)

	sudoku_square[cord_x,cord_y] = value

	possibles_mat[cord_x,cord_y] = set()
	


	for i in range(0,row_size):
		possibles_mat[i,cord_y].discard(value)
		possibles_mat[cord_x,i].discard(value)


	large_square_index = determine_elements_square(cord_x,cord_y,row_size)

	squares_sets[large_square_index].add(value)
	
	row_sets[cord_x].add(value)
	
	column_sets[cord_y].add(value)


def solve_sudoku_from_array(sudoku_square,verbose = False):	
	moves_list = []	
	moves_left = True
	move_index = 0

	row_size = sudoku_square.shape[0]

	lines = [0] * row_size

	remaining_deductions = row_size * row_size

	if(verbose):
		print("Game Square:")
		print(sudoku_square)
	
	possibles_mat = make_possibles_matrix(row_size)
	
	if(verbose):
		print("Possibles Matrix:")
		print(possibles_mat)

	row_sets,column_sets = create_sets_arrays(sudoku_square,row_size)
	
	if(verbose):
		print("Row Sets:")
		print(row_sets)
	if(verbose):
		print("Column Sets:")
		print(column_sets)

	squares_sets = make_sqaures_sets_array(row_size)
	
	if(verbose):
		print("Square Sets:")
		print(squares_sets)

	if(verbose):
		print("Remaining Deductions:")
		print(remaining_deductions)

	remaining_deductions = initial_setup(sudoku_square,possibles_mat,squares_sets,row_sets,column_sets,row_size,remaining_deductions,True)


	while_count = 0

	current_move = ""



	while(moves_left and while_count < 100):
		while_count += 1
		if(verbose):
			print(sudoku_square)

		if(remaining_deductions == 0):
			if(verbose):
				print("Sudoku Beaten!")
			return(sudoku_square)
			break

		possible_moves = 0
		#check for possible move lists that have only 1 number left and then check row clues and column clues
		if(current_move == ""):
			look_for_moves(sudoku_square,possibles_mat,squares_sets,row_sets,column_sets,row_size,moves_list,remaining_deductions,False)
			#print(moves_list[move_index])
			possible_moves = len(moves_list)
		else:
			look_for_moves(sudoku_square,possibles_mat,squares_sets,row_sets,column_sets,row_size,current_move.further_moves_list,remaining_deductions,False)
			#print(current_move.further_moves_list)
			possible_moves = len(current_move.further_moves_list)

		
		#check the squares for move clues if no moves from rows//columns
		if(possible_moves ==0):
			look_for_square_based_moves(possibles_mat,row_size,current_move.further_moves_list,remaining_deductions,False)
			possible_moves = len(current_move.further_moves_list)
			if(possible_moves ==0):
				moves_left = False
				if(verbose):
					print(possibles_mat)
					print("No moves left")

		if(moves_left):	
			if(current_move ==""):
				current_move = moves_list[0]
			else:
				current_move = current_move.further_moves_list[0]

		
			make_move(sudoku_square,possibles_mat,squares_sets,row_sets,column_sets,row_size,current_move,False)
			remaining_deductions -= 1
			move_index+=1


#sudoku imaging
def determine_black_or_white(r,g,b):
	mean = (int(r)+int(g)+int(b))/3

	return (0 if (mean < 180) else 1)

def image_resize(file_path,width=270):
	
	outfile = str(uuid.uuid1())+".jpg"
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


def create_sudoko_image_matrix(file_path,width):
	resized_file_path = image_resize(file_path,width)

	with Image.open(resized_file_path) as im:

		rgb_list = list(im.getdata())

		rows = []

		im_width,im_height = im.size

		for row in range(1,im_height+1):
			rows.append([determine_black_or_white(rgb_list[pixel_index][0],rgb_list[pixel_index][1],rgb_list[pixel_index][2]) for pixel_index in range((row-1)*im_width,((row)*im_width)-1) ] )
	
	os.remove(resized_file_path)
	return np.matrix(rows)		


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
	input_list = get_tile_pixel_list(file_path,width)

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

	temp_id = str(uuid.uuid1())

	output_to_file(sudoku_image_path,270,temp_id+".csv")

	NN = NeuralNetwork(input_data_path = temp_id+".csv",weights_data_path ="sudoku_number_weights.xml")

	NN.run_net()

	os.remove(temp_id+".csv")

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

	return np.array(values),solve_sudoku_from_array(np.array(values))



def sudoku_segmentation(image_path,image_resize_width):

	pixel_matrix = create_sudoko_image_matrix(image_path,image_resize_width)
	
	current_pixel = 0
	x_start = 0
	y_start = 0
	x_increased = 0

	while((y_start < pixel_matrix.shape[1]) and (current_pixel == 0)):
		
		current_pixel = pixel_matrix.item(y_start,x_start)
		
		if(current_pixel == 0):
			if(x_increased ==0):
				x_start +=1
				x_increased = 1
			else:
				y_start +=1
				x_increased = 0

	if(current_pixel == 1):
		current_scan_colour = 1
		current_scan_index = x_start +1
		x_list_starts = []
		x_list_stops = []

		x_list_starts.append(x_start)
		scan_width = pixel_matrix.shape[1]
		scan_height = pixel_matrix.shape[0]
		while(current_scan_index < scan_width):
			pix_colour = pixel_matrix.item(y_start,current_scan_index)
			
			if pix_colour != current_scan_colour:
				
				if(current_scan_colour == 0):
					x_list_starts.append(current_scan_index)
				else:
					x_list_stops.append(current_scan_index-1)
				
				current_scan_colour = pix_colour

			current_scan_index +=1

		x_list_stops.append(scan_width-1)



		current_scan_colour = 1
		current_scan_index = y_start +1
		y_list_starts = []
		y_list_stops = []

		y_list_starts.append(y_start)
		scan_height = pixel_matrix.shape[1]
		while(current_scan_index < scan_height):
			pix_colour = pixel_matrix.item(current_scan_index,x_start)
			
			if pix_colour  != current_scan_colour:
				
				if(current_scan_colour == 0):
					y_list_starts.append(current_scan_index)
				else:
					y_list_stops.append(current_scan_index-1)
				
				current_scan_colour = pix_colour

			current_scan_index +=1

		y_list_stops.append(scan_height-1)



	tile_list = []
	for y in range(0,len(y_list_starts)):
		for x in range(0,len(x_list_starts)):
			tile_list.append((pixel_matrix[y_list_starts[y]:y_list_stops[y],x_list_starts[x]:x_list_stops[x]]).flatten().tolist()[0])

	return tile_list

def get_tile_pixel_list(file_path,resize_image_width):
	tiles = sudoku_segmentation(file_path,resize_image_width)

	max_len = 900
	tile_list =  [(x + [0] * (max_len - len(x))) for x in tiles]
	print(len(tile_list))
	return tile_list

