from PIL import Image
import numpy as np

def determine_black_or_white(r,g,b):
	mean = (int(r)+int(g)+int(b))/3

	return (0 if (mean < 128) else 1)


im = Image.open("431.jpg")

rgb_list = list(im.getdata())

bw_list = []
sequentials = []

last_pix = -1
sequential_count = 0

rows = []
new_row = False

char_max_width = 200

im_width,im_height = im.size
pixel_for_row_count = 0

for pixel_index in range(0,len(rgb_list)):
	pix_colour = determine_black_or_white(rgb_list[pixel_index][0],rgb_list[pixel_index][1],rgb_list[pixel_index][2])
	pixel_for_row_count += 1

	if((pix_colour != last_pix) and last_pix != -1):
		sequentials.append(str(last_pix)+"-"+str(sequential_count))
		sequential_count = 1
	else:
		sequential_count += 1
	last_pix = pix_colour


	if(pixel_for_row_count == im_width):
		new_row = True
		pixel_for_row_count = 0

	if(new_row):
		sequentials.append(str(last_pix)+"-"+str(sequential_count))
		sequential_count = 0
		last_pix = -1
		rows.append(bw_list[:])
		bw_list = []
		new_row = False

	bw_list.append(pix_colour)

print(sequentials)

pix_mat = np.matrix(rows)

print(pix_mat)