from pytesseract import image_to_string
from PIL import Image


im1 = Image.open("431.jpg")
width, height = im1.size
im2 = im1.resize((int(width*5), int(height*5)), Image.ANTIALIAS)
im2.save("read_from.png", dpi=(600.0,600.0))


im = Image.open('read_from.png')
print(im)

print(image_to_string(im))