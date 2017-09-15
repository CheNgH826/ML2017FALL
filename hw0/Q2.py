from PIL import Image
import sys

im = Image.open(sys.argv[1])
im = im.convert('RGB')
pixels = im.load()

for i in range(im.size[0]):
    for j in range(im.size[1]):
        pixels[i, j] = (int(pixels[i, j][0]/2), int(pixels[i, j][1]/2), int(pixels[i, j][2]/2))

im.save("Q2.png")