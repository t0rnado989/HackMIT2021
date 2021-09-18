from PIL import Image, ImageChops

import os
from matplotlib import pyplot as plt

directory = "./brain_tumor_dataset/no"
# widths = []
# heights = []
# for file in os.listdir(directory):
#     print(file)
#     try:
#         with Image.open(directory + "/" + file) as image:
#             # print(image.format, f"{image.size}x{image.mode}")
#             pixels = list(image.getdata())
#             width, height = image.size
#             widths.append(width)
#             heights.append(height)
#     except OSError:
#         pass
#
# print(heights)
# print(widths)
# plt.hist(widths)
# plt.show()
#
# plt.hist(heights)
# plt.show()

# with Image.open(directory + "/11 no.jpg") as image:
#     pixels = list(image.getdata())

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

im = Image.open(directory + "/11 no.jpg")
im = trim(im)
im.show()