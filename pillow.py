from PIL import Image
import os
from matplotlib import pyplot as plt

directory = "./brain_tumor_dataset/no"
widths = []
heights = []
for file in os.listdir(directory):
    print(file)
    try:
        with Image.open(directory + "/" + file) as image:
            # print(image.format, f"{image.size}x{image.mode}")
            pixels = list(image.getdata())
            width, height = image.size
            widths.append(width)
            heights.append(height)
    except OSError:
        pass

print(heights)
print(widths)
plt.hist(widths)
plt.show()

plt.hist(heights)
plt.show()
