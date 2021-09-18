import os
from PIL import Image
import pandas as pd
import numpy as np

directory = "./brain_tumor_dataset/no"

dataframe_n = pd.DataFrame()
for file in os.listdir(directory):
    try:
        with Image.open(directory + "/" + file) as image:
            image = image.convert("L")
            image = image.resize((500, 500))
            pixels = list(image.getdata())
            pixels.append(0)
            array = np.array(pixels)
            df = pd.DataFrame(array).transpose()
            dataframe_n = dataframe_n.append(df)
    except OSError:
        pass

directory = "./brain_tumor_dataset/yes"

dataframe_y = pd.DataFrame()
for file in os.listdir(directory):
    try:
        with Image.open(directory + "/" + file) as image:
            image = image.convert("L")
            image = image.resize((500, 500))
            pixels = list(image.getdata())
            pixels.append(1)
            array = np.array(pixels)
            df = pd.DataFrame(array).transpose()
            dataframe_y = dataframe_y.append(df)
    except OSError:
        pass

dataframe = dataframe_n.append(dataframe_y, ignore_index=True)
print(dataframe)



# for i in range(0, len(file_names), 5):
#     dataframe = pd.DataFrame()
#     for j in range(i * 5, (i * 5) + 5):
#         try:
#             with Image.open(file_names[i]) as image:
#                 print(image.format, f"{image.size}x{image.mode}")
#                 image = image.convert("L")
#                 image = image.resize((50, 50))
#                 pixels = list(image.getdata())
#                 print(len(pixels))
#                 array = np.array(pixels)
#                 # print(array)
#                 df = pd.DataFrame(array).transpose()
#                 print(df)
#                 dataframe = dataframe.append(df)
#         except OSError:
#             pass
#     print(dataframe)