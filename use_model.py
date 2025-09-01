from tkinter import Tk, filedialog
from PIL import Image
import numpy as np
from MDP_function import MDP


root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

# Gray Scale
img = Image.open(file_path).convert("L")
img_array = np.array(img)

# inversion
if np.mean(img_array) > 127:
    img_array = 255 - img_array
else: img_array


threshold = 30
img_array[img_array <= threshold] = 0

img_binary = img_array.copy()
img_binary[img_binary > threshold] = 255


result = MDP(img_array)
print("result: ", result)