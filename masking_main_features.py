import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2 as cv
from sklearn.cluster import KMeans
import os


folder = os.getcwd()
path = os.path.join(folder, 'T32')
print(path)

filenames = os.listdir(path)
img_files = []
for file in filenames:
    if file.lower().endswith(('.jpg', '.png')):
        img_files.append(file)

# img = cv.imread(os.path.join(path, img_files[3]), 0)
path2 = 'C:\\Users\\E17538\\OneDrive - Uniper SE\\Desktop\\DailyActivities\\FAD\\ACV_Ses2\\T32\\masked_T32_K_256.jpg'
img = cv.imread(path2, 0)

# black is 0, white is 255
T = 16
img[img >= T] = 255
img[img < T] = 1

cv.imshow('T_32_K_512', img)
cv.waitKey(0)

