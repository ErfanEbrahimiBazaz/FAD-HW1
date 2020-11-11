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
img_copy = img.copy()

path_origin_img = 'C:\\Users\\E17538\\OneDrive - Uniper SE\\Desktop\\DailyActivities\\Data_Management\\FAD\\2011\\img.jpg'
img_org = cv.imread(path_origin_img, 0)
cv.imshow('original image', img_org)

# black is 0, white is 255
# T = 16
# img[img >= T] = 255
# img[img < T] = 1

T = 32
# car doesn't exist
img[img >= T] = 0
# Where car features are
img_copy[img_copy < T] = 255

mask = img*img_copy
masked_img = mask * img_org

im_h = cv.hconcat([img_org, masked_img])
cv.imshow('original and masked image', im_h)
# cv.imshow('T_32_K_512', mask)
# cv.imshow('masked image', masked_img)


dest_folder = os.path.join(os.getcwd(), 'task3')
cv.imwrite(os.path.join(dest_folder, 'masked_image.jpg'), im_h)
cv.waitKey(0)

