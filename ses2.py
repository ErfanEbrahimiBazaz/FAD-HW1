import cv2 as cv
import tensorflow as tf
import keras
import keras_applications
import os
import numpy as np


img_path = 'C:\\Users\\E17538\\OneDrive - Uniper SE\\Desktop\\DailyActivities\\Data_Management\\FAD\\2011\\img.jpg'
img = cv.imread(img_path, 1)
# clone otherwise call by reference
img2 = img.copy()

#gary_scale
img_bw = cv.imread(img_path, 0)

img_bw2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_bw2_cp = img_bw2.copy()

# cv.imshow('some img',img_bw2)
print(len(img))
print(img.shape)
# for i in range(0,img.shape[0]):
#     for j in range(0,img.shape[1]):
#         print(img[i,j])
# print(img)
# cv.waitKey(0)

# for i in range(0,img.shape[0]):
#     for j in range(0,img.shape[1]):
#         if img[i,j]>0.5:
#             img[i,j] = 255
#         else:
#             img[i,j] = 0

# CHanging grayscale to black and white (bitmap)
T = 128
img_bw2[img_bw2 > T] = 1
img_bw2[img_bw2 <= T] = 0

cv.imshow('black and white', img_bw2)

# np.dot() doesn't work here. It must be dot product of corresponding pixels and not 
# new_img = np.dot(img_bw2, img_bw2_cp)
new_img = img_bw2 * img_bw2_cp
cv.imshow('segmented (filtered) image', img_bw2_cp)
cv.waitKey(0)







