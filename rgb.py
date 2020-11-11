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

b, g, r = cv.split(img)

cv.imshow('Red scale',r)
cv.imshow('Blue scale',b)
cv.imshow('Green scale',g)

#HSV: Hue: main color, Saturation (gray to main color), Value (intensity, brightness or lightness)
# Hue is the degree of the olor wheel
img_hue = cv.cvtColor(img, cv.COLOR_HSV2BGR)
h, s, v = cv.split(img_hue)

img_hue2 = img_hue.copy()

T = 128
# This sharpens ALL colors
img_hue[img_hue > T] = 255
img_hue[img_hue <= T] = 0

cv.imshow('HSV scale',img_hue)
cv.imshow('Hue image without saturation and brightness', h)
cv.waitKey(0)