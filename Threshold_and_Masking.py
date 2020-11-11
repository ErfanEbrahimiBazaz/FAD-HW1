import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2 as cv
from sklearn.cluster import KMeans
import os


# Loading all images from a folder
folder = os.getcwd()
def load_images_from_folder(folder):
    images = []
    image_list = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('jpg', 'png')):
            img = cv.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
                image_list.append(filename)
    return images, image_list

images, image_list  = load_images_from_folder(folder)
# print(len(images)) # 9
print(image_list)

# img_path = 'C:\\Users\\E17538\\OneDrive - Uniper SE\\Desktop\\DailyActivities\\FAD\\ACV_Ses2\\K_2.jpg'
counter = 0
path = 'C:\\Users\\E17538\\OneDrive - Uniper SE\\Desktop\\DailyActivities\\FAD\\ACV_Ses2\\Images'
for image in image_list:
    img = cv.imread(image, 1)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Masking
    T = 255
    img_gray[img_gray >= T] = 255
    img_gray[img_gray < T] = 0
    # cv.imshow('black and white T_' + str(T) + '_' + image, img_gray)
    cv.imwrite(os.path.join(path, 'masked_T'+ str(T) + '_' + image), img_gray)
    print(counter)
    counter += 1
#
cv.waitKey(0)