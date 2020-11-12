# Task 1
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2 as cv
from sklearn.cluster import KMeans

img_path = 'C:\\Users\\E17538\\OneDrive - Uniper SE\\Desktop\\DailyActivities\\Data_Management\\FAD\\2011\\img.jpg'
img = cv.imread(img_path, 1)

Z = img.reshape((-1, 3))  #an array of 568408 BGR values
# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
# Criteria is iteration termination.
'''
    cv.TERM_CRITERIA_EPS:  specified accuracy,
    cv.TERM_CRITERIA_MAX_ITER: specified number of iterations, max_iter, 
    
    cv.KMEANS_RANDOM_CENTERS: one of the two flags to chose centers for kmeans.
    cv2.KMEANS_PP_CENTERS and cv2.KMEANS_RANDOM_CENTERS
'''
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
for i in range(1, 10):
    '''
        ret: compactness
        labels: marking each element
        center: centroids of kmeans
    '''
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # save file
    file_name = 'K_' + str(K) + '.jpg'
    cv.imwrite(file_name, res2)
    cv.imshow(file_name, res2)
    K *= 2

cv.waitKey(0)
cv.destroyAllWindows()