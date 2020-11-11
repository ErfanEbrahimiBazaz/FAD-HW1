import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2 as cv
from sklearn.cluster import KMeans

img_path = 'C:\\Users\\E17538\\OneDrive - Uniper SE\\Desktop\\DailyActivities\\Data_Management\\FAD\\2011\\img.jpg'
img = cv.imread(img_path, 1)


# Converting from BGR to RGB
cv.cvtColor(img, cv.COLOR_BGR2RGB)

#Reshaping image from (M,N,3) to (M*N, 3)
img = img.reshape((img.shape[1]*img.shape[0], 3))

# Setting number of cluster
# Two alternatives: set manually, set with elbow
kmeans = KMeans(n_clusters=5)
s = kmeans.fit(img)
labels = kmeans.labels_
labels2 = labels
print(labels[284204:])
labels = list(labels)

# Determining centroids
centroid = kmeans.cluster_centers_
print(centroid)

# Calculating the percentages
percent = []
for i in range(len(centroid)):
  j = labels.count(i)
  j = j/(len(labels))
  percent.append(j)
print(percent)


plt.pie(percent, colors=np.array(centroid/255),labels=np.arange(len(centroid)))
plt.show()
# plt.imshow(img)
# cv.imshow('Car image', img)
# cv.waitKey(0)

# Now convert back into uint8, and make original image
center = np.uint8(centroid)
res = center[labels]
res2 = res.reshape((img.shape))

cv.imshow('res2',res2)
cv.waitKey(0)
cv.destroyAllWindows()