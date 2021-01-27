import cv2
import numpy as np
import matplotlib.pyplot as plt


path="C:/Users/Asus/Desktop/objectDetectionOpencv/img/coins.jpg"

img=cv2.imread(path)
plt.figure(),plt.imshow(img),plt.axis('off')

#blurring
img_blured=cv2.medianBlur(img,13)
plt.figure(),plt.imshow(img_blured),plt.axis('off'),plt.title('Blurred image')

img2=cv2.cvtColor(img_blured,cv2.COLOR_BGR2GRAY)
plt.figure(),plt.imshow(img2,cmap='gray'),plt.axis('off'),plt.title('Gray scaled image')

#binary threshold

ret,img_thresh=cv2.threshold(img2,75,255,cv2.THRESH_BINARY)
plt.figure(),plt.imshow(img_thresh,cmap='gray'),plt.axis('off'),plt.title('Threshold image')

#contur aşaması
""" contours, hierarchy= cv2.findContours(img_thresh.copy(),cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
 
   if hierarchy[0][i][3] == -1 : 
       cv2.drawContours(img,contours,i,(0,255,0),10)

plt.figure(),plt.imshow(img),plt.axis('off'),plt.title('Contoured image')"""

#watershed

kernel= np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(img_thresh,cv2.MORPH_OPEN,kernel,iterations=2)
plt.figure(),plt.imshow(opening,cmap='gray'),plt.axis('off'),plt.title('Opened image')

#distance

distance_transform=cv2.distanceTransform(opening,cv2.DIST_L2,5)
plt.figure(),plt.imshow(distance_transform,cmap='gray'),plt.axis('off'),plt.title('Distance image')

ret,sure_foreground=cv2.threshold(distance_transform,0.4*np.max(distance_transform),255,0)
plt.figure(),plt.imshow(sure_foreground,cmap='gray'),plt.axis('off'),plt.title('Watershed image')

#background
sure_background=cv2.dilate(opening,kernel,iterations=1)
sure_foreground=np.uint8(sure_foreground)
unknown=cv2.subtract(sure_background,sure_foreground)
plt.figure(),plt.imshow(unknown,cmap='gray'),plt.axis('off'),plt.title('Segmenteted image')

#baglantı

ret,marker=cv2.connectedComponents(sure_foreground)
marker=marker+1
marker[unknown== 255] = 0
plt.figure(),plt.imshow(marker,cmap='gray'),plt.axis('off'),plt.title('Marker image')

#watershed
marker=cv2.watershed(img,marker)
plt.figure(),plt.imshow(marker,cmap='gray'),plt.axis('off'),plt.title('Final image')

contours, hierarchy= cv2.findContours(marker.copy(),cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE) 

for i in range(len(contours)):
 
   if hierarchy[0][i][3] == -1 : 
       cv2.drawContours(img,contours,i,(255,0,0),10)

plt.figure(),plt.imshow(img),plt.axis('off'),plt.title('FINAL')




plt.show()