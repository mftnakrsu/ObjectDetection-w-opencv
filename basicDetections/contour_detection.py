import numpy as np
import cv2
import matplotlib.pyplot as plt

path="C:/Users/Asus/Desktop/objectDetectionOpencv/img/"

img=cv2.imread(path+'contour.jpg',0)
plt.figure(),plt.imshow(img,cmap='gray'),plt.axis('off'),plt.title('Contour Original Image')


#kontur tespiti
#im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours,hierarchy=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


#retr_ccomp:internal ya da external sekilde extraction yapay
#chain aprox yatay dikey çapraz sıkıştırır ve yalnızca uç noktaları bırakır.

external_contour=np.zeros(img.shape)
internal_contour=np.zeros(img.shape)

for i in range(len(contours)):
    #external
    if hierarchy[0][i][3]==-1:
        cv2.drawContours(external_contour,contours,i,255,-1)
    else:#internal
        cv2.drawContours(internal_contour,contours,i,255,-1)

plt.figure(),plt.imshow(external_contour,cmap='gray'),plt.axis('off'),plt.title('ExternalContoured Image')
plt.figure(),plt.imshow(internal_contour,cmap='gray'),plt.axis('off'),plt.title('Internel Contoured Image')


plt.show()