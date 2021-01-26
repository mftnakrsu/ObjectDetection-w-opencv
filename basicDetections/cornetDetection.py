import numpy as np
import cv2
import matplotlib.pyplot as plt

path="C:/Users/Asus/Desktop/objectDetectionOpencv/img/"

img=cv2.imread(path+'sudoku.jpg',0)
img=np.float32(img )

plt.figure(), plt.imshow(img, cmap='gray'),plt.axis('off')
#Harris corner detection
dst=cv2.cornerHarris(img,blockSize=2,ksize=3,k=0.04)#blocksize komşuluk
plt.figure(), plt.imshow(dst, cmap='gray'),plt.axis('off'),plt.title('Corner image')

dst=cv2.dilate(dst,None)
img[dst>0.2*dst.max()]=1
plt.figure(), plt.imshow(dst, cmap='gray'),plt.axis('off'),plt.title('dilated image')

#shi tomsai dtection
img=cv2.imread(path+'sudoku.jpg',0)
img=np.float32(img )

corners=cv2.goodFeaturesToTrack(img,100,0.01,10)
corners=np.int64(corners)

for i in corners:
    x,y=i.ravel()
    cv2.circle(img,(x,y),3,(125,125,125),-1)

plt.figure(), plt.imshow(img, cmap='gray'),plt.axis('off'),plt.title('Newest image')


plt.show()