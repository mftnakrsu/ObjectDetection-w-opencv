import numpy as np
import cv2
import matplotlib.pyplot as plt

path="C:/Users/Asus/Desktop/objectDetectionOpencv/img/"

img=cv2.imread(path+'london.jpg',0)
plt.figure(),plt.imshow(img,cmap='gray'),plt.axis('off')

edges=cv2.Canny(img,0,255)
plt.figure(),plt.imshow(edges,cmap='gray'),plt.axis('off')

med_val=np.median(img)
print(med_val)
mean_val=np.mean(img)
print(mean_val)

low=int(max(0,(1 - 0.33)*med_val))
high=int(min(255,(1+0.33)*med_val))

print(low,high)

edges2=cv2.Canny(img,low,high)
plt.figure(),plt.imshow(edges2,cmap='gray'),plt.axis('off')

#blurring
blurred_img=cv2.blur(img, ksize=(7,7))
edges3=cv2.Canny(blurred_img,low,high)
plt.figure(),plt.imshow(edges3,cmap='gray'),plt.axis('off')





plt.show()
