import cv2
import numpy as np 
import matplotlib.pyplot as plt


path="C:/Users/Asus/Documents/objectDetectionOpencv/img/"

#template matching:
img=cv2.imread(path+'cat.jpg',0)
print(img.shape)

tempImg=cv2.imread(path+'cat_face.jpg',0)
print(tempImg.shape)

h,w=tempImg.shape

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    method=eval(meth)#to func

    res=cv2.matchTemplate(img,tempImg,method)
    print(res.shape)

    min_val,max_val, min_loc,max_loc=cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left=min_loc
    else:
        top_left=max_loc

    bottom_right=(top_left[0]+w, top_left[1]+h)

    cv2.rectangle(img,top_left, bottom_right,255,2)

    plt.figure()
    plt.subplot(121),plt.imshow(res,cmap='gray')
    plt.title('Templated Img'),plt.axis('off')
       
    plt.subplot(122),plt.imshow(img,cmap='gray')
    plt.title('Detected Img'),plt.axis('off')

    plt.suptitle(meth)

    plt.show()
