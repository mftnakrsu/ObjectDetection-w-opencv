import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path="C:/Users/Asus/Desktop/objectDetectionOpencv/img/"

files=os.listdir()
img_list=[]

for i in files:
    if i.endswith('.jpg'):
        img_list.append(i)

#HOG 
hog= cv2.HOGDescriptor()

#SVM
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for i in img_list:
    print(img_list)

    img=cv2.imread(i)

    (rects, weigths)=hog.detectMultiScale(img,padding=(8,8),scale=1.05)

    for (x,y,w,h) in rects:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)


    cv2.imshow('YAYA:',img)

    if cv2.waitKey(0) & 0xFF == ord('q'): continue
    
