import cv2
import numpy as np
from collections import deque

#nesne merkezine depolayacak veri tipi
buffer_size=16
pts=deque(maxlen= buffer_size)

# mavi renk aralığı HSV formatı
#ton h, doygun s, brign v
#(84,179),(98,255),(0,255)

blue_l=(84,98,0)
blue_u=(179,255,255)
#capture
cap=cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4,480)

while 1:
    _,img=cap.read()

    #blur noise elimine et
    blurred=cv2.GaussianBlur(img,(11,11),0)

    #hsv
    hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV Image',hsv)

    #mavi için mask
    mask=cv2.inRange(hsv,blue_l,blue_u)
    cv2.imshow('mask Image',mask)

    #maske etrafındaki gürültüleri sil
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)
    cv2.imshow('cleaned mask Image',mask)

    #kontur
    (contours,_)=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center=None

    if len(contours) > 0:
        #max kontur al
        c = max(contours, key = cv2.contourArea)
        #dikdortgene cevir
        rect = cv2.minAreaRect(c)

        ((x,y),(width,height),rotation)=rect

        s="x: {}, y: {}, width: {}, rotation: {} ".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
        print(s)

        box=cv2.boxPoints(rect)
        box=np.int64(box)

        # moment
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

        #draw 
        cv2.drawContours(img,[box],0,(0,255,255),2)

        #merkeze nokta çiz
        cv2.circle(img,center,5,(255,0,255),-1)

        #text
        cv2.putText(img,s,(50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),2)


    #deque
    pts.appendleft(center)
    for i in range (1,len(pts)):
        if pts[i-1] is None or pts[i] is None : continue

        cv2.line(img,pts[i-1], pts[i],(0,255,255),3)

    cv2.imshow('Detetcion',img)





    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()