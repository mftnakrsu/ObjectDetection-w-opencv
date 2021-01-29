import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('einstein.jpg',0)
plt.figure(),plt.imshow(img,cmap='gray'),plt.axis('off')

#sınıflandırıcı
face_Cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_rect=face_Cascade.detectMultiScale(img)

for (x,y,w,h) in face_rect:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),10)

plt.figure(),plt.imshow(img,cmap='gray'),plt.axis('off')


#barcelone team

barca=cv2.imread('barcelona.jpg',0)
plt.figure(),plt.imshow(barca,cmap='gray'),plt.axis('off')
face_rect=face_Cascade.detectMultiScale(barca,minNeighbors=6)#nn parametresi

for (x,y,w,h) in face_rect:
    cv2.rectangle(barca,(x,y),(x+w,y+h),(255,255,255),10)



#realtime
cap=cv2.VideoCapture(0)

while True:
    
    _,frame=cap.read()
    face_rect=face_Cascade.detectMultiScale(frame,minNeighbors=7)

    for (x,y,w,h) in face_rect:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),10)

    cv2.imshow('Face Detection Real time',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()




plt.figure(),plt.imshow(barca,cmap='gray'),plt.axis('off')

plt.show()