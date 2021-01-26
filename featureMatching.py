import cv2
import numpy as np
import matplotlib.pyplot as plt

path="C:/Users/Asus/Documents/objectDetectionOpencv/img/"

#chocolates
img=cv2.imread(path+'chocolates.jpg',0)
plt.figure(),plt.imshow(img,cmap='gray'),plt.axis('off')

#nestle
img2=cv2.imread(path+'nestle.jpg',0)
plt.figure(),plt.imshow(img2,cmap='gray'),plt.axis('off')

#ORB tanımlayıcısı(feature)
orb=cv2.ORB_create()

kp1,des1=orb.detectAndCompute(img2,None)
kp2,des2=orb.detectAndCompute(img,None)

#bf matcher
bf=cv2.BFMatcher(cv2.NORM_HAMMING)
matches=bf.match(des1,des2)

#mesafeye göre sırala

matches=sorted(matches, key=lambda x: x.distance)

plt.figure()
img_match=cv2.drawMatches(img2,kp1,img,kp2,matches[:20],None,flags=2)
plt.imshow(img_match),plt.axis('off'),plt.title('ORB')

#SIFT tanımlayıcısı FEATURELERI CIKARAN
sift=cv2.xfeatures2d.SIFT_create()
#bf
bf=cv2.BFMatcher()
#featur tespiti w sift
kp1,des1=sift.detectAndCompute(img2,None)
kp2,des2=sift.detectAndCompute(img,None)

matches=bf.knnMatch(des1,des2,k=2)

good_matches=[]

for m1,m2 in matches:
    if m1.distance < 0.75*m2.distance:
        good_matches.append([m1])


plt.figure()
sift_matches=cv2.drawMatchesKnn(img2,kp1,img,kp2,good_matches,None,flags=2)
plt.imshow(sift_matches),plt.axis('off'),plt.title('SIFT')


plt.show()