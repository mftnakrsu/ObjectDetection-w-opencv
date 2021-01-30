# ObjectDetection-w-opencv
You can see simple detection apps inside the 'basicDetection' file.  

 
## In 'objectDetectionTracking' projects:

These are the HSV values of the blue color. If you want to detect another rangetrack color, you can change the min and max values in HSV. 
	
	blue_l=(84,98,0)
	blue_u=(179,255,255)

 
 I stored a point in the center of the object with the 'deque' command.So, I made a simple 'tracking' application:
	
	from collections import deque

	buffer_size=16
	pts=deque(maxlen= buffer_size)
	
    pts.appendleft(center)
    for i in range (1,len(pts)):
        if pts[i-1] is None or pts[i] is None : continue

        cv2.line(img,pts[i-1], pts[i],(0,255,255),3)


## Feature matching and Template Matching:
You should check this out links.

1)**FEATURE MATCHING:**

![alt text](https://docs.opencv.org/master/matcher_result1.jpg)
-https://www.docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

2)**TEMPLATE MATCHING:**

![alt text](https://docs.opencv.org/master/template_ccoeff_1.jpg)
-https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html 


# Custom Haar Cascade
	
	1) dataset: Need 2 folder negatif and pozitive images named 'p' and 'n'.
	2) Download the cascade GUI program. https://amin-ahmadi.com/cascade-trainer-gui/
	3) Create cascaade
	4) Detect object
	
## Dataset: 
You need to collect positive and negative pictures of your object. You can do it easily with 'custom_cascade.py' script with your cam.

![alt text](https://r.resimlink.com/2mN.jpg)

## Create Cascade:
![alt text](https://r.resimlink.com/rL6at8.jpg)
![alt text](https://r.resimlink.com/UZ6CJ5.jpg)
We can start the cascade trainer:
![alt text](https://r.resimlink.com/UdeBL.jpg)
Our files will be like this after the Cascade train.
![alt text](https://r.resimlink.com/59vh.jpg)

## Detect object:

Run the 'own_cascade_detecion.py' scripts.
This piece of code will make us to detect the object.

	# cascade classifier
    cascade = cv2.CascadeClassifier("cascade.xml")

    while True:
    
   
    # read img
    success, img = cap.read()
    
    if success:
        # convert bgr2gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # detection parameters
        scaleVal = 1 + (cv2.getTrackbarPos("Scale","Sonuc")/1000)
        neighbor = cv2.getTrackbarPos("Neighbor","Sonuc")
        # detectiom
        rects = cascade.detectMultiScale(gray, scaleVal, neighbor)
    
        for (x,y,w,h) in rects:
            
            cv2.rectangle(img, (x,y),(x+w,y+h), color, 3)
            cv2.putText(img, objectName, (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            
        cv2.imshow("Sonuc", img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break


Test time:

![alt text](https://r.resimlink.com/Q6lzku.jpg)

### If you want to know more:

https://docs.opencv.org/master/dc/d88/tutorial_traincascade.html

https://www.youtube.com/watch?v=jG3bu0tjFbk

https://medium.com/@vipulgote4/guide-to-make-custom-haar-cascade-xml-file-for-object-detection-with-opencv-6932e22c3f0e
