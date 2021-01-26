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

