import numpy as np
import cv2 as cv2

cap = cv2.VideoCapture('/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/NewData/walking/person01_walking_d1_uncomp.avi')
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()

    thresh_delta = cv2.threshold(frame, 90, 255, cv2.THRESH_BINARY)[1]

    thresh_delta = cv2.dilate(thresh_delta, None, iterations=0)

    cv2.imshow('bg', thresh_delta)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
      
    # define range of red color in HSV 
    lower_red = np.array([30,150,50]) 
    upper_red = np.array([255,255,180]) 
      
    # create a red HSV colour boundary and  
    # threshold HSV image 
    mask = cv2.inRange(hsv, lower_red, upper_red) 

    # Bitwise-AND mask and original image 
    res = cv2.bitwise_and(frame,frame, mask= mask) 
  
    # Display an original image 
    cv2.imshow('Original',frame) 
  
    # finds edges in the input image image and 
    # marks them in the output map edges 
    edges = cv2.Canny(frame,100,200) 
  
    # Display edges in a frame 
    cv2.imshow('Edges',edges)

    

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q') or k == 27:
        break

cap.release()
cv2.destroyAllWindows()