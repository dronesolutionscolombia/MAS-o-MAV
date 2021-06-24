# import the necessary packages

import argparse
import cv2
import numpy as np
import threading
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
n_uavs = 5
flag =0 


filename = "/home/liseth/MEGA/DecompositionCodes/geometry_decom/guacas_3_google_ali.tif"
 
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cropping = True
 
    # check to see if the left mouse button was released
    if event == cv2.EVENT_RBUTTONDOWN:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        #refPt.append((x, y))
        cropping = False
 
        # draw a rectangle around the region of interest
        #cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.polylines(image, np.array([refPt]), True, (0, 0, 0), 3, 8)
        #cv2.imwrite("/home/liseth/catkin_ws/maps/map3_modificado_polygon.tif", image)

        #(1) Crop the bounding rect
        pts = np.array(refPt)
        x,y,w,h = cv2.boundingRect(pts)
        #croped = image[y:y+h, x:x+w].copy()
        croped = image.copy()
        #(2) make mask
        #pts = pts - pts.min(axis=0)
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        #(3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        #(4) add the white background
        bg = np.ones_like(croped, np.uint8)*255
        cv2.bitwise_not(bg,bg, mask=mask)
        dst2 = bg+ dst
        #cv2.imshow("cropped", croped)
        #cv2.imshow("mask.png", mask)
        cv2.imwrite(filename.replace('_ali.tif', '_cropped.tif'), dst)
        #cv2.imwrite("/home/liseth/catkin_ws/maps/map3_modificado_cropped.tif", dst)
        #cv2.imshow("dst2.png", dst2)
        

        
# construct the argument parser and parse the arguments

 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(filename)

clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:

    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        break

cv.destroyWindow("image")





   
 
