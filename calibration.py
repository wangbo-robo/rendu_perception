#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:05:12 2023

@author: smallsmallv
"""

import cv2
import numpy as np
import glob
 
# Threshold
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#Checkerboard grid
w = 9
h = 6

objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)

#20mm long
objp = 20*objp

# Store pairs of objet and image coordinates for corner points of the checkerboard grid
objpoints = [] 
imgpoints = [] 
 
images = glob.glob('photocalibration\\' + '*.JPG')
for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, [int(4032*0.25),int(3024*0.25)],interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the corner points of the checkerboard grid
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    
    # If enough point pairs are found, store them
    if ret == True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        # Display corner points on the image
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        cv2.imshow('findCorners',img)
        cv2.waitKey(1)
cv2.destroyAllWindows()
 
print("calculating")
# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("mtx=",mtx)
print("dist=",dist)
