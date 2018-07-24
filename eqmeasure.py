#!/usr/bin/python
#################################
# eqmeasure.py
#
#this program takes a Stereo Pair and generates a measure of the stereo accuracy
#
# author: jonathan aparicio, steve cutchin, Boise State University
#
# eqmeasure imgleft imgright
#
# imgleft the equirectangural left image
# imgright the equirectangular right image
#
#
##################################################
import sys;
import glob;

import cv2
import numpy as np


def printUsage():
    print("eqmeasure.py <imgleft> <imgright>");

# first step is to check the arguments
# eqmeasure.py <imgleft> <imgright>
if len(sys.argv) != 3:
   printUsage();
   exit(-1);

imgleftname = sys.argv[1]
imgrightname = sys.argv[2]

print (imgleftname,imgrightname)

#first thing is load in the two images
imgleft = cv2.imread(imgleftname)
imgright = cv2.imread(imgrightname)
imgleftgray = cv2.cvtColor(imgleft, cv2.COLOR_BGR2GRAY)
imgrightgray = cv2.cvtColor(imgright, cv2.COLOR_BGR2GRAY)
#cv2.imshow("test",imgleft)
#cv2.waitKey(0)
#at some point we would like to add a spherical projection step in here
#pwarp = cv2.warpSpherical(1.0)


#second we use SIFT to find matches between the two images
#sift = cv2.xfeatures2d.SIFT_create()
detector = cv2.xfeatures2d.SIFT_create()
#descriptor = cv2.DescriptorExtractor_create("SIFT")

lkp, ld = detector.detectAndCompute(imgleftgray, None)
rkp, rd = detector.detectAndCompute(imgrightgray, None)
print ("keypoints found")

#use FLANN matcher
flann_params = dict(algorithm=1, trees=5)
matcher =cv2.FlannBasedMatcher(flann_params,{})
matches = matcher.knnMatch(ld,rd,k=2)

                      
#img3 = cv2.drawMatchesKnn(imgleftgray,lkp,imgrightgray,rkp,matches,None)
#cv2.imshow("match",img3)
#cv2.waitKey(0)
#for m in matches:
#   print (m[0].distance, m[1].distance)

print ("FLANN matched")

#third we calculate our error for the SIFT matches


#output the final error measure

print ("Finished")

