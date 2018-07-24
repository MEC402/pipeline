#!/usr/bin/python
#################################
# eq2tiles.py
#
# this program takes a Stereo Pair and generates a stereo tile set
# that can be loaded by Marzipano and tileViewer
#
# author: steve cutchin, Boise State University
#
# eq2tiles imgdirectory imgprefix tiledirectory
#
# imgdirectory is the source directory for the equirectangular panoramas.
# imgprefix is the filname with out the '_{L,R}.tiff' component
# tiledirectory is the directory where the final tiles will end up
#   the tiles will be located in tiledirectory/imgprefix/
#
#
# planned extension option to update JSON file in tiledirectory
#
##################################################
import sys;
import glob;

#temporary directory for intermediary files
Gimgtmpdir="/cornea/scratch/"

def printUsage():
    print("eq2tiles.py <imgdirectory> <imgprefix> <tiledirectory>");

# first step is to check the arguments
# eq2tiles imgdirectory imgprefix tiledirectory
if len(sys.argv) != 3:
   printUsage();
   exit(-1);


# convert the equirectangular tif images into png files.

# convert the equirectangular png files into individual faces 
# use panorama program


# rename face files to input format for gentiles
# face list for output of panorama 0=back,1=left,2=front,3=right,4=up,5=down


# call gentiles.py on face files


#optional remove intermediary files

#finished 
