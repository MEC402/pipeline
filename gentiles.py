#!/usr/local/bin/python3.6
#python code for generating png tiles in the marzipano compatible layout
# gentiles Image_prefix
#

import os
import sys
import glob
import ntpath
from wand.image import Image

def mktiles(basename, imgprefix):

	print("Making directory %s" % basename)
	os.mkdir(basename)
	# Make base directories ahead of time for easier loop logic
	for level in range(1,5):
		os.mkdir(basename+str(level))

	for face in {"u","d","b","f","l","r"}:
		# Get the original copy of the image
		imgname = "%s.%c.png" % (imgprefix, face)
		original = Image(filename=imgname)
		for level in range(1,5):
			adirname = "%s%d/%c" % (basename, level, face)
			os.mkdir(adirname)
			h = pow(2,(level-1))
			tsize = h*512
			# Clone the original image so we don't have to reload from disk
			with original.clone() as copy:
				# Resize the copy to whatever our depth level dictates
				copy.resize(tsize, tsize)
				for row in range(0,h):
					adirname = "%s%d/%c/%d" % (basename, level, face, row)
					os.mkdir(adirname)
					offsety = row * 512
					for col in range(0,h):
						offsetx = col*512
						print("%d %s %d %d" % (level,face,row,col))
						outfile = "%s/%d.png" % (adirname,col)
						# Copy slices of image data for rapid cropping
						with copy[offsetx:offsetx+512, offsety:offsety+512] as crop:
							crop.format = 'png'
							crop.save(filename=outfile)
		# I don't trust python to do this for us, so make an explicit call
		original.destroy()

if (len(sys.argv) != 3):
	print("Arguments: <base directory> <image prefix>")
	exit(0)

mktiles(sys.argv[1], sys.argv[2])
