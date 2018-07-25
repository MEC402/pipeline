#!/usr/local/bin/python3.6


import os
import sys
import glob
import ntpath
import threading
from wand.image import Image

faces = {"u", "d", "b", "f", "l", "r"}

def make_directories(basename):
	os.mkdir(basename)
	for level in range(1,5):
		basedir = "%s/%d" % (basename, level)
		os.mkdir(basedir)
		for face in faces:
			facedir = "%s/%c" % (basedir, face)
			os.mkdir(facedir)
			for depth in range(0,pow(2,(level-1))):
				depthdir = "%s/%d" % (facedir, depth)
				os.mkdir(depthdir)
	print("Directory tree built")

def mktiles(basename, imgprefix, face):
	imgname = "%s.%c.png" % (imgprefix, face)
	original = Image(filename=imgname)
	for level in range(1,5):
		dirname = "%s/%d/%c" % (basename, level ,face)
#		os.mkdir(dirname)
		h = pow(2,(level-1))
		tsize = h*512
		with original.clone() as copy:
			copy.resize(tsize, tsize)
			for row in range(0,h):
				dirname = "%s/%d/%c/%d" % (basename, level, face, row)
#				os.mkdir(dirname)
				offsety = row*512
				for col in range(0,h):
					offsetx = col*512
					outfile = "%s/%d.png" % (dirname, col)
					with copy[offsetx:offsetx+512, offsety:offsety+512] as crop:
						crop.format = 'png'
						crop.save(filename=outfile)
	original.destroy()

make_directories(sys.argv[1])
threads = []
for face in faces:
	t = threading.Thread(target=mktiles, args=(sys.argv[1], sys.argv[2], face,))
	threads.append(t)
	t.start()
	#mktiles(sys.argv[1], sys.argv[2], face)
