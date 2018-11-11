#
# pallpull.py
#
# pallpull.py targetdir
#
# extract all panoramas with 136 images into a set of named subdirectories in the targetdir

import sqlite3;
import os;
import sys;
import glob;
import pipeconfig

def printUsage():
    print("pallpull targetdir")

#parse command line options
print (sys.argv)
if (len(sys.argv) != 2):
 printUsage();
 exit(-1)

targdir = sys.argv[1];


#add the list of image files into the image file table.
dbase = pipeconfig.GBASE + "/" + pipeconfig.GPIPE + "/" + pipeconfig.GDBASE
pdb = sqlite3.connect(dbase)
pdbcursor = pdb.cursor();

#first get a list of all the panoramas and their names
pdbcursor.execute("SELECT rowid,* FROM panoramas WHERE numimages=136");
panolist = pdbcursor.fetchall()

#pdb.execute('''CREATE TABLE ingests (date text, basefolder text, folder text)''')
aleftdir = targdir + "/left"
arightdir = targdir + "/right"

cmdline = "mkdir %s" % (aleftdir)
print(cmdline)
os.mkdir(aleftdir)
cmdline = "mkdir %s" % (arightdir)
print(cmdline)
os.mkdir(arightdir)

for apano in panolist:
  print (apano)
  pdbcursor.execute("SELECT * FROM images WHERE panorama=%s" % (apano[0]))
  imglist = pdbcursor.fetchall()

  eyedir = imglist[0][5]
  print(eyedir)
  panodir = targdir+"/"+eyedir +"/"+apano[2];
  cmdline = "mkdir %s" % (panodir)
  os.mkdir(panodir);
  print(cmdline)

  imgfile =" "
  for aimg in imglist:
    imgfile = imgfile + " " + pipeconfig.GBASE + "/" + pipeconfig.GPIPE + "/storage/" + aimg[1] + "/" + aimg[2]
    if (len(imgfile) > 5000):
       cmdline = "cp  %s %s" % (imgfile,panodir)
       print (cmdline)
       os.system(cmdline)
       imgfile = " "
  if (len(imgfile) > 5):
     cmdline = "cp  %s %s" % (imgfile,panodir)
     print (cmdline)
     os.system(cmdline)


pdb.commit();
pdb.close();



#finished
