#
# ppanodir.py
#
# ppanodir.py pano{folder}
#
# create a folder for the material to go with the panorama

import sqlite3;
import os;
import sys;
import glob;
import datetime;
import pipeconfig


#create a new Folder record in the ingests table"
def newPanoFolderRecord(dbcursor,folderbase,panofolder):
#    print(imgfolder);
    try:
       os.mkdir(pipeconfig.GBASE + "/" + folderbase + "/" + panofolder)
    except OSError:
       print("OSERROR")

    sqlstring = "INSERT INTO ingests VALUES (%s,%s,%s,%s,%s);" % ("\'date\'","\'"+folderbase+"\'","\'"+panofolder+"\'","'none'","'none'")
    dbcursor.execute(sqlstring)
#    print( sqlstring)
    #dbcursor.commit();
#    dbcursor.execute('SELECT rowid, * FROM panoramas WHERE panoname=-1')
#    panores = pdbcursor.fetchall();
    return panofolder;



def printUsage():
    print("ppanodir panofolder") 

#parse command line options
#print (sys.argv)
if (len(sys.argv) != 2):
 printUsage();
 exit(-1)

panofold = sys.argv[1];

#print (eye);

#load database configuration information


#first extract the list of images not classified into panoramas.
dbase = pipeconfig.GBASE + "/" + pipeconfig.GPIPE + "/" + pipeconfig.GDBASE
pdb = sqlite3.connect(dbase)
pdbcursor = pdb.cursor();

afolder = newPanoFolderRecord(pdbcursor,pipeconfig.GPIPE+"/storage",panofold)
print(afolder)

#finished
pdb.commit();
pdb.close()
