#
# peqingest.py
#
# peqingest.py equirectfile panoid
#
# register an equirectangular file with panorama
# and ingest the file into the database structure

import sqlite3;
import os;
import sys;
import glob;
import pipeconfig

def printUsage():
    print("peqingest EquirectFile PanoID")

#parse command line options
print (sys.argv)
if (len(sys.argv) < 3):
 printUsage();
 exit(-1)

eqfile = sys.argv[1];
panoid = sys.argv[2];


#load database configuration information
dbase = pipeconfig.GBASE + "/" + pipeconfig.GPIPE+"/"+pipeconfig.GDBASE
pdb = sqlite3.connect(dbase)
pdbcursor = pdb.cursor();
#pdb.execute('''CREATE TABLE ingests (date text, basefolder text, folder text)''')
#pdbcursor.execute("DELETE FROM images;")

#first we get the folder name for the given panorama

sqlstring = "SELECT panoname FROM panoramas WHERE rowid=%s;" % (panoid)
pdbcursor.execute(sqlstring)
pname = pdbcursor.fetchall()
(panofolder,) = pname[0];
print (pname,panofolder)

#second copy the equirect file into the directory
#cp file tree from indirectory to ingestdir
cmdline = "cp %s %s" % (eqfile,panofolder)
os.chdir(pipeconfig.GBASE+"/" + pipeconfig.GPIPE + "/storage")
os.system(cmdline)
print (cmdline)
eqbasefile = os.path.basename(eqfile)

#add a new record into the equirect table
sqlstring = "INSERT INTO equirect VALUES (%s,%s,%s,%s);" % ("\'date\'","\'"+panofolder+"\'","\'"+eqbasefile+"\'",panoid)
print( sqlstring)
pdbcursor.execute(sqlstring)

pdb.commit();
pdb.close();
#finished
