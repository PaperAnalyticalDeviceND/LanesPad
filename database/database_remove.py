#!/usr/bin/python
import datetime, os
import sys
import subprocess
import MySQLdb
import getopt
from PIL import Image, ImageEnhance

#inline parameter?
testName = "12LanePADKenya2015"
sampleName = "FHI2020 TEMP"

#set to my folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#get database credentials
with open('credentials.txt') as f:
    line = f.readline()
    split = line.split(",")
f.close()

#open database
db = MySQLdb.connect(host="localhost", # your host, usually localhost
                     user=split[0], # your username
                      passwd=split[1], # your password
                      db="pad") # name of the data base

# you must create a Cursor object. It will let
#  you execute all the queries you need
cur = db.cursor()

# Use all the SQL you like
cur.execute('SELECT `id`,`picture_1_location` FROM `card` WHERE `category`="'+sampleName+'"')

counter = 0

# print all the first cell of all the rows
for row in cur.fetchall() :


    # obtain image data
    pil = Image.open(row[1])
    width, height = pil.size

    #rectified?
    if width == 730 and height == 1220:
        counter = counter + 1
        print row[0],row[1],width,height

        #remove it
        os.remove(row[1])

        #remove from db
        cur.execute("DELETE FROM `card` WHERE `id`=%s" % (row[0]));
        # commit your changes
        db.commit()

        #break
    # #if sucessful set daabase
    # if code==0:
    #     resultsfilename = '/'.join(filename.split('/')[:-1])+'/'+'processed'+'/'+'.'.join(filename.split('/')[-1].split('.')[:-1])+'.processed.png'
    #     cur.execute("UPDATE `card` SET `processed_file_location`='%s', `processing_date`='%s' WHERE `id`=%s" % (resultsfilename, datetime.datetime.now(), id));
    #     # commit your changes
    #     db.commit()
    # else:
    #     #else flag the error code in database
    #     cur.execute("UPDATE `card` SET `rectification_code`='%d' WHERE `id`=%s" % (code, id));
    #     # commit your changes
    #     db.commit()
    #break


# Close all cursors
cur.close()
# Close all databases
db.close()

print counter
