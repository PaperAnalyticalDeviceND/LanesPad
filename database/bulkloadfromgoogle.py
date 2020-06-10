#!/usr/bin/python
import datetime, os
import sys
import subprocess
import MySQLdb
import getopt
from qrtools import QR
import random
import os.path
from shutil import copyfile
from datetime import datetime
import re
#import numpy as np
from sys import argv
import zbar
from PIL import Image, ImageEnhance
#import cv
#import cv2

# scan and rectify image function
def scan_rectify_image(filename):
    # create a reader
    scanner = zbar.ImageScanner()

    # configure the reader
    scanner.parse_config('enable')

    # obtain image data
    pil = Image.open(filename).convert('L')
    width, height = pil.size
    newsize = (1000, (1000* height)/width )
    pil = pil.resize(newsize)

    bright = ImageEnhance.Brightness(pil)
    pil = bright.enhance(1.0)
    contrast = ImageEnhance.Contrast(pil)
    pil = contrast.enhance(2.0)
    sharp = ImageEnhance.Sharpness(pil)
    pil = sharp.enhance(0.3)
    width, height = pil.size
    raw = pil.tobytes()
    pil.save('tmp.png')

    # wrap image data
    image = zbar.Image(width, height, 'Y800', raw)

    # scan the image for barcodes
    scanner.scan(image)

    # extract results
    for symbol in image.symbols:
        # print what we have found
        #print 'decoded', symbol.type, 'symbol', '"%s"' % symbol.data

        #get serial number if exists
        loc = symbol.data.find("padproject.nd.edu/?s=")

        if loc == -1:
            loc = symbol.data.find("padproject.nd.edu/?t=")

        #did we get sn?
        if loc != -1:
            #seperate out the code
            serial_no = symbol.data[21:]

            #print "Serial number ", serial_no

            return serial_no

    return ""

#inline parameter?
optlist, args = getopt.getopt(sys.argv[1:], 't:l:')

category = "FHI2020 TEMP"
test_name = "12LanePADKenya2015"
location = "FHI_H" #"FHI 2020"
store_location = "/var/www/html/joomla/images/padimages/raw_local/"
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
#cur.execute('SELECT `id`,`picture_'+str(picture_number)+'_location` FROM `card` WHERE `processed_file_location`="" AND `rectification_code`=0')

#save_data = False

#walk subfolders
dirs = os.listdir(location)

for dir in dirs:
    sample_name = dir.strip()
    print dir

    subfolder = location + "/" + dir
    fldrs = os.listdir(subfolder)

    #next folder
    for sfile in fldrs:
        res = re.split('_', sfile)
        camera_type_1 = res[0].strip()
        quantity = int(res[1].strip()[0:3])
        print "camera_type_1",camera_type_1, "quantity", "'"+str(quantity)+"'"

        #next folder
        subsubfolder = subfolder + "/" + sfile
        files = os.listdir(subsubfolder)

        #get files
        for file in files:
            # #hack for fail on HEIC
            # if file == "IMG_1795.jpg":
            #     save_data = True
            #
            # if not save_data:
            #     continue

            #get sample_id
            code = scan_rectify_image(subfolder + "/" + sfile + "/" + file)
            sample_id = 0
            if code !="":
                sample_id = int(code)

            #print data
            print category, test_name, sample_name, camera_type_1, str(quantity), file, str(sample_id), store_location, subsubfolder

            # move file to website
            base_name = store_location+sample_name.replace(" ","_")+'-'+test_name.replace(" ","_")+'-'+str(sample_id)
            raw_file_location = base_name+'.original.jpg'
            #check doesnt exist
            while os.path.isfile(raw_file_location):
                raw_file_location	= base_name+'_'+str(random.randrange(1,32767))+'.original.jpg'

            print raw_file_location

            #here we have everything
            #"INSERT INTO card(sample_name,test_name,user_name,date_of_creation,picture_1_location,picture_2_location,camera_type_1,camera_type_2,notes) VALUES('$sample_name', '$test_name', '$uname', now(), '$image1_path', '$image2_path','$camera1','$camera2','$notes')";
            query = "INSERT INTO card(sample_name,test_name,user_name,date_of_creation,picture_1_location,picture_2_location,camera_type_1,camera_type_2,notes,category,sample_id,quantity) " \
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            args = (sample_name, test_name, "Kathleen Hayes", datetime.now(), raw_file_location, "", camera_type_1, "", "Google drive FHI 2020.", category, str(sample_id), str(quantity))

            #do database insert
            cur.execute(query, args)

            # commit your changes
            db.commit()

            print "saved to db"

            #mogrify
            #run fix orientation
            inp = ["mogrify", "-auto-orient", subsubfolder + "/" + file]
            p = subprocess.Popen(inp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (stdout, stderr) = p.communicate()

            #copy
            copyfile(subsubfolder + "/" + file, raw_file_location)

            #break
        #break
    #break
