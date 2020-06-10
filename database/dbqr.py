#!/usr/bin/python
import datetime, os
import sys
import subprocess
import MySQLdb
import getopt
import zbar
from PIL import Image, ImageEnhance

testName = "12LanePADKenya2015"
sampleName = ""
category = "FHI2020 TEMP"
picture_number = 1

# scan and rectify image function
def scan_rectify_image(filename):
    # create a reader
    scanner = zbar.ImageScanner()

    # configure the reader
    scanner.parse_config('enable')

    # obtain image data
    pil = Image.open(filename).convert('L')
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
        artwork = 0;
        loc = symbol.data.find("padproject.nd.edu/?s=")

        if loc == -1:
            artwork = 1;
            loc = symbol.data.find("padproject.nd.edu/?t=")

        #did we get sn?
        if loc != -1:
            #seperate out the code
            serial_no = symbol.data[21:]

            #print "Serial number ", serial_no
            return int(serial_no)

    return 0

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
cur.execute('SELECT `id`,`picture_'+str(picture_number)+'_location` FROM `card` WHERE `sample_id`=0 AND `category`="FHI2020 TEMP"')

#time
print ("Date", datetime.datetime.now())
#with open('/var/log/padprocess.log', "a") as myfile:
#myfile.write('Date,'+str(datetime.datetime.now())+',\n');
# with open('/var/log/padprocess.html', "a") as myfile:
#     myfile.write('<p>Process files, Date '+str(datetime.datetime.now())+'.<br>\n');

# print all the first cell of all the rows
for row in cur.fetchall() :
    #print row[0],row[1]
    filename = row[1].replace("(","").replace(")","")
    id = row[0]

    # #get test name and compare?
    # if testName != "":
    #     #get test name
    #     teststr = filename.split("-")
    #     compstr = teststr[1]
    #     for i in range(2, len(teststr) - 2):
    #         compstr = compstr + '-' + teststr[i]
    #     #print "Filename",compstr,"Samplename",teststr[0].split("/")[-1]
    #     if testName != compstr:
    #         continue
    #     else:
    #         if sampleName != "":
    #             if sampleName != teststr[0].split("/")[-1]:
    #                 continue
    #             #else:
    #                 #print "Matched Test and Sample"

    #run fix orientation
    # inp = ["mogrify", "-auto-orient", filename]
    # p = subprocess.Popen(inp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8')
    # (stdout, stderr) = p.communicate()
    #run qr_rectify.py
    #inp = ["python", "qr_rectify.py", filename]

    # #name change for pre-rectified
    # pre_rectified_filename = '/'.join(filename.split('/')[:-2])+'/pre_correct/'+'.'.join(filename.split('/')[-1].split('.')[:-2])+ '.precorrected.png'
    # if not os.path.exists(pre_rectified_filename):
    #     continue
    #
    # inp = ["python3", "lanes.py", "-r", "../processed", "-t", "lanetemplate.png", pre_rectified_filename]
    #
    # #run lanes.py
    # #inp = ["python3", "lanes.py", "-r", "../processed", "-t", "lanetemplate.png", filename]
    #
    # p = subprocess.Popen(inp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # (stdout, stderr) = p.communicate()
    # code = p.returncode
    #print "Return code",code
    sample_id = scan_rectify_image(filename)
    print ("ID/File",id,filename,str(sample_id))
#    with open('/var/log/padprocess.log', "a") as myfile:
#        myfile.write('Code/ID/File,'+str(code)+","+str(id)+","+str(filename)+',\n');
    # with open('/var/log/padprocess.html', "a") as myfile:
    #     myfile.write('Code '+str(code)+', ID '+str(id)+', File <a href="http://pad.crc.nd.edu/images/padimages/'+filename.split('/')[-1]+'" target="_blank">'+filename.split('/')[-1]+'</a>.<br>\n');

    #if sucessful set daabase
    if sample_id != 0:
        #resultsfilename = '/'.join(filename.split('/')[:-2])+'/processed/'+'.'.join(filename.split('/')[-1].split('.')[:-2])+'.processed.png'
        cur.execute("UPDATE `card` SET `sample_id`='%d' WHERE `id`=%s" % (sample_id, id));
        # commit your changes
        db.commit()

        print("Saved")
    # else:
    #     #else flag the error code in database
    #     cur.execute("UPDATE `card` SET `rectification_code`='%d' WHERE `id`=%s" % (code, id));
    #     # commit your changes
    #     db.commit()
    #break

#closeout html log file
# with open('/var/log/padprocess.html', "a") as myfile:
#     myfile.write('</p>')

# Close all cursors
cur.close()
# Close all databases
db.close()
