#!/usr/bin/python
import subprocess
import MySQLdb
import os, os.path
import datetime

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
cur.execute('SELECT `id`,`picture_1_location` FROM `card` WHERE `processed_file_location` IS NULL AND `sample_id`>0 AND `category`="MSH Kenya"')

# print all the first cell of all the rows
row_count = 0

#loop over rows
for row in cur.fetchall() :
    row_count += 1
    #check for pre-process
    filename = row[1]
    id = row[0]
    base_file = '/'.join(filename.split('/')[:-1])+'/processed/'+'.'.join(filename.split('/')[-1].split('.')[:-1])
    preprocessed_file = '/'.join(filename.split('/')[:-1])+'/preprocessed/'+'.'.join(filename.split('/')[-1].split('.')[:-1])+'.preprocessed.png'
    if os.path.isfile(preprocessed_file):
        #run lanes.py
        inp = ["python", "lanes_pre-rectified.py", "-r", "../processed", "-t", "lanetemplate.png", preprocessed_file]
        p = subprocess.Popen(inp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate()
        code = p.returncode
        print "file",filename,"Code", code
        #if sucessful set daabase
        if code==0:
            print "rectified", id
            resultsfilename = base_file+'.processed.png'
            os.rename(base_file+'.preprocessed.processed.png', base_file+'.processed.png')
            os.rename(base_file+'.preprocessed.csv', base_file+'.csv')

            #write to database
            cur.execute("UPDATE `card` SET `processed_file_location`='%s', `processing_date`='%s' WHERE `id`=%s" % (resultsfilename, datetime.datetime.now(), id));
            # commit your changes
            db.commit()

            #write error file to file
            f = open("processed_pp.txt", "a+")
            f.write(filename + '\n')
            f.close()
        else:
            #write error file to file
            f = open("not_processed_pp.txt", "a+")
            f.write(filename + '\n')
            f.close()

    #break
