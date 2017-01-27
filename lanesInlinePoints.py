#!/usr/bin/env python
import sys
import subprocess
import cv
import cv2
import os
import math
import numpy as np
import getopt

####################################################################################################
# Chris Sweet. Computer Science and Engineering. Notre Dame/CRC.
# 08/01/2014
# Rotation, Translation and scaling from 2 points geometrically
####################################################################################################
def RotTrans2Points(srcpoints, dstpoints):
    # Calculate Centroids
    centroid_a = (0, 0)
    centroid_b = (0, 0)
    for i in range(0, len(srcpoints)):
        centroid_a = (centroid_a[0] + srcpoints[i][0], centroid_a[1] + srcpoints[i][1])
        centroid_b = (centroid_b[0] + dstpoints[i][0], centroid_b[1] + dstpoints[i][1])
    centroid_a = (centroid_a[0] / len(srcpoints), centroid_a[1] / len(srcpoints))
    centroid_b = (centroid_b[0] / len(srcpoints), centroid_b[1] / len(srcpoints))

    # Remove Centroids
    new_src = np.copy(srcpoints)
    new_dst = np.copy(dstpoints)
    for i in range(0, len(srcpoints)):
        new_src[i] = (new_src[i][0] - centroid_a[0], new_src[i][1] - centroid_a[1])
        new_dst[i] = (new_dst[i][0] - centroid_b[0], new_dst[i][1] - centroid_b[1])

    #get rotation
    v1 = [(new_src[0][0] - new_src[1][0]), (new_src[0][1] - new_src[1][1])]
    v2 = [(new_dst[0][0] - new_dst[1][0]), (new_dst[0][1] - new_dst[1][1])]
    ang = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
    cosang = math.cos(ang)
    sinang = math.sin(ang)
    
    #create rotation matrix
    R = np.matrix([
        [cosang, -sinang],
        [sinang, cosang]
    ])
        
    # Calculate Scaling
    Source = R * new_src.T

    sum_ss = 0
    sum_tt = 0
    for i in range(0, len(srcpoints)):
        sum_ss += new_src[i][0] * new_src[i][0]
        sum_ss += new_src[i][1] * new_src[i][1]
               
        sum_tt += new_dst[i][0] * Source.A[0][i];
        sum_tt += new_dst[i][1] * Source.A[1][i];

    # Scale Matrix
    R = (sum_tt / sum_ss) * R
    
    # Calculate Translation
    C_A = np.matrix([[-centroid_a[0], -centroid_a[1]]])
    C_B = np.matrix([[centroid_b[0], centroid_b[1]]])
    
    TL = (C_B.T + (R * C_A.T))
    
    # Combine Results
    # version for image transformation
    T = np.matrix([
        [R.A[0][0], R.A[0][1], TL.A[0][0]],
        [R.A[1][0], R.A[1][1], TL.A[1][0]]
    ])
                   
    #return partial matrix
    return T

####################################################################################################
# Chris Sweet. Computer Science and Engineering. Notre Dame/CRC.
# 06/12/2014
# Start of code
####################################################################################################
if len(sys.argv) < 4:
    #rl.write('Insufficient parameters '+len(sys.argv)+'\n')
    #rl.close()
    print 'Insufficient parameters!'
    sys.exit(1)

artwork = -1

#get filenames and roots
filename = sys.argv[1]
filenameroot = '.'.join(filename.split('.')[:-1])
resultsfilenameroot = filenameroot
resultsfilenameroot = '/'.join(filename.split('/')[:-1])+'/processed/'+'.'.join(filename.split('/')[-1].split('.')[:-1])

#open filename /var/www/html/joomla/neuralnetworks/
pos1 = filename.rfind('-')
pos2 = filename.rfind('.')

randpart = filename[pos1+1:pos2]
#print "Rand part", randpart

rl = open('log/rectifylog'+randpart+'.txt', "w")
rl.write('Filename '+filename+'\n')

# OK load image
print 'filename is :', filename

orig_im = cv2.imread(filename)
(h, w, p) = orig_im.shape

#need to rotate image?
if w>h:
    orig_im = np.rot90( orig_im, 1 )

(h, w, p) = orig_im.shape

#~~~~get points
strpoints = sys.argv[2].split(',')

#test not too many points
if len(strpoints) != 16:
    rl.write('Wrong number of points found, '+len(strpoints)+','+strpoints+'\n')
    rl.close()
    print "Error: Wrong number of points found.",len(strpoints),strpoints
    sys.exit(2)

selectWidth = 0.0
try:
    selectWidth = float(sys.argv[3])
except ValueError:
    rl.write('Selection width not a float.\n')
    rl.close()
    print "Error: Selection width incorrect."
    sys.exit(3)

#scale points
factor = w / selectWidth

print "Factor", factor

#convert to float array
actualPoints_x = []
actualPoints_y = []

try:
    for i in range(0,8):
        actualPoints_x.append(float(strpoints[i * 2]) * factor)
        actualPoints_y.append(float(strpoints[i * 2 + 1]) * factor)
except ValueError:
    rl.write('Cannot convert points to floats.\n')
    rl.close()
    print "Error: Cannot convert points to floats."
    sys.exit(4)

#print "Points",actualPoints_x,actualPoints_y

#now sort them
qrpoints = []
outerpoints = []

ids_x = np.array(actualPoints_x).argsort()
ids_y = np.array(actualPoints_y).argsort()

#find index in both top 3 x and top 3 y
#now the vertical group
startindex = 0
idxqr_1 = 0
idxop_1 = ids_x[0]
for i in range(0, 3): #over x
    if actualPoints_y[ids_x[i]] > actualPoints_y[idxop_1]:
        idxop_1 = ids_x[i]

    for j in range(0, 3):
        if ids_x[i] == ids_y[j]:
            startindex = j
            idxqr_1 = ids_x[i]
            qrpoints = qrpoints + [[actualPoints_x[idxqr_1], actualPoints_y[idxqr_1]]]
        

outerpoints = outerpoints + [[actualPoints_x[idxop_1], actualPoints_y[idxop_1]]]

idxqr_2 = 0
for i in range(0, 3):
    if ids_x[i]!= idxqr_1 and ids_x[i]!= idxop_1:
        idxqr_2 = ids_x[i]
        qrpoints = qrpoints + [[actualPoints_x[idxqr_2], actualPoints_y[idxqr_2]]]

#good so far
idxqr_3 = ids_y[(startindex + 1) % 3]
idxop_3 = ids_y[(startindex + 1) % 3]
#now the horizontal group
for i in range(0, 3):       #over x
    if i == startindex:
        continue
    if actualPoints_x[ids_y[i]] < actualPoints_x[idxqr_3]:
        idxqr_3 = ids_y[i]
    if actualPoints_x[ids_y[i]] > actualPoints_x[idxop_3]:
        idxop_3 = ids_y[i]

qrpoints = qrpoints + [[actualPoints_x[idxqr_3], actualPoints_y[idxqr_3]]]

#get max point in x/y
idxop_2 = 0
for i in range(6, 8):
    for j in range(5, 8):
        if ids_x[i] == ids_y[j]:
            idxop_2 = ids_x[i]
            outerpoints = outerpoints + [[actualPoints_x[idxop_2], actualPoints_y[idxop_2]]]
            continue

#final outer point
outerpoints = outerpoints + [[actualPoints_x[idxop_3], actualPoints_y[idxop_3]]]

#get wax points
waxpoints = []

for i in range(0, 8):
    if i!=idxqr_1 and i!=idxqr_2 and i!=idxqr_3 and i!=idxop_1 and i!=idxop_2 and i!=idxop_3:
        waxpoints = waxpoints + [[actualPoints_x[i], actualPoints_y[i]]]

#sort waxpoints on y
if waxpoints[0][1] > waxpoints[1][1]:
    tmpx = waxpoints[0][0]
    tmpy = waxpoints[0][1]
    waxpoints[0][0] = waxpoints[1][0]
    waxpoints[0][1] = waxpoints[1][1]
    waxpoints[1][0] = tmpx
    waxpoints[1][1] = tmpy

qrstr = ''.join(str(e) for e in qrpoints)
otstr = ''.join(str(e) for e in outerpoints)
wxstr = ''.join(str(e) for e in waxpoints)
rl.write('Points QR '+qrstr+', Outer '+otstr+', Wax '+wxstr+'\n')
print "QR points", qrpoints
print "Outer points", outerpoints
print "Wax points", waxpoints

#points for transformation
src_points = []
dst_points = []
src_tests = []
dst_tests = []

#add outerpoints and their transform
transpoints = [[85, 1163], [686, 1163], [686, 77]]

for i in range(0, 3):
    src_points.append(outerpoints[i])
    dst_points.append(transpoints[i])

#add qr points and their transform
transqrpoints = [[82, 64], [82, 226], [244, 64]]

#add QR points
for i in range(0, 3):
    if i == 0:
        src_points.append(qrpoints[i])
        dst_points.append(transqrpoints[i])
    else:
        src_tests.append(qrpoints[i])
        dst_tests.append(transqrpoints[i])


print "Source points", src_points
print "Destination points", dst_points

#end of transformation point acquisition

with open(resultsfilenameroot + '.csv', "w") as myfile:
    myfile.write('points,'+str(len(qrpoints))+','+str(len(outerpoints))+',\n');

####################################################################################################
# James Sweet. Computer Science and Engineering. Notre Dame.
# 03/12/2014
# Single Value Decomposition code. This takes the over defined problem and solves for a
# rotaton and translation (Affine) matrix which is then applied to the image.
# Based on the DHARMA java code for mapping point-cloud views.
####################################################################################################
srcpoints = np.array(src_points, np.float32)
dstpoints = np.array(dst_points, np.float32)

np.set_printoptions(precision=4, suppress=True)

#use points to find perspective matrix
TI = cv2.getPerspectiveTransform(srcpoints, dstpoints)

# calculate errors by transforming points
maxerror = 0
for i in range(0, len(src_tests)):
    transformed_point = TI * np.matrix([src_tests[i][0], src_tests[i][1], 1.0]).T
    transformed_point.A1[0] = transformed_point.A1[0] / transformed_point.A1[2]
    transformed_point.A1[1] = transformed_point.A1[1] / transformed_point.A1[2]
    transformed_point.A1[2] = 1.0
    error = np.linalg.norm(np.array(transformed_point.A1[:2] - dst_tests[i]))
    if error > maxerror:
        maxerror = error

rl.write('Transformation maximum error,'+str(maxerror)+'\n')

print "Transformation maximum error,",maxerror
with open(resultsfilenameroot + '.csv', "a") as myfile:
    myfile.write('maximum_error,'+str(round(maxerror,2))+',\n');

# bail if error exceeds 15 pixels (relates to sample circle in relation to sample well)
if maxerror > 15:
    rl.close()
    print "Error: Transformation error exceeds threshold of 15 pixels.",filename
    sys.exit(5)

#if debug_images:
    #im_scaled = cv2.resize(orig_im, (601, 1086))
    #cv2.imwrite(filenameroot + '.scaled.png', im_scaled, [cv.CV_IMWRITE_PNG_COMPRESSION, 0])

#eye candy
im_warped = cv2.warpPerspective(orig_im, TI, (690 + 40, 1230 + 20),borderMode=cv2.BORDER_REPLICATE)
gim_warped = cv2.cvtColor(im_warped, cv.CV_BGR2GRAY)
fgim_warped = gim_warped.astype(np.float32)

#if debug_images:
#    cv2.imwrite(filenameroot + '.warped.png', im_warped, [cv.CV_IMWRITE_PNG_COMPRESSION, 0])

mask = np.zeros(im_warped.shape[0:2], np.uint8)
sim_warped = im_warped   # handle the case where neither black nor white balance

#### Find wax "bleed" thickness ###################################################################
# adaptive threshold for black/white
athresh = cv2.adaptiveThreshold(gim_warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 127, 0)
#if debug_images:
#cv2.imwrite(filenameroot + '.athresh.png', athresh, [cv.CV_IMWRITE_PNG_COMPRESSION, 0])

#find width of thickest bar and estimate the thin one
wax_width = 0;
on_wax = False
for i in range(280,375):
    if on_wax:
        if athresh[i][25] < 128: #12
            wax_width += 1
        else:
            break
    else:
        if athresh[i][25] < 128: #34
            on_wax = True
            wax_width += 1

#print it and save
print "Wax width",wax_width,(wax_width * 15) / 30
with open(resultsfilenameroot + '.csv', "a") as myfile:
    myfile.write('wax_width,'+str(wax_width)+',\n');

#### Find squares #################################################################################
white_square = (0, 0)
template_squares = cv2.imread("padscrs2.png", cv2.CV_LOAD_IMAGE_GRAYSCALE).astype(np.float32) / 255.0
result_squares = cv2.matchTemplate(fgim_warped, template_squares, cv.CV_TM_CCOEFF_NORMED)
sqminVal, sqmaxVal, sqminLoc, sqmaxLoc = cv2.minMaxLoc(result_squares)
#print "Squares at",sqmaxLoc[0]+120,sqmaxLoc[1]+76,"with threshold",sqmaxVal
if sqmaxVal > 0.80:
    #TODO read in the template offsets (120, 76)
    white_square = (sqmaxLoc[0]+120,sqmaxLoc[1]+76)
    print "Squares at", white_square, "with threshold", sqmaxVal

cellPoints = []
#points from artwork, chosen by artwork variable set by -a X.
comparePoints = [[[387, 214], [387, 1164]], [[387-17, 214], [387, 1164]], [[387+5, 214], [387-11, 1164]]]

#need to transform waxpoints first
for i in range(0, 2):
    transformed_point = TI * np.matrix([waxpoints[i][0], waxpoints[i][1], 1.0]).T
    transformed_point.A1[0] = transformed_point.A1[0] / transformed_point.A1[2]
    transformed_point.A1[1] = transformed_point.A1[1] / transformed_point.A1[2]
    transformed_point.A1[2] = 1.0
    cellPoints.append((transformed_point.A1[0], transformed_point.A1[1]))

#check order of points and add equivalent points from artwork
dist1 = np.linalg.norm(np.array([cellPoints[0][0] - 387, cellPoints[0][1] - 214]))
dist2 = np.linalg.norm(np.array([cellPoints[0][0] - 387, cellPoints[0][1] - 1164]))

#flip points if required
if dist1 > dist2:
    temp = cellPoints[0]
    cellPoints[0] = cellPoints[1]
    cellPoints[1] = temp
    print "Flipped",cellPoints[0]

print "Wax Points",cellPoints,"actual",comparePoints[0]

#print targets found
k= 0
for i in range(0, len(cellPoints)):
    cv2.circle(im_warped, (int(cellPoints[i][0]), int(cellPoints[i][1])), 17, (255, 255, 255, 255), 2)
    cv2.putText(im_warped, str(k), (int(cellPoints[i][0]), int(cellPoints[i][1])), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
    k = k + 1

#do SVD for rotation/translation?
if len(cellPoints) > 1:
    if artwork == -1:
        #do SVD for fringes targets
        TCPA = []
        for i in range(0, len(comparePoints)):
            TCPA.append(RotTrans2Points(cellPoints, comparePoints[i]))

        #find minimum angles, and select that as template values if artwork
        minangle = sys.float_info.max
        for i in range(0,len(comparePoints)):
            print "data",TCPA[i].A[0][1]
            iangl = math.fabs(math.asin(min(TCPA[i].A[0][1],1.0)))
            if iangl < minangle:
                minangle = iangl
                artwork = i

        print "Minimum angle was at index",artwork
        TCP = TCPA[artwork]
    else:
        TCP = RotTrans2Points(cellPoints, comparePoints[artwork])

    #get full matrix
    print "Mat",TCP.A[0][2],TCP.A[1][2]
    TICP = np.matrix([
        [TCP.A[0][0], TCP.A[0][1], TCP.A[0][2]],
        [TCP.A[1][0], TCP.A[1][1], TCP.A[1][2]],
        [0, 0, 1]
    ])

#flag artwork used in csv
with open(resultsfilenameroot + '.csv', "a") as myfile:
    myfile.write('artwork,'+str(artwork+1)+',\n');

# Calculate data values
# Handle Colour Squares
fout = sys.stdout

print >>fout,'File name,%s' % (filename)
print >> fout, 'i, j, red, green, blue, A'
print 'File name,%s' % (filename)
print 'i, j, red, green, blue, A'

colour_mask = np.zeros(im_warped.shape[0:2], np.uint8)

colour_square_center = [
    [[507, 132]],
    [[555, 106], [555, 152]],
    [[600, 106], [600, 152]]
]

A = 70
k = 0
#calculate offset if detected
square_offset = (0, 0)
if white_square[0] > 0 and white_square[1] > 0:
    square_offset = (colour_square_center[2][1][0] - white_square[0], colour_square_center[2][1][1] - white_square[1])

for i in range(0, len(colour_square_center)):
    for j in range(0, len(colour_square_center[i])):
        # Offset location by averages
        cx = colour_square_center[i][j][0] - square_offset[0]
        cy = colour_square_center[i][j][1] - square_offset[1]

        pt1 = (cx - 8, cy - 8)
        pt2 = (cx + 8, cy + 8)

        colour_mask.fill(0)
        cv2.rectangle(colour_mask, pt1, pt2, (255, 0, 0), 1)
        s = cv2.mean(sim_warped, colour_mask)

        with open(resultsfilenameroot + '.csv', "a") as myfile:
            myfile.write('square,'+str(i)+','+str(j)+','+str(round(s[0],2))+','+str(round(s[1],2))+','+str(round(s[2],2))+',\n');

        if i == 2 and j == 1:
            A = 255 - (s[0] + s[1] + s[2]) / 3
            #print "A value", A
            print >> fout, '%d, %d, %d, %d, %d, %d' % (i, j, s[0], s[1], s[2], A)
            print '%d, %d, %d, %d, %d, %d' % (i, j, s[0], s[1], s[2], A)
        else:
            print >> fout, '%d, %d, %d, %d, %d' % (i, j, s[0], s[1], s[2])
            print '%d, %d, %d, %d, %d' % (i, j, s[0], s[1], s[2])

        cv2.rectangle(im_warped, pt1, pt2, (255, 0, 0), 1)
        cv2.putText(im_warped, str(k), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        k = k + 1

print >> fout, ''

#Fringe lines
#fringe = [
#    [[706, 339], [706, 1095]],
#    [[653, 339], [653, 1095]],
#    [[69, 339], [69, 1095]]
#]

#put transformed markers on image
pnt1 = np.matrix([cellPoints[0][0], cellPoints[0][1], 1.0])
pnt2 = np.matrix([cellPoints[1][0], cellPoints[1][1], 1.0])
trans1 = TICP * pnt1.T
trans2 = TICP * pnt2.T
cv2.line(im_warped,(int(trans1[0])+10, int(trans1[1])),(int(trans1[0])-10, int(trans1[1])),(255,0,0),2)  # New blue line
cv2.line(im_warped,(int(trans1[0]), int(trans1[1])+10),(int(trans1[0]), int(trans1[1])-10),(255,0,0),2)
cv2.line(im_warped,(int(trans2[0])+10, int(trans2[1])),(int(trans2[0])-10, int(trans2[1])),(255,0,0),2)
cv2.line(im_warped,(int(trans2[0]), int(trans2[1])+10),(int(trans2[0]), int(trans2[1])-10),(255,0,0),2)

#print fringes
for i in range(0, 13):
    px = 706 - 53 * i
    pnt1 = np.matrix([px, 339, 1.0])
    pnt2 = np.matrix([px, 1095, 1.0])
    trans1 = TICP.I * pnt1.T
    trans2 = TICP.I * pnt2.T
    cv2.line(im_warped,(int(trans1[0]), int(trans1[1])),(int(trans2[0]), int(trans2[1])),(0,0,255),2)  # New red line
    cv2.line(im_warped,(px, 339),(px, 1095),(0,255,0),2)  # Drawing orig line

#actual transformed fringes
TALL = TICP * TI
#fringe_warped = cv2.warpAffine(cv2.warpAffine(orig_im, T, (690 + 40, 1200)), TCP, (690 + 40, 1200))
#fringe_warped = cv2.warpPerspective(im_warped, TICP, (690 + 40, 1220))
fringe_warped = cv2.warpPerspective(orig_im, TALL, (690 + 40, 1220),borderMode=cv2.BORDER_REPLICATE)

#print fringes/sample areas
buffer = 5 #sample area buffer
for i in range(0, 13):
    px = 706 - 53 * i
    #cv2.line(fringe_warped,(px, 339+wax_width/2),(px, 1095),(0,255,0),1)  # Drawing orig line
    cv2.line(fringe_warped,(px, 339+20),(px, 1095),(0,255,0),1)  # Drawing orig line
    #sample box
    #if i > 0:
    #    cv2.line(fringe_warped,(px+wax_width/4+buffer, 339+wax_width/2),(px+wax_width/4+buffer, 615),(0,0,255),1)  # Drawing orig line
    #    cv2.line(fringe_warped,(px+53-wax_width/4-buffer, 339+wax_width/2),(px+53-wax_width/4-buffer, 615),(0,0,255),1)  # Drawing orig line


#print marker points
targetloc = comparePoints[artwork]
cv2.line(fringe_warped,(targetloc[0][0],targetloc[0][1]-5),(targetloc[0][0],targetloc[0][1]+5),(0,255,0),1)
cv2.line(fringe_warped,(targetloc[0][0]-5,targetloc[0][1]),(targetloc[0][0]+5,targetloc[0][1]),(0,255,0),1)
cv2.line(fringe_warped,(targetloc[1][0],targetloc[1][1]-5),(targetloc[1][0],targetloc[1][1]+5),(0,255,0),1)
cv2.line(fringe_warped,(targetloc[1][0]-5,targetloc[1][1]),(targetloc[1][0]+5,targetloc[1][1]),(0,255,0),1)

#top bounding line
#cv2.line(fringe_warped,(70, 339+wax_width/2),(706, 339+wax_width/2),(0,255,0),1)

#output file
cv2.imwrite(resultsfilenameroot + '.processed.png', fringe_warped, [cv.CV_IMWRITE_PNG_COMPRESSION, 0])

# Print annotated image
#if debug_images:
#    cv2.imwrite(filenameroot + '.ann.png', im_warped, [cv.CV_IMWRITE_PNG_COMPRESSION, 0])

rl.close()
sys.exit(0)
