#!/usr/bin/env python
import sys
import subprocess
#import cv
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
if len(sys.argv) < 2:
    print 'Usage: ' + sys.argv[
        0] + '[-i value] [-s] [-w] [-b] [-g] [-d] [-m] [-t templatefile] [-o] imagefile [guess1 [guess2]]'
    print '-i is Interactive: use mouse to select QR code rectangle.'
    print '      0 no interaction, 1 interact if fails automatic, 2 force interactive.'
    print '-s is Smooth: blur the image a little (can be used multiple times.)'
    print '-g is graphics: show partial results in windows, press a key to continue.'
    print '-w is white balance: make average color in white color square pure white.'
    print '-b is "black balance": make average color in black color square pure black.'
    print '-m is Matrix: print mapping matrix.'
    print '-l is tempLate method: Use to force template matching, not line search.'
    print '-r is results in specified sub-folder.'
    print '-a is artwork: Pick wax artwork used.'
    sys.exit(-1)

optlist, args = getopt.getopt(sys.argv[1:], 'wbgdsi:mlt:o:c:r:a:')

save_correlation = False
debug_images = False
whitebalance = False
blackbalance = False
graphics = False
smoo = 0
interactive = 1
mouseflag = False
mappingmatrix = False
templatefile = 'template2.png'
templatemethod = False
resultsfile = ""
file_rights = 'a'
resultsfolder = ""
artwork = -1

calibrationFile = ""

for o, a in optlist:
    if o == '-w':
        whitebalance = True
    elif o == '-b':
        blackbalance = True
    elif o == '-g':
        graphics = True
    elif o == "-d":
        debug_images = True
    elif o == '-s':
        smoo = smoo + 1
    elif o == '-i':
        interactive = int(float(a))
    elif o == '-a':
        artwork = int(float(a)) - 1
        if artwork > 2 or artwork < 0:
            artwork = -1
    elif o == '-t':
        templatefile = a
    elif o == "-o":
        resultsfile = a
    elif o == "-m":
        mappingmatrix = True
    elif o == "-l":
        templatemethod = True
    elif o == "-c":
        calibrationFile = a
    elif o == "-r":
        resultsfolder = a
    else:
        print 'Unhandled option: ', o
        sys.exit(-2)

print 'args: ', args

#get filenames and roots
filename = args[0]
filenameroot = '.'.join(filename.split('.')[:-1])
resultsfilenameroot = filenameroot
if resultsfolder != "":
    resultsfilenameroot = '/'.join(filename.split('/')[:-1])+'/'+resultsfolder+'/'+'.'.join(filename.split('/')[-1].split('.')[:-1])
#print "Results root:",resultsfilenameroot
if resultsfile == "auto":
    resultsfile = filenameroot+'.csv'
    file_rights = 'w'

if len(args) > 2:
    guess1 = args[1]
    if len(args) > 3:
        guess2 = args[2]
    else:
        guess2 = None
else:
    guess1 = None
    guess2 = None

# OK load image
print 'filename is :', filename

orig_im = cv2.imread(filename)
(h, w, p) = orig_im.shape

#need to rotate image?
if w>h:
    orig_im = np.rot90( orig_im, 1 )

(h, w, p) = orig_im.shape

#eye candy
im_warped = orig_im.copy() #cv2.warpPerspective(orig_im, TI, (690 + 40, 1230 + 20),borderMode=cv2.BORDER_REPLICATE)

gim_warped = cv2.cvtColor(im_warped, cv2.COLOR_BGR2GRAY)
fgim_warped = gim_warped.astype(np.float32)

if graphics:
    cv2.imshow("Warped image", im_warped)
    cv2.waitKey(0)

if debug_images:
    cv2.imwrite(filenameroot + '.warped.png', im_warped, [cv2.IMWRITE_PNG_COMPRESSION, 0])

mask = np.zeros(im_warped.shape[0:2], np.uint8)
sim_warped = im_warped   # handle the case where neither black nor white balance

#### Find wax "bleed" thickness ###################################################################
# adaptive threshold for black/white
athresh = cv2.adaptiveThreshold(gim_warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 127, 0)
if debug_images:
    cv2.imwrite(filenameroot + '.athresh.png', athresh, [cv2.IMWRITE_PNG_COMPRESSION, 0])

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
template_squares = cv2.imread("padscrs2.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
result_squares = cv2.matchTemplate(fgim_warped, template_squares, cv2.TM_CCOEFF_NORMED)
sqminVal, sqmaxVal, sqminLoc, sqmaxLoc = cv2.minMaxLoc(result_squares)
#print "Squares at",sqmaxLoc[0]+120,sqmaxLoc[1]+76,"with threshold",sqmaxVal
if sqmaxVal > 0.80:
    #TODO read in the template offsets (120, 76)
    white_square = (sqmaxLoc[0]+120,sqmaxLoc[1]+76)
    print "Squares at", white_square, "with threshold", sqmaxVal

#### Use template matching to gather evidence for the twelve cells. ###############################
# Load cell template image
# this cell template was chopped out of a normalized image.
template = cv2.imread(templatefile, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
(ch, cw) = template.shape

#drop blue channel
coefficients = [0.163, 0.837, 0]#[0.163, 0.587, 0.0]#, bgr
m = np.array(coefficients).reshape((1,3))
im_warped_nb = cv2.transform(im_warped, m)
fgim_warped_nb = im_warped_nb.astype(np.float32)
if debug_images:
    cv2.imwrite(filenameroot + '.warped_nb.png', fgim_warped_nb, [cv2.IMWRITE_PNG_COMPRESSION, 0])

result = cv2.matchTemplate(fgim_warped_nb, template, cv2.TM_CCOEFF_NORMED)
if save_correlation:
    np.savetxt("targetresult.txt", result)

cellPoints = []
#points from artwork, chosen by artwork variable set by -a X.
#comparePoints = [[[387, 214], [387, 1164]], [[387-17, 214], [387, 1164]], [[387+5, 214], [387-11, 1164]]]
#comparePoints = [[[387, 214], [387, 1164]], [[387, 214], [387, 1164]], [[387, 214], [387, 1164]]]
comparePoints = [[387, 214], [387, 1164]]

####################################################################################################
# Chris Sweet. Center for Research Computing. Notre Dame.
# 05/04/2014
# Replacement code for finding points whose correlation exceeds 75%. New method finds the global
# maximum then masks out an area around this point equal to the template size. The routine then finds
# the next maximum etc. until the required number is found or the correlation falls below a set level.
####################################################################################################
#get maximum points until we have all three or the certainty is <=threshold
cellmask = np.ones(result.shape, np.uint8)
cellmaxVal = 1
cellthr = 0.70

while len(cellPoints) < 2 and cellmaxVal > cellthr:
    cellminVal, cellmaxVal, cellminLoc, cellmaxLoc = cv2.minMaxLoc(result, cellmask);
    if cellmaxVal <= cellthr:
        break
    print "Max cell point location", cellmaxLoc, ",", cellmaxVal
    #TODO read in the template offsets (2, 1)
    cellPoints.append((cellmaxLoc[0] + cw / 2.0 - 0, cellmaxLoc[1] + ch / 2.0 - 0))
    rect = [[cellmaxLoc[0] - cw / 2, cellmaxLoc[1] - ch / 2], [cellmaxLoc[0] + cw / 2, cellmaxLoc[1] - ch / 2],
            [cellmaxLoc[0] + cw / 2, cellmaxLoc[1] + ch / 2], [cellmaxLoc[0] - cw / 2, cellmaxLoc[1] + ch / 2]]
    poly = np.array([rect], dtype=np.int32)
    cv2.fillPoly(cellmask, poly, 0)
####################################################################################################
# bail if error exceeds 15 pixels (relates to sample circle in relation to sample well)
if len(cellPoints) != 2:
    print "Error: Wax target not found with > 0.70 confidence.",filename
    sys.exit(-5)

#check order of points and add equivalent points from artwork
dist1 = np.linalg.norm(np.array([cellPoints[0][0] - 387, cellPoints[0][1] - 214]))
dist2 = np.linalg.norm(np.array([cellPoints[0][0] - 387, cellPoints[0][1] - 1164]))

#flip points if required
if dist1 > dist2:
    temp = cellPoints[0]
    cellPoints[0] = cellPoints[1]
    cellPoints[1] = temp
    print "Flipped",cellPoints[0]

print "Wax Points",cellPoints,"actual",comparePoints

#print targets found
k= 0
for i in range(0, len(cellPoints)):
    cv2.circle(im_warped, (int(cellPoints[i][0]), int(cellPoints[i][1])), 17, (255, 255, 255, 255), 2)
    cv2.putText(im_warped, str(k), (int(cellPoints[i][0]), int(cellPoints[i][1])), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
    k = k + 1

#do SVD for rotation/translation?
if len(cellPoints) > 1:
    TCP = RotTrans2Points(cellPoints, comparePoints)

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
if resultsfile == "":
	fout = sys.stdout
else:
	fout = file(resultsfile, file_rights)

print >>fout,'File name,%s' % (filename)
print >> fout, 'i, j, red, green, blue, A'
if resultsfile != "":
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
            if resultsfile != "":
                print '%d, %d, %d, %d, %d, %d' % (i, j, s[0], s[1], s[2], A)
        else:
            print >> fout, '%d, %d, %d, %d, %d' % (i, j, s[0], s[1], s[2])
            if resultsfile != "":
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
fringe_warped = cv2.warpPerspective(orig_im, TICP, (690 + 40, 1220),borderMode=cv2.BORDER_REPLICATE)

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
cv2.line(fringe_warped,(comparePoints[0][0],comparePoints[0][1]-5),(comparePoints[0][0],comparePoints[0][1]+5),(0,255,0),1)
cv2.line(fringe_warped,(comparePoints[0][0]-5,comparePoints[0][1]),(comparePoints[0][0]+5,comparePoints[0][1]),(0,255,0),1)
cv2.line(fringe_warped,(comparePoints[1][0],comparePoints[1][1]-5),(comparePoints[1][0],comparePoints[1][1]+5),(0,255,0),1)
cv2.line(fringe_warped,(comparePoints[1][0]-5,comparePoints[1][1]),(comparePoints[1][0]+5,comparePoints[1][1]),(0,255,0),1)

#top bounding line
#cv2.line(fringe_warped,(70, 339+wax_width/2),(706, 339+wax_width/2),(0,255,0),1)

#output file
cv2.imwrite(resultsfilenameroot + '.processed.png', fringe_warped, [cv2.IMWRITE_PNG_COMPRESSION, 0])

#show annotated?
if graphics:
    cv2.imshow('annotated', im_warped)
    cv2.waitKey(0)

# Print annotated image
if debug_images:
    cv2.imwrite(filenameroot + '.ann.png', im_warped, [cv2.IMWRITE_PNG_COMPRESSION, 0])

sys.exit(0)
