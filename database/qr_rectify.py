#!/usr/bin/python

####################################################################################################
# Chris Sweet. Computer Science and Engineering. Notre Dame/CRC.
# 08/01/2014
# Separate QR and Marker points and remove additional points
####################################################################################################

#imports
import numpy as np
from sys import argv
import zbar
from PIL import Image, ImageEnhance
#import cv
import cv2

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

            print "Serial number ", serial_no

            #output location
            resultsfilenameroot = '/'.join(filename.split('/')[:-2])+'/pre_correct/'+'.'.join(filename.split('/')[-1].split('.')[:-2])+ '.precorrected.png'

            #get location of input file
            #resultsfilenameroot = "/var/www/html/joomla/images/padimages/msh/"

            #get location of QR code
            topLeftCorners, bottomLeftCorners, bottomRightCorners, topRightCorners = [item for item in symbol.location]
            #print topLeftCorners, bottomLeftCorners, bottomRightCorners, topRightCorners

            #collect data
            qr_edge_points = np.array([[50, 33], [50, 256], [273, 256], [273, 33]], np.float32)
            if artwork == 1:
                #float dest_points[][] = {{85, 1163, 686, 1163, 686, 77, 244, 64, 82, 64, 82, 226},
                #{85, 1163, 686, 1163, 686, 77, 255, 64, 82, 64, 82, 237}};
                qr_edge_points = np.array([[50, 33], [50, 267], [284, 267], [284, 33]], np.float32)

            src_points = np.array([topLeftCorners, bottomLeftCorners, bottomRightCorners, topRightCorners], np.float32)
            #qr_edge_points = np.array([[50, 33], [50, 256], [273, 33]], np.float32)
            #src_points = np.array([topLeftCorners, bottomLeftCorners, topRightCorners], np.float32)
            #print "Source", src_points, qr_edge_points

            #read in file
            orig_im = cv2.imread(filename)
            rows, cols, depth = orig_im.shape

            #orthoganialize
            #figure out orientation
            if abs(src_points[0][0] - src_points[1][0]) < abs(src_points[0][1] - src_points[1][1]):
                #if here then QR upright and we can fix across x's
                print "Orientation upright"
                src_points[0][0] = src_points[1][0] = int((src_points[0][0] + src_points[1][0]) / 2)
                src_points[2][0] = src_points[3][0] = int((src_points[2][0] + src_points[3][0]) / 2)
                #and y's
                src_points[0][1] = src_points[3][1] = int((src_points[0][1] + src_points[3][1]) / 2)
                src_points[1][1] = src_points[2][1] = int((src_points[1][1] + src_points[2][1]) / 2)

                sf = abs((qr_edge_points[0][0] - qr_edge_points[3][0]) / (src_points[0][0] - src_points[3][0]))

                op_sz = (int(cols * sf), int(rows * sf))
            else:
                #if here then QR upright and we can fix across y's
                print "Orientation on side"
                src_points[0][1] = src_points[1][1] = int((src_points[0][1] + src_points[1][1]) / 2)
                src_points[2][1] = src_points[3][1] = int((src_points[2][1] + src_points[3][1]) / 2)
                #and x's
                src_points[0][0] = src_points[3][0] = int((src_points[0][0] + src_points[3][0]) / 2)
                src_points[1][0] = src_points[2][0] = int((src_points[1][0] + src_points[2][0]) / 2)

                sf = abs((qr_edge_points[0][0] - qr_edge_points[3][0]) / (src_points[0][1] - src_points[3][1]))

                op_sz = (int(rows * sf), int(cols * sf))

            #print src_points[0], src_points[1], src_points[2], src_points[3], src_points[0:3]

            #use points to find perspective matrix
            #TI = cv2.getPerspectiveTransform(src_points, qr_edge_points)
            TI = cv2.getAffineTransform(src_points[0:3], qr_edge_points[0:3])

            #correct it
            #im_warped = cv2.warpPerspective(orig_im, TI, (690 + 40, 1230 + 20),borderMode=cv2.BORDER_REPLICATE)
            #print sf, op_sz
            im_warped = cv2.warpAffine(orig_im, TI, (800, 1300), borderMode=cv2.BORDER_REPLICATE)

            #write out corrected file
            #cv2.imwrite(resultsfilenameroot + serial_no + '.png', im_warped, [cv.CV_IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(resultsfilenameroot, im_warped, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            #print "Image saved to",serial_no + '.png'

            #write sn to file
            #f = open(resultsfilenameroot + "serial_numbers.txt", "a+")
            #f.write(serial_no + '\n')
            #f.close()

        else: #error?
            print "Serial number not found in QR code"

    # clean up
    del(image)

#check we have an inline image
if len(argv) < 2: exit(1)

#scan image
scan_rectify_image(argv[1])
