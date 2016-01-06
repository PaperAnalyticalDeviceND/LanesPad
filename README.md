# LanesPad
PAD repo for lanes cards

Information on the Paper Analytical Device Project can be found here: http://padproject.nd.edu

The PAD repository contains a mixture of Python and C++ code to allow the analysis of the Lanes Pads, from Notre Dame University, for detection of drug contents. The C++ code must be compiled, on Linux/OSX (assuming OpenCV libraries installed):
>cmake .
>make

To run the code:
>python lanes.py -r processed -t lanetemplate.png text/img.jpg
