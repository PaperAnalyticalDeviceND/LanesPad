//______________________________________________________________________________________
// Program : OpenCV based QR code Detection and Retrieval
// Author  : Bharath Prabhuswamy
//______________________________________________________________________________________

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

const int CV_QR_NORTH = 0;
const int CV_QR_EAST = 1;
const int CV_QR_SOUTH = 2;
const int CV_QR_WEST = 3;

float cv_distance(Point2f P, Point2f Q);					// Get Distance between two points
float cv_lineEquation(Point2f L, Point2f M, Point2f J);		// Perpendicular Distance of a Point J from line formed by Points L and M; Solution to equation of the line Val = ax+by+c 
float cv_lineSlope(Point2f L, Point2f M, int& alignement);	// Slope of a line by two Points L and M on it; Slope of line, S = (x1 -x2) / (y1- y2)
void cv_getVertices(vector<vector<Point> > contours, int c_id,float slope, vector<Point2f>& X);
void cv_updateCorner(Point2f P, Point2f ref ,float& baseline,  Point2f& corner);
void cv_updateCornerOr(int orientation, vector<Point2f> IN, vector<Point2f> &OUT);
bool getIntersectionPoint(Point2f a1, Point2f a2, Point2f b1, Point2f b2, Point2f& intersection);
float cross(Point2f v1,Point2f v2);

//sort method
bool orderfunction (Point2f i,Point2f j) { return (i.y<j.y); }

// Start of Main Loop
//------------------------------------------------------------------------------------------------------------------------
int main ( int argc, char **argv )
{

    bool show = false;

    if(argc >= 3 && strcmp("-i", argv[2]) == 0){
        show = true;
    }

	Mat imagein = imread(argv[1]);

	if(imagein.empty()){ cerr << "ERR: Unable to find image.\n" << endl;
		return -1;
	}
    
    //get image size
    //####std::cout << "Input size " << imagein.size().width << ", " << imagein.size().height << "." << std::endl;
    
    Mat image;
    
    float ratio = imagein.size().width / 600.0;
    
    //####std::cout << "Ratio " << ratio << "." << std::endl;
    
    resize(imagein, image, Size(600, (imagein.size().height  * 600 )/ imagein.size().width), 0, 0, INTER_LINEAR );
    
    //get image size
    //####std::cout << "Working size " << image.size().width << ", " << image.size().height << "." << std::endl;
	
    //get mid point
    Point2f midpoint = Point2f(image.size().width/2, image.size().height/2);

	// Creation of Intermediate 'Image' Objects required later
	Mat gray(image.size(), CV_MAKETYPE(image.depth(), 1));			// To hold Grayscale Image
    Mat gray_blur(image.size(), CV_MAKETYPE(image.depth(), 1));			// To hold Grayscale Image
	Mat edges(image.size(), CV_MAKETYPE(image.depth(), 1));			// To hold Grayscale Image
	//Mat traces(image.size(), CV_8UC3);								// For Debug Visuals
	//Mat qr,qr_raw,qr_gray,qr_thres;
	    
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	int mark,A,B,C,top,right,bottom,median1,median2,outlier;
	float AB,BC,CA, dist,slope, areat,arear,areab, large, padding;
	
	int align,orientation;

	int DBG=1;						// Debug Flag

	int key = 0;

    //while(key != 'q')				// While loop to query for Image Input frame
    //{
    //traces = Scalar(0,0,0);
    //qr_raw = Mat::zeros(100, 100, CV_8UC3 );
    //qr = Mat::zeros(100, 100, CV_8UC3 );
    //qr_gray = Mat::zeros(100, 100, CV_8UC1);
    //qr_thres = Mat::zeros(100, 100, CV_8UC1);
    
    // capture >> image;				// For Video input		// Capture Image from Image Input

    cvtColor(image,gray,CV_RGB2GRAY);		// Convert Image captured from Image Input to GrayScale
    /// Reduce noise with a kernel 3x3
    blur( gray, gray_blur, Size(3,3) );

    Canny(gray_blur, edges, 40 , 150, 3);		// Apply Canny edge detection on the gray image


    findContours( edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE); // Find contours with hierarchy

    mark = 0;								// Reset all detected marker count for this frame


    //for( int i = 0; i < contours.size(); i++ )
    //{
     //   mu[i] = moments( contours[i], false );
    //	mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
    //}


    // Start processing the contour data

    // Find Three repeatedly enclosed contours A,B,C
    // NOTE: 1. Contour enclosing other contours is assumed to be the three Alignment markings of the QR code.
    // 2. Alternately, the Ratio of areas of the "concentric" squares can also be used for identifying base Alignment markers.
    // The below demonstrates the first method
    vector<int> Markers;
    for( int i = 0; i < contours.size(); i++ )
    {
        int k=i;
        int c=0;

        while(hierarchy[k][2] != -1)
        {
            k = hierarchy[k][2] ;
            c = c+1;
        }
        if(hierarchy[k][2] != -1)
        c = c+1;

        if (c >= 5)
        {
            Markers.push_back(i);
            //if (mark == 0)		A = i;
            //else if  (mark == 1)	B = i;		// i.e., A is already found, assign current contour to B
            //else if  (mark == 2)	C = i;		// i.e., A and B are already found, assign current contour to C
            //mark = mark + 1 ;
        }
    }
    
    // Get Moments for all Contours and the mass centers
    //vector<Moments> mu(contours.size());
    vector<Point2f> mc(Markers.size());
    vector<Point2f> order;
    
    for( int i=0; i < Markers.size(); i++){
        Scalar color( 128,0,0);//rand()&200, rand()&200, rand()&200 );
        //if(Markers[i] > 1000){
        drawContours(image, contours, Markers[i], color, 2, 8, hierarchy);
        
        Moments mum = moments( contours[Markers[i]], false );
        mc[i] = Point2f( mum.m10/mum.m00 , mum.m01/mum.m00 );

        //circle( image, mc[i], 10.0, Scalar( 0, 0, 255 ), 1, 8 );
        //}
        //std::cout << "Markers " << Markers[i] << " index " << i << ", dist to mid " << cv_distance(midpoint, mc[i]) << "." << std::endl;
        
        //calculate distance to nearest edge
        float dist = std::min(std::min(std::min(mc[i].x, 600 - mc[i].x), mc[i].y), image.size().height - mc[i].y);
        
        Rect box = boundingRect(contours[Markers[i]]);
        
        float dia = std::max(box.width, box.height) / 2;

        //only add it if sensible
        if(dia < 40){
            order.push_back(Point2f(i, dist));
        }
    }

    //sort vector
    std::sort(order.begin(), order.end(), orderfunction);
    
    //get size up to 6
    int sz = std::min(6, (int)order.size());
    
    //loop
    for( int j=0; j<sz; j++){
        int i = order[j].x;
        
        Rect box = boundingRect(contours[Markers[i]]);
        
        float dia = std::max(box.width, box.height) / 2;
        
        circle( image, mc[i], dia, Scalar( 0, 0, 255 ), 1, 8 );
        
        //std::cout << "i "<< i << ", " << order[j] << " Markers " << Markers[i] << " index " << i << ", dist to edge " << order[j].y << "." << std::endl;
        std::cout << "Point: "<< int(mc[i].y * ratio + 0.5) << ", " << int(mc[i].x * ratio  + 0.5) << ", " << int(dia * ratio + 0.5) << std::endl;
        
    }
    
    if(show){
        imshow ( "Image", image );

        key = waitKey(500000);	// OPENCV: wait for 1ms before accessing next frame
    }

	return 0;
}

// End of Main Loop
//--------------------------------------------------------------------------------------


// Routines used in Main loops

// Function: Routine to get Distance between two points
// Description: Given 2 points, the function returns the distance

float cv_distance(Point2f P, Point2f Q)
{
	return sqrt(pow(abs(P.x - Q.x),2) + pow(abs(P.y - Q.y),2)) ; 
}


// Function: Perpendicular Distance of a Point J from line formed by Points L and M; Equation of the line ax+by+c=0
// Description: Given 3 points, the function derives the line quation of the first two points,
//	  calculates and returns the perpendicular distance of the the 3rd point from this line.

float cv_lineEquation(Point2f L, Point2f M, Point2f J)
{
	float a,b,c,pdist;

	a = -((M.y - L.y) / (M.x - L.x));
	b = 1.0;
	c = (((M.y - L.y) /(M.x - L.x)) * L.x) - L.y;
	
	// Now that we have a, b, c from the equation ax + by + c, time to substitute (x,y) by values from the Point J

	pdist = (a * J.x + (b * J.y) + c) / sqrt((a * a) + (b * b));
	return pdist;
}

// Function: Slope of a line by two Points L and M on it; Slope of line, S = (x1 -x2) / (y1- y2)
// Description: Function returns the slope of the line formed by given 2 points, the alignement flag
//	  indicates the line is vertical and the slope is infinity.

float cv_lineSlope(Point2f L, Point2f M, int& alignement)
{
	float dx,dy;
	dx = M.x - L.x;
	dy = M.y - L.y;
	
	if ( dy != 0)
	{	 
		alignement = 1;
		return (dy / dx);
	}
	else				// Make sure we are not dividing by zero; so use 'alignement' flag
	{	 
		alignement = 0;
		return 0.0;
	}
}



// Function: Routine to calculate 4 Corners of the Marker in Image Space using Region partitioning
// Theory: OpenCV Contours stores all points that describe it and these points lie the perimeter of the polygon.
//	The below function chooses the farthest points of the polygon since they form the vertices of that polygon,
//	exactly the points we are looking for. To choose the farthest point, the polygon is divided/partitioned into
//	4 regions equal regions using bounding box. Distance algorithm is applied between the centre of bounding box
//	every contour point in that region, the farthest point is deemed as the vertex of that region. Calculating
//	for all 4 regions we obtain the 4 corners of the polygon ( - quadrilateral).
void cv_getVertices(vector<vector<Point> > contours, int c_id, float slope, vector<Point2f>& quad)
{
	Rect box;
	box = boundingRect( contours[c_id]);
	
	Point2f M0,M1,M2,M3;
	Point2f A, B, C, D, W, X, Y, Z;

	A =  box.tl();
	B.x = box.br().x;
	B.y = box.tl().y;
	C = box.br();
	D.x = box.tl().x;
	D.y = box.br().y;


	W.x = (A.x + B.x) / 2;
	W.y = A.y;

	X.x = B.x;
	X.y = (B.y + C.y) / 2;

	Y.x = (C.x + D.x) / 2;
	Y.y = C.y;

	Z.x = D.x;
	Z.y = (D.y + A.y) / 2;

	float dmax[4];
	dmax[0]=0.0;
	dmax[1]=0.0;
	dmax[2]=0.0;
	dmax[3]=0.0;

	float pd1 = 0.0;
	float pd2 = 0.0;

	if (slope > 5 || slope < -5 )
	{

	    for( int i = 0; i < contours[c_id].size(); i++ )
	    {
		pd1 = cv_lineEquation(C,A,contours[c_id][i]);	// Position of point w.r.t the diagonal AC 
		pd2 = cv_lineEquation(B,D,contours[c_id][i]);	// Position of point w.r.t the diagonal BD

		if((pd1 >= 0.0) && (pd2 > 0.0))
		{
		    cv_updateCorner(contours[c_id][i],W,dmax[1],M1);
		}
		else if((pd1 > 0.0) && (pd2 <= 0.0))
		{
		    cv_updateCorner(contours[c_id][i],X,dmax[2],M2);
		}
		else if((pd1 <= 0.0) && (pd2 < 0.0))
		{
		    cv_updateCorner(contours[c_id][i],Y,dmax[3],M3);
		}
		else if((pd1 < 0.0) && (pd2 >= 0.0))
		{
		    cv_updateCorner(contours[c_id][i],Z,dmax[0],M0);
		}
		else
		    continue;
             }
	}
	else
	{
		int halfx = (A.x + B.x) / 2;
		int halfy = (A.y + D.y) / 2;

		for( int i = 0; i < contours[c_id].size(); i++ )
		{
			if((contours[c_id][i].x < halfx) && (contours[c_id][i].y <= halfy))
			{
			    cv_updateCorner(contours[c_id][i],C,dmax[2],M0);
			}
			else if((contours[c_id][i].x >= halfx) && (contours[c_id][i].y < halfy))
			{
			    cv_updateCorner(contours[c_id][i],D,dmax[3],M1);
			}
			else if((contours[c_id][i].x > halfx) && (contours[c_id][i].y >= halfy))
			{
			    cv_updateCorner(contours[c_id][i],A,dmax[0],M2);
			}
			else if((contours[c_id][i].x <= halfx) && (contours[c_id][i].y > halfy))
			{
			    cv_updateCorner(contours[c_id][i],B,dmax[1],M3);
			}
	    	}
	}

	quad.push_back(M0);
	quad.push_back(M1);
	quad.push_back(M2);
	quad.push_back(M3);
	
}

// Function: Compare a point if it more far than previously recorded farthest distance
// Description: Farthest Point detection using reference point and baseline distance
void cv_updateCorner(Point2f P, Point2f ref , float& baseline,  Point2f& corner)
{
    float temp_dist;
    temp_dist = cv_distance(P,ref);

    if(temp_dist > baseline)
    {
        baseline = temp_dist;			// The farthest distance is the new baseline
        corner = P;						// P is now the farthest point
    }
	
}

// Function: Sequence the Corners wrt to the orientation of the QR Code
void cv_updateCornerOr(int orientation, vector<Point2f> IN,vector<Point2f> &OUT)
{
	Point2f M0,M1,M2,M3;
    	if(orientation == CV_QR_NORTH)
	{
		M0 = IN[0];
		M1 = IN[1];
	 	M2 = IN[2];
		M3 = IN[3];
	}
	else if (orientation == CV_QR_EAST)
	{
		M0 = IN[1];
		M1 = IN[2];
	 	M2 = IN[3];
		M3 = IN[0];
	}
	else if (orientation == CV_QR_SOUTH)
	{
		M0 = IN[2];
		M1 = IN[3];
	 	M2 = IN[0];
		M3 = IN[1];
	}
	else if (orientation == CV_QR_WEST)
	{
		M0 = IN[3];
		M1 = IN[0];
	 	M2 = IN[1];
		M3 = IN[2];
	}

	OUT.push_back(M0);
	OUT.push_back(M1);
	OUT.push_back(M2);
	OUT.push_back(M3);
}

// Function: Get the Intersection Point of the lines formed by sets of two points
bool getIntersectionPoint(Point2f a1, Point2f a2, Point2f b1, Point2f b2, Point2f& intersection)
{
    Point2f p = a1;
    Point2f q = b1;
    Point2f r(a2-a1);
    Point2f s(b2-b1);

    if(cross(r,s) == 0) {return false;}

    float t = cross(q-p,s)/cross(r,s);

    intersection = p + t*r;
    return true;
}

float cross(Point2f v1,Point2f v2)
{
    return v1.x*v2.y - v1.y*v2.x;
}

// EOF
