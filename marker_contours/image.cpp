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

    //show image?
    bool show = false;

    if(argc >= 3 && strcmp("-i", argv[2]) == 0){
        show = true;
    }

	Mat imagein = imread(argv[1]);

	if(imagein.empty()){ cerr << "ERR: Unable to find image.\n" << endl;
		return -1;
	}
    
    float new_width = 600.0;
    
    //get image size
    //####std::cout << "Input size " << imagein.size().width << ", " << imagein.size().height << "." << std::endl;
    
    Mat image;
    
    float ratio = imagein.size().width / new_width;
    
    //####std::cout << "Ratio " << ratio << "." << std::endl;
    
    resize(imagein, image, Size(new_width, (imagein.size().height  * new_width )/ imagein.size().width), 0, 0, INTER_LINEAR );
    
    //get image size
    //####std::cout << "Working size " << image.size().width << ", " << image.size().height << "." << std::endl;
	
    //get mid point
    Point2f midpoint = Point2f(image.size().width/2, image.size().height/2);

	// Creation of Intermediate 'Image' Objects required later
	Mat gray(image.size(), CV_MAKETYPE(image.depth(), 1));			// To hold Grayscale Image
    Mat gray_blur(image.size(), CV_MAKETYPE(image.depth(), 1));			// To hold Grayscale Image
	Mat edges(image.size(), CV_MAKETYPE(image.depth(), 1));			// To hold Grayscale Image
    
    //vectors for contour data
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	int key = 0;

    cvtColor(image,gray,CV_RGB2GRAY);		// Convert Image captured from Image Input to GrayScale
    /// Reduce noise with a kernel 3x3
    blur( gray, gray_blur, Size(2,2) );

    Canny(gray_blur, edges, 40 , 150, 3);		// Apply Canny edge detection on the gray image


    findContours( edges, contours, hierarchy, RETR_TREE, CV_CHAIN_APPROX_SIMPLE); // Find contours with hierarchy

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
        float dist = std::min(std::min(std::min(mc[i].x, new_width - mc[i].x), mc[i].y), image.size().height - mc[i].y);
        
        Rect box = boundingRect(contours[Markers[i]]);
        
        float dia = std::max(box.width, box.height) / 2;

        //only add it if sensible
        if(dia < 26){
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


// EOF
