//______________________________________________________________________________________
// Program : OpenCV based QR code Detection and Retrieval
// Author  : Bharath Prabhuswamy
//______________________________________________________________________________________

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

struct data_point {
    data_point(int ii, float id, float idi, Point2f imc){
        i = ii;
        dist = id;
        dia = idi;
        mc = imc;
        valid = true;
    }
    int i;
    float dist;
    float dia;
    Point2f mc;
    bool valid;
};

//sort method
bool orderfunction (data_point i,data_point j) { return (i.dist<j.dist); }

void balance_white(cv::Mat mat);

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
    
    if(imagein.size().width >  imagein.size().height){
        transpose(imagein, imagein);
        flip(imagein, imagein, 1);
    }

    balance_white(imagein);
    
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

    Canny(gray, edges, 40 , 150, 3);		// Apply Canny edge detection on the gray image


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

        if (c >= 3)
        {
            Markers.push_back(i);
        }
        //if( c > 1)
          //  std::cout << "Depth  " << c << " size " << i << std::endl;
    }
    
    // Get Moments for all Contours and the mass centers
    vector<data_point> order;
    
    for( int i=0; i < Markers.size(); i++){
        Scalar color( 0,128,0);
        drawContours(image, contours, Markers[i], color, 2, 8, hierarchy);
        
        Moments mum = moments( contours[Markers[i]], false );
        Point2f mc = Point2f( mum.m10/mum.m00 , mum.m01/mum.m00 );

        //std::cout << "Markers " << Markers[i] << " index " << i << ", dist to mid " << cv_distance(midpoint, mc[i]) << "." << std::endl;
        
        //calculate distance to nearest edge
        float dist = std::min(std::min(std::min(mc.x, new_width - mc.x), mc.y), image.size().height - mc.y);
        
        Rect box = boundingRect(contours[Markers[i]]);
        
        float dia = std::max(box.width, box.height) / 2;

        //only add it if sensible
        if(dia < 30){
            order.push_back(data_point(i, dist, dia, mc));
        }
    }

    //remove duplicates by taking the largest when points close
    for( int i=0; i < order.size(); i++){
        if(order[i].valid){
            for( int j=i+1; j < order.size(); j++){
                if(order[j].valid){
                    const float ix = order[i].mc.x;
                    const float iy = order[i].mc.y;
                    const float jx = order[j].mc.x;
                    const float jy = order[j].mc.y;
                    
                    if(fabs(ix-jx) < 5 && fabs(iy-jy) < 5){
                        if(order[i].dia < order[j].dia){
                            order[i].valid = false;
                            break;
                        }else{
                            order[j].valid = false;
                        }
                    }
                }
            }
        }
    }
    
    //sort vector
    std::sort(order.begin(), order.end(), orderfunction);
    
    //count points
    int pcount = 0;
    
    //loop
    for( int j=0; j<order.size(); j++){
        if(order[j].valid){
            int i = order[j].i;
            float dia = order[j].dia;
            Point2f mcd = order[j].mc;
            
            circle( image, mcd, dia, Scalar( 0, 0, 255 ), 1, 8 );
            
            //std::cout << "i "<< i << ", " << order[j] << " Markers " << Markers[i] << " index " << i << ", dist to edge " << order[j].y << "." << std::endl;
            std::cout << "Point: "<< int(mcd.y * ratio + 0.5) << ", " << int(mcd.x * ratio  + 0.5) << ", " << int(dia * ratio + 0.5) << std::endl;
            if(pcount++ >= 5) break;
        }
    }
    
    if(show){
        imshow ( "Image", image );

        key = waitKey(0);	// OPENCV: wait for 1ms before accessing next frame
    }

	return 0;
}

// End of Main Loop
//--------------------------------------------------------------------------------------
void balance_white(cv::Mat mat) {
    double discard_ratio = 0.05;
    int hists[3][256];
    memset(hists, 0, 3*256*sizeof(int));
    
    for (int y = 0; y < mat.rows; ++y) {
        uchar* ptr = mat.ptr<uchar>(y);
        for (int x = 0; x < mat.cols; ++x) {
            for (int j = 0; j < 3; ++j) {
                hists[j][ptr[x * 3 + j]] += 1;
            }
        }
    }
    
    // cumulative hist
    int total = mat.cols*mat.rows;
    int vmin[3], vmax[3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 255; ++j) {
            hists[i][j + 1] += hists[i][j];
        }
        vmin[i] = 0;
        vmax[i] = 255;
        while (hists[i][vmin[i]] < discard_ratio * total)
        vmin[i] += 1;
        while (hists[i][vmax[i]] > (1 - discard_ratio) * total)
        vmax[i] -= 1;
        if (vmax[i] < 255 - 1)
        vmax[i] += 1;
    }
    
    
    for (int y = 0; y < mat.rows; ++y) {
        uchar* ptr = mat.ptr<uchar>(y);
        for (int x = 0; x < mat.cols; ++x) {
            for (int j = 0; j < 3; ++j) {
                int val = ptr[x * 3 + j];
                if (val < vmin[j])
                val = vmin[j];
                if (val > vmax[j])
                val = vmax[j];
                ptr[x * 3 + j] = static_cast<uchar>((val - vmin[j]) * 255.0 / (vmax[j] - vmin[j]));
            }
        }
    }
}

// EOF
