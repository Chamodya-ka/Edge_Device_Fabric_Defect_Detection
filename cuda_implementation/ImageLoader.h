#include <iostream>
#include <opencv/cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include "Image.h"

using namespace cv;
using namespace std;

/* 
Class used to read images
 */
class ImageLoader{

public:

    Image readImage(string filename);
    //Image readImage(int index);

private:

    Mat readImageFromFile(string filename);
    Mat toGrayScale(const Mat& image);
    Mat quantize(Mat& image, unsigned int graylevel = 8,unsigned int minlevel=0);
    
};