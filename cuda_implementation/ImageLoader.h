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

    static Image readImage(string filename, unsigned int inmaxlevel =255 , unsigned int inminlevel= 0, unsigned int maxgraylevel = 8);
    //Image readImage(int index);

private:

    static Mat readImageFromFile(string filename);
    //static Mat toGrayScale(const Mat& image);
    static Mat quantize(Mat& image, unsigned int maxgraylevel ,unsigned int inmaxgraylevel,unsigned int inmingraylevel);
    
};