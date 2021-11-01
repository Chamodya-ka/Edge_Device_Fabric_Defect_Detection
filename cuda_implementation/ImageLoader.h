#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <string>
#ifndef IMAGE_H
#define IMAGE_H
#include "Image.h"

using namespace cv;
using namespace std;

/* 
Class used to read images
 */
class ImageLoader{

public:

    static Image readImage(string filename, unsigned int inmaxlevel =255 , unsigned int inminlevel= 0, unsigned int maxgraylevel = 7,
                            unsigned int d_sizex = 2048, unsigned int d_sizey = 2048);
    //Image readImage(int index);

private:
    
    static Mat readImageFromFile(string filename,unsigned int d_sizex, unsigned int d_sizey);
    //static Mat toGrayScale(const Mat& image);
    static vector<int> quantize(Mat& image, unsigned int maxgraylevel ,unsigned int inmaxgraylevel,unsigned int inmingraylevel);
    static vector<int> quantize2(Mat& image, unsigned int maxgraylevel ,unsigned int inmaxgraylevel,unsigned int inmingraylevel);
    static Mat getQuantizedMat(Mat& image, unsigned int maxgraylevel ,unsigned int inmaxgraylevel,unsigned int inmingraylevel);
    
};
#endif
