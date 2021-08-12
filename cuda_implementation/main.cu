#include <stdio.h>
#include "ImageLoader.h"

//#include <opencv2/opencv.hpp>
//using namespace cv;
/* 
    Main method  --
 */
using namespace std;

int main(){
    string fname = "../testimg/noise.jpg";
    unsigned int  maxgl = 255;
    unsigned int  mingl = 0;
    unsigned int  desiredgl = 8;
    
    Image img = ImageLoader::readImage(fname,maxgl,mingl,desiredgl);
    vector<uint> pixels = img.getPixels();
    uint zero,one,two,three,four,five,six,seven,eight = 0;

    

    for(std::size_t i = 0; i < pixels.size(); ++i) {
        
        if (pixels[i]==0)
            zero+=1;
        else if (pixels[i]==1)
            one+=1;
        else if (pixels[i]==2)
            two+=1;
        else if (pixels[i]==3)
            three+=1;
        else if (pixels[i]==4)
            four+=1;
        else if (pixels[i]==5)
            five+=1;
        else if (pixels[i]==6)
            six+=1;
        else if (pixels[i]==7)
            seven+=1;
        else if (pixels[i]==8)
            eight+=1;

    } 
    std::cout << zero << "\n";
    std::cout << one << "\n";
    std::cout << two << "\n";
    std::cout << three << "\n";
    std::cout << four << "\n";
    std::cout << five << "\n";
    std::cout << six << "\n";
    std::cout << seven << "\n";
    std::cout << eight << "\n";
/* 
    Implement

    For training iterate over a images and proceed

    For usage get images using camera
 */



}
