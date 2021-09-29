#include "ImageLoader.h"
#include "GLCMComputation.h"
#include "FeatureComputation.h"
#include "Image.h"
#include <stdio.h>
#include <stdint.h>
#include <chrono>
using namespace std::chrono;


//#include <opencv2/opencv.hpp>
//using namespace cv;
/* 
    Main method  --
 */
using namespace std;

int main(){
    cudaFree(0);
    auto start = high_resolution_clock::now();
    string fname = "../testimg/gradient.png";
    unsigned int  maxgl = 255;
    unsigned int  mingl = 0;
    unsigned int  desiredgl = 7;
    Image img = ImageLoader::readImage(fname,maxgl,mingl,desiredgl);

    /*vector<int> pixels = img.getPixels();
    //vector<vector<uint8_t>> vector2d = img.get2dVector();

    cout << "matrix size "+to_string(img.get_rows() * img.get_cols())+" vector size = "+to_string(pixels.size()) << endl;
    uint r = img.get_rows();
    uint c = img.get_cols(); 
    Mat dst = Mat(r,c, CV_8UC1 ,&pixels,c * sizeof(uint8_t));
    
    cout << "NEW MAT "+to_string(dst.rows * dst.cols) << endl;
    cout << dst.total() << endl;
    cout << dst.rows << endl;
    cout << dst.cols << endl;
    cout << dst.type() << endl;
    cout << dst.depth() << endl;
    cout << dst.channels() << endl;


    //used for testing
    
    uint zero= 0;uint one= 0;uint two= 0;uint three= 0;uint four= 0;uint five= 0;uint six= 0;uint seven= 0;uint eight = 0;
     
    for(int i = 0; i < pixels.size(); i++) {
        
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
        else
            cout<<"OTHERSS!!"<<endl;
    } 
    std::cout << "0 : " + to_string(zero) << endl;
    std::cout << "1 : " + to_string(one) << endl;
    std::cout << "2 : " + to_string(two) << endl;
    std::cout << "3 : " + to_string(three) << endl;
    std::cout << "4 : " + to_string(four) << endl;
    std::cout << "5 : " + to_string(five) << endl;
    std::cout << "6 : " + to_string(six) << endl;
    std::cout << "7 : " + to_string(seven) << endl;  
    std::cout << "8 : " + to_string(eight) << endl; 
    std::cout << "Size of pixels vector : " + to_string(pixels.size()) << endl; 
    */
  
    GLCMComputation glcm = GLCMComputation();
    int* out = glcm.GetSubGLCM(img,1,1);

  

/*
     int j = 2 << 10;
    cout << j << endl;
    for (int i = 0 ; i < 64 * 4 ; i ++){
        if (i%8==0)
            cout << "\n";
        cout<<  *(out + 3169 * 64 * 4 +i);
    }  

*/
 
    cout << "\n";

    float* h = FeatureComputation::getFeatures(out,8);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
 
    cout << duration.count() << endl;

//end testing

/*
    Implement

    For training iterate over a images and proceed

    For usage get images using camera
 */
    





}
