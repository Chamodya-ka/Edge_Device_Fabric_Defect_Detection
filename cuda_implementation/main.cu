#include "ImageLoader.h"
#include "GLCMComputation.h"
#include "FeatureComputation.h"
#include "Image.h"
#include <stdio.h>
#include <stdint.h>
#include <chrono>
using namespace std::chrono;

//nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
//#include <opencv2/opencv.hpp>
//using namespace cv;
/* 
    Main method  --
 */
using namespace std;

int main(int argc, char *argv[]){
    unsigned int  maxgl = 255;
    unsigned int  mingl = 0;
    unsigned int  desiredgl = 7;
    unsigned int subImgDim = 32;
    uint r = 2048;
    uint c = 2048 ;
    int N = r*c/(subImgDim*subImgDim);
    GLCMComputation glcm = GLCMComputation();
     if (argc>1){
        if (!strcmp(argv[1],"-n")){
            cout<<"Generating csv records"<<"\n";
            fstream fout;
            fout.open("../ref_data/ref_csv/ref_data.csv",ios::out);
            string fname = argv[2];
            vector<String> filenames;
            cv::glob(fname, filenames);
            Image img = ImageLoader::readImage(filenames[0],maxgl,mingl,desiredgl,r,c);
            float* features = (float*) malloc(N *5* sizeof(float));
            float* d_out;
            std::string segment;
            for (size_t i=0; i<filenames.size(); i++)
            {
                
                img = ImageLoader::readImage(filenames[i],maxgl,mingl,desiredgl,r,c);
                d_out = glcm.GetSubGLCM(img,1,1,subImgDim);
                FeatureComputation::getFeatures(d_out,8,r,c,subImgDim,features);
                for (int j  = 0 ; j< N * 5 ; j++){
                    if(j==0)
                        fout<< filenames[i]<<",";
                    if (j!=0 and j%5==0)
                        {fout << "\n";
                        fout<< filenames[i]<<",";}
                         
                    fout << *(features+j);
                    if (j%5!=4)
                        fout<<",";
                    //cout<<  *(features + j)<<" ";
                }
                fout << "\n";
                
            }
            free(d_out);
            free(features);
            cout<<argv[1]<<"\n";
            return 0 ;

        }
        
    }  

    cudaFree(0);
    auto start = high_resolution_clock::now();
    string fname = "../testimg/1.bmp";

    Image img = ImageLoader::readImage(fname,maxgl,mingl,desiredgl,2048,2048);
        auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
 
    cout << duration.count() <<" ms"<< endl;    

    //cout<<"From main r : "<< r << "\n";
    //cout<<"From main c : "<< c << "\n";
    //auto stop = high_resolution_clock::now();
    //auto duration = duration_cast<milliseconds>(stop - start);
 
    //cout << duration.count() << endl;
    //vector<int> pixels = img.getPixels();
    //vector<vector<uint8_t>> vector2d = img.get2dVector();

    //cout << "matrix size "+to_string(img.get_rows() * img.get_cols())+" vector size = "+to_string(pixels.size()) << endl;
 
    //Mat dst = Mat(r,c, CV_8UC1 ,&pixels,c * sizeof(uint8_t));
    
    //cout << "NEW MAT "+to_string(dst.rows * dst.cols) << endl;
    //cout << dst.total() << endl;
    //cout << dst.rows << endl;
    //cout << dst.cols << endl;
    //cout << dst.type() << endl;
    //cout << dst.depth() << endl;
    //cout << dst.channels() << endl;


    //used for testing
    
    /* uint zero= 0;uint one= 0;uint two= 0;uint three= 0;uint four= 0;uint five= 0;uint six= 0;uint seven= 0;uint eight = 0;
     
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
 
    
    float* d_out = glcm.GetSubGLCM(img,1,1,subImgDim); // host pointer.

  
  
    /*  for (int i = 0 ; i < 64*64*64*4  ; i ++){
        if (i%8==0)
            cout << "\n";
        if (i%256==0)
            cout << "\n";
        
        cout<<  *(d_out + i);
    }                            
 */


 
    //cout <<"GLCMs computed"<< "\n";
    float* features = (float*) malloc(N *5* sizeof(float));
    FeatureComputation::getFeatures(d_out,8,r,c,subImgDim,features);

    for (int i = 0 ; i < N*5  ; i ++){
        if (i%5==0 and i !=0)
            {cout  <<i ;
            cout  <<"\n" ;} 
        
        
        cout<<  *(features + i)<<" ";
    }                        
//end testing

/*
    Implement

    For training iterate over a images and proceed

    For usage get images using camera
 */
    





}
