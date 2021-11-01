#define GLOBAL __global__ 
#define HOSTDEV __host__ __device__
#define DEV __device__
#ifndef GLCMCOMPUTATION
#define GLCMCOMPUTATION
#include "Image.h"

class GLCMComputation {

public:
    GLCMComputation(){} 
    float* GetSubGLCM(Image img, const int d, const int angle,unsigned int subImgDim);
    


private:
    //TAKES PIXELS VECTOR TO GLOBAL MEM AND BROKEN TO 32x32 CHUNKS | SHARED MEM HAS A gl x gl VECTOR TO STORE GLCM LATER USED FOR FEATURE CALS
    //BROKEN DOWN TO 2 KERNALS FOR EASE AT FIRST IMPLEMENTATION

    //GLOBAL void ComputeCoOccurenceMat(const int *pixels,int *d_out, const int N, const int rows, const int cols
    //        , const int gl);

    //GLOBAL void ComputeFeatures(const unsigned int *pixels, const int rows, const int cols
     //       ,const int d, const int angle);

    

};
#endif