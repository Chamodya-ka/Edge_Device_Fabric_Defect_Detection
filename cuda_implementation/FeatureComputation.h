#ifndef FEATURECOMPUTATION
#define FEATURECOMPUTATION
#define GLOBAL __global__ 
#define HOSTDEV __host__ __device__
#define DEV __device__
#include <iostream>

class FeatureComputation{
    public:
        static void getFeatures(float* subGLCMs,int gl, int rows, int cols,unsigned int subImgDim,float* features);


};


#endif