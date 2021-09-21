#include "GLCMComputation.h"
#include <assert.h>
#include "FeatureCalculation.cu"

/* DEV void EnergyFeature(int id, int gl, int* subGLCM, float* feature){
    __shared__ float a,b,c,d;
    switch (id)
    {
    case 0:
        a=0;
        break;
    case 1:
        b=0;
    break;
    case 2:
        c=0;
    break;
    case 3:
        d=0;
    break;
    
    default:
        break;
    }

    __syncthreads();
    
    if (id<gl*gl){
        atomicAdd(&a,pow(subGLCM[id],2));
    }
    else if(id<gl*gl*2){
        atomicAdd(&b,pow(subGLCM[id],2));
    }
    else if(id<gl*gl*3){
        atomicAdd(&c,pow(subGLCM[id],2));
    }
    else if(id<gl*gl*4){
        atomicAdd(&d,pow(subGLCM[id],2));
    }
    //printf("%d",a);
    __syncthreads();
    if (id<=0)
        feature[id] = (float)(a+b+c+d)/4;

} */

GLOBAL void ComputeCoOccurenceMat(const int *pixels, int *d_out, float *d_feat, const int N,const int rows, const int cols
            , int gl){
                float* feature; 
                int* subMat;
                //HERE HARDCORED SIZE OF GL * GL * 4 DUE TO CONSTANT INT REQUIREMENT
                __shared__ int subGLCM[8 * 8 * 4];
                __shared__ float featureVector[5];
                feature = featureVector;
                subMat = subGLCM;
                //int row = blockIdx.x * blockDim.x + threadIdx.x;
                //int col = blockIdx.y * blockDim.y + threadIdx.y;
                //int idX = col + row * cols;
                int blockID = blockIdx.x + blockIdx.y *gridDim.x;
                int idX = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
                int localIdX = threadIdx.x + threadIdx.y * blockDim.x;
                if (localIdX<gl*gl*4){
                    subGLCM[localIdX] = 0;
                    
                }
                
                __syncthreads();
                
                if (idX< N){ 
                    if(idX + 1 < N && floorf((idX + 1)/blockDim.x)==floorf(idX/blockDim.x)){
                        //d = 0 - Compare and add Current index and Current Index  + 1  
                        atomicAdd( &subGLCM[pixels[idX] * gl + pixels[idX+1] ],1);   
                    }
                    if(((int(idX)-int(blockDim.x))>=0) && (floorf((idX - blockDim.x)/(blockDim.x * blockDim.y))== floorf(idX /(blockDim.x * blockDim.y)))){
                        //d = 90
                        atomicAdd( &subGLCM[(1 * gl * gl) + pixels[idX] * gl +  pixels[idX-blockDim.x]], 1);
                    }
                    if (floorf((idX - blockDim.x+1)/(blockDim.x * blockDim.y) )== floorf(idX /(blockDim.x * blockDim.y))){
                        //d = 45
                        if (floorf((idX - blockDim.x+1)/blockDim.x)  < floorf(idX /blockDim.x))
                            atomicAdd( &subGLCM[(2 * gl * gl) + pixels[idX] * gl +  pixels[idX-blockDim.x+1]], 1);
                    }
                    if(floorf((idX - blockDim.x-1)/(blockDim.x * blockDim.y))== floorf(idX /(blockDim.x * blockDim.y))){
                        //d = 135
                        if (floorf((idX - blockDim.x-1)/blockDim.x) + 1 == floorf(idX /blockDim.x)){
                            atomicAdd( &subGLCM[(3 * gl* gl)  + pixels[idX] * gl] +  pixels[idX - blockDim.x-1], 1);
                        }                  
                    }
                }
                
                __syncthreads();
                
                
                if (localIdX< gl * gl * 4){
                    
                    //COMMENTED TO TEST CONTRAST
                     /* EnergyFeature(localIdX,gl,subMat,feature);
                    __syncthreads();
                    if (localIdX<=0){
                        //printf("%f",feature[localIdX]);
                        d_feat[blockID + 0] = (float)featureVector[localIdX];
                    }  */

                    ContrastFeature(localIdX,gl,subMat,feature);
                    __syncthreads();
                    if (localIdX==1){
                        //printf("%f",feature[localIdX]);
                        d_feat[blockID + 0] = (float)featureVector[localIdX];
                    }
                    //printf("%d\f",&feature); // LOOKS WRONG CHECK
                    d_out[(blockIdx.x + blockIdx.y * gridDim.x) * gl * gl * 4 + localIdX] = subGLCM[localIdX];
                    
                }
                
                

                 
            }


int* GLCMComputation :: GetSubGLCM(Image img,const int d, const int angle){
    int* h_out;
    float* h_feat;
    int* d_pixels;
    int* d_out;
    float* d_feat;
    std::vector<int> v = img.getPixels();
    int* host_pixels = &v[0];
    int rows = img.get_rows();
    int cols = img.get_cols();
    int gl = img.get_maxGL();
    int N = rows * cols;

    size_t bytes = rows * cols * sizeof(int);
    size_t intsize = sizeof(int);
    size_t floatsize = sizeof(float);
    

    int THREADS = 32;
    //rows = cols because square shaped
    int BLOCKS = ( rows + THREADS -1 )/ THREADS;
    cout<<"BLOCKS :";
    cout<< BLOCKS << endl;
    dim3 threadsPerBlock(THREADS,THREADS);
    dim3 blocksPerGrid(BLOCKS,BLOCKS);

    h_out = (int*)malloc(intsize * gl *gl *4 *BLOCKS *BLOCKS);
    h_feat = (float*)malloc(floatsize * BLOCKS * BLOCKS * 5);

    cudaMalloc(&d_pixels,bytes);
    cudaMalloc(&d_out,BLOCKS*BLOCKS*gl*gl*intsize*4);
    cudaMalloc(&d_feat,BLOCKS * BLOCKS * floatsize*5);
    cudaMemset(d_out, 0,BLOCKS*BLOCKS*gl*gl*intsize*4);
    cudaMemset(d_feat,0,BLOCKS * BLOCKS * floatsize*5);
    cudaMemcpy(d_pixels,host_pixels,bytes,cudaMemcpyHostToDevice);
    
    //LAUCNCH KERNEL HERE
    ComputeCoOccurenceMat<<<blocksPerGrid,threadsPerBlock>>>(d_pixels,d_out,d_feat,N,rows,cols,8);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, BLOCKS*BLOCKS*gl*gl*intsize*4, cudaMemcpyDeviceToHost );
    cudaMemcpy(h_feat, d_feat, BLOCKS*BLOCKS*floatsize*5, cudaMemcpyDeviceToHost );
    
    cudaFree(d_pixels);
    cudaFree(d_out);
    cudaFree(d_feat);
     for(int h =0 ; h < BLOCKS*BLOCKS*5;h++){
        cout<< ("%f",h_feat[h])<< " ";
    } 

    return h_out;
}

