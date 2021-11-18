#include "GLCMComputation.h"
#include <assert.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

 __device__ void warpReduceGLCM(volatile float* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32]; 
    sdata[tid] += sdata[tid + 16]; 
    sdata[tid] += sdata[tid +  8];
    sdata[tid] += sdata[tid +  4]; 
    sdata[tid] += sdata[tid +  2]; 
    sdata[tid] += sdata[tid +  1];  
} 
__device__ void normalizeGLCM(volatile int* histogramSubGLCM,volatile float* subGLCM,unsigned  int id,int gl){
    __shared__ float tempGLCM[8*8*4];
    /*   for (unsigned int i =0 ;i < 4; i++){
        tempGLCM[id+gl*gl*i] = histogramSubGLCM[id+gl*gl*i];
    }  */ 
    
    subGLCM[id] = histogramSubGLCM[id];
    

     tempGLCM[id] = histogramSubGLCM[id];
    __syncthreads();
    /* for (unsigned int s=gl*gl/2; s>32; s>>=1) {
        if (id < s) {
            tempGLCM[id]         += tempGLCM[id + s];
            tempGLCM[id+gl*gl*1] += tempGLCM[id+gl*gl*1 + s];
            tempGLCM[id+gl*gl*2] += tempGLCM[id+gl*gl*2 + s];
            tempGLCM[id+gl*gl*3] += tempGLCM[id+gl*gl*3 + s];
        }
        __syncthreads();
    } */
    if (id < 32){
        warpReduceGLCM(tempGLCM, id);
    }
    
    else if (64<=id <96){
        warpReduceGLCM(tempGLCM, id);
    }else if(128<=id<160){
        warpReduceGLCM(tempGLCM, id);
    }else if(192<=id<224){
        warpReduceGLCM(tempGLCM, id);
    } 
    __syncthreads();
    subGLCM[id] = __fdividef(histogramSubGLCM[id],tempGLCM[gl*gl*(id/(gl*gl))]);
    
    /* subGLCM[id + gl * gl * 0] = __fdividef(subGLCM[id + gl * gl * 0],tempGLCM[gl*gl*0]);

    subGLCM[id + gl * gl * 1] = 6.0;//tempGLCM[gl*gl*1];//__fdividef(subGLCM[id + gl * gl * 1],tempGLCM[gl*gl*1]);

    subGLCM[id + gl * gl * 2] = tempGLCM[gl*gl*2];//__fdividef(subGLCM[id + gl * gl * 2],tempGLCM[gl*gl*2]);

    subGLCM[id + gl * gl * 3] = tempGLCM[gl*gl*3];//__fdividef(subGLCM[id + gl * gl * 3],tempGLCM[gl*gl*3]);
 */

}

GLOBAL void ComputeCoOccurenceMat(const int *pixels, float *d_out, const int N,const int rows, const int cols
            , int gl,int sizeDout){
                //float* feature; 
                __shared__ float subGLCM[8 * 8 * 4];
                __shared__ int subHistogrambGLCM[8 * 8 * 4];
                //__shared__ float featureVector[5];
                int blockID = blockIdx.x + blockIdx.y *gridDim.x;
                int idX = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
                int localIdX = threadIdx.x + threadIdx.y * blockDim.x;
                if (localIdX<gl*gl*4){
                    subHistogrambGLCM[localIdX] = 0;
                }
                
                __syncthreads();
                
                if (idX< N){ 
                    if(idX + 1 < N && floorf((idX + 1)/blockDim.x)==floorf(idX/blockDim.x)){
                        //d = 0 - Compare and add Current index and Current Index  + 1  
                        atomicAdd( &subHistogrambGLCM[pixels[idX] * gl + pixels[idX+1] ],1);   
						//atomicAdd( &d_out[blockID * gl * gl * 4 + (int)pixels[idX] * gl + (int)pixels[idX+1]] , (int)1);
                    }
                    if(((int(idX)-int(blockDim.x))>=0) && (floorf((idX - blockDim.x)/(blockDim.x * blockDim.y))== floorf(idX /(blockDim.x * blockDim.y)))){
                        //d = 90
                        atomicAdd( &subHistogrambGLCM[(1 * gl * gl) + pixels[idX] * gl +  pixels[idX-blockDim.x]], 1);
						//atomicAdd( &d_out[blockID * gl * gl * 4 + gl*gl*1 +(int)pixels[idX] * gl + (int)pixels[idX+1]] , (int)1);
                    }
                    if (floorf((idX - blockDim.x+1)/(blockDim.x * blockDim.y) )== floorf(idX /(blockDim.x * blockDim.y))){
                        //d = 45
                        if (floorf((idX - blockDim.x+1)/blockDim.x)  < floorf(idX /blockDim.x))
                            atomicAdd( &subHistogrambGLCM[(2 * gl * gl) + pixels[idX] * gl +  pixels[idX-blockDim.x+1]], 1);
							//atomicAdd( &d_out[blockID * gl * gl * 4 + gl*gl*2 +(int)pixels[idX] * gl + (int)pixels[idX+1]] ,(int) 1);
                    }
                    if(floorf((idX - blockDim.x-1)/(blockDim.x * blockDim.y))== floorf(idX /(blockDim.x * blockDim.y))){
                        //d = 135
                        if (floorf((idX - blockDim.x-1)/blockDim.x) + 1 == floorf(idX /blockDim.x)){
                            atomicAdd( &subHistogrambGLCM[(3 * gl* gl)  + pixels[idX] * gl] +  pixels[idX - blockDim.x-1], 1);
							//atomicAdd( &d_out[blockID * gl * gl * 4 + gl*gl*3 +(int)pixels[idX] * gl + (int)pixels[idX+1]] , (int)1);
                        }                  
                    }
                }
                
                 __syncthreads();
                //used to normalize the GLCM
                /* if(localIdX < gl*gl){
                    normalizeGLCM(subGLCM,localIdX,gl);
                }  */
                 if (localIdX<gl*gl*4){
                    normalizeGLCM(subHistogrambGLCM,subGLCM,localIdX,gl);
                }  
                __syncthreads();
                if (localIdX< gl * gl * 4){
                    if (blockID * gl * gl * 4 + localIdX < sizeDout)
                    	d_out[blockID * gl * gl * 4 + localIdX] = subGLCM[localIdX];
                    
                }            
                 
            }


float* GLCMComputation :: GetSubGLCM(Image img,const int d, const int angle,unsigned int subImgDim){
    float* h_out;
    //float* h_feat;
    int* d_pixels;
    //int* d_out;
    float* d_out;
    //float* d_feat;
    std::vector<int> v = img.getPixels();
    int* host_pixels = &v[0];
    int rows = img.get_rows();
    int cols = img.get_cols();
    int gl = img.get_maxGL() + 1;
    int N = rows * cols;

    size_t bytes = rows * cols * sizeof(int);
    size_t intsize = sizeof(int);
    size_t floatsize = sizeof(float);
   //size_t floatsize = sizeof(float);
    

    int THREADS = subImgDim;
    //rows = cols because square shaped
    int BLOCKSx = ( rows + THREADS -1 )/ THREADS;
    int BLOCKSy = ( cols + THREADS -1 )/ THREADS;

	//int sizeDout = BLOCKSx*BLOCKSy*gl*gl*intsize*4;
    int sizeDout = BLOCKSx*BLOCKSy*gl*gl*floatsize*4;
    std::cout<<"BLOCKSx :";
    std::cout<< BLOCKSx << endl;
    std::cout<<"BLOCKSy :";
    std::cout<< BLOCKSy << endl;
    dim3 threadsPerBlock(THREADS,THREADS,1);
    dim3 blocksPerGrid(BLOCKSx,BLOCKSy,1);

    h_out = (float*)malloc(sizeDout);
    std::cout<<"TESTING!@";
    //h_feat = (float*)malloc(floatsize * BLOCKS * BLOCKS * 5);

    cudaMalloc(&d_pixels,bytes);
    cudaMalloc(&d_out,sizeDout);
    cudaMemset(d_out, 0,sizeDout);
    //cudaMalloc(&d_feat,BLOCKS * BLOCKS * floatsize*5);
    
    //cudaMemset(d_feat,0,BLOCKS * BLOCKS * floatsize*5);
    cudaMemcpy(d_pixels,host_pixels,bytes,cudaMemcpyHostToDevice);
    
    //LAUCNCH KERNEL HERE
    ComputeCoOccurenceMat<<<blocksPerGrid,threadsPerBlock>>>(d_pixels,d_out,N,rows,cols,8,sizeDout);
    gpuErrchk( cudaDeviceSynchronize() );
    
    cudaMemcpy(h_out, d_out, BLOCKSx*BLOCKSy*gl*gl*floatsize*4, cudaMemcpyDeviceToHost );

    //cudaMemcpy(h_feat, d_feat, BLOCKS*BLOCKS*floatsize*5, cudaMemcpyDeviceToHost );
    
    cudaFree(d_pixels);
    cudaFree(d_out);
    //cudaFree(d_feat);


    return h_out;
}

