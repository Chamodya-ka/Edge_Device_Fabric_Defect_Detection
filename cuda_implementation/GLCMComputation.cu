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

GLOBAL void ComputeCoOccurenceMat(const int *pixels, int *d_out, const int N,const int rows, const int cols
            , int gl,int sizeDout){
                //float* feature; 
                __shared__ int subGLCM[8 * 8 * 4];
                //__shared__ float featureVector[5];
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
                        atomicAdd( &subGLCM[pixels[idX] * gl + pixels[idX+1] ],(int)1);   
						//atomicAdd( &d_out[blockID * gl * gl * 4 + (int)pixels[idX] * gl + (int)pixels[idX+1]] , (int)1);
                    }
                    if(((int(idX)-int(blockDim.x))>=0) && (floorf((idX - blockDim.x)/(blockDim.x * blockDim.y))== floorf(idX /(blockDim.x * blockDim.y)))){
                        //d = 90
                        atomicAdd( &subGLCM[(1 * gl * gl) + pixels[idX] * gl +  pixels[idX-blockDim.x]], (int)1);
						//atomicAdd( &d_out[blockID * gl * gl * 4 + gl*gl*1 +(int)pixels[idX] * gl + (int)pixels[idX+1]] , (int)1);
                    }
                    if (floorf((idX - blockDim.x+1)/(blockDim.x * blockDim.y) )== floorf(idX /(blockDim.x * blockDim.y))){
                        //d = 45
                        if (floorf((idX - blockDim.x+1)/blockDim.x)  < floorf(idX /blockDim.x))
                            atomicAdd( &subGLCM[(2 * gl * gl) + pixels[idX] * gl +  pixels[idX-blockDim.x+1]],(int) 1);
							//atomicAdd( &d_out[blockID * gl * gl * 4 + gl*gl*2 +(int)pixels[idX] * gl + (int)pixels[idX+1]] ,(int) 1);
                    }
                    if(floorf((idX - blockDim.x-1)/(blockDim.x * blockDim.y))== floorf(idX /(blockDim.x * blockDim.y))){
                        //d = 135
                        if (floorf((idX - blockDim.x-1)/blockDim.x) + 1 == floorf(idX /blockDim.x)){
                            atomicAdd( &subGLCM[(3 * gl* gl)  + pixels[idX] * gl] +  pixels[idX - blockDim.x-1],(int) 1);
							//atomicAdd( &d_out[blockID * gl * gl * 4 + gl*gl*3 +(int)pixels[idX] * gl + (int)pixels[idX+1]] , (int)1);
                        }                  
                    }
                }
                
                 __syncthreads();
                 
                
                if (localIdX< gl * gl * 4){
                    if (blockID * gl * gl * 4 + localIdX < sizeDout)
                    	d_out[blockID * gl * gl * 4 + localIdX] = subGLCM[localIdX];
                    
                }            
                 
            }


int* GLCMComputation :: GetSubGLCM(Image img,const int d, const int angle){
    int* h_out;
    //float* h_feat;
    int* d_pixels;
    int* d_out;
    //float* d_feat;
    std::vector<int> v = img.getPixels();
    int* host_pixels = &v[0];
    int rows = img.get_rows();
    int cols = img.get_cols();
    int gl = img.get_maxGL() + 1;
    int N = rows * cols;

    size_t bytes = rows * cols * sizeof(int);
    size_t intsize = sizeof(int);
    //size_t floatsize = sizeof(float);
    

    int THREADS = 32;
    //rows = cols because square shaped
    int BLOCKS = ( rows + THREADS -1 )/ THREADS;

	int sizeDout = BLOCKS*BLOCKS*gl*gl*intsize*4;

    cout<<"BLOCKS :";
    cout<< BLOCKS << endl;
    dim3 threadsPerBlock(THREADS,THREADS,1);
    dim3 blocksPerGrid(BLOCKS,BLOCKS,1);

    h_out = (int*)malloc(sizeDout);
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
    
    cudaMemcpy(h_out, d_out, BLOCKS*BLOCKS*gl*gl*intsize*4, cudaMemcpyDeviceToHost );
    //cudaMemcpy(h_feat, d_feat, BLOCKS*BLOCKS*floatsize*5, cudaMemcpyDeviceToHost );
    
    cudaFree(d_pixels);
    cudaFree(d_out);
    //cudaFree(d_feat);


    return h_out;
}

