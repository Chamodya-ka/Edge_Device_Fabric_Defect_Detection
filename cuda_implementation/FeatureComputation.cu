#include "FeatureComputation.h"
#include "FeatureCalculation.cu"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


GLOBAL void computeFeatures(int* d_subGLCM , float* d_feat,int gl){
    
    int blockID = blockIdx.x;
    int idX = blockID * blockDim.x + threadIdx.x;
    int localIdX = threadIdx.x;

    __shared__ int subGLCMs[8 * 8 * 4];
    __shared__ float features[5];


    if (localIdX<gl*gl*4){
        subGLCMs[localIdX] = d_subGLCM[idX];    
    }

    //subGLCMs[localIdX] = d_subGLCM[localIdX];
    __syncthreads();
    
    if (localIdX< gl*gl*4){
        EnergyFeature(localIdX,gl,subGLCMs,features);
        ContrastFeature(localIdX,gl,subGLCMs,features);
        EntropyFeature(localIdX,gl,subGLCMs,features);
        HomogeneityFeature(localIdX,gl,subGLCMs,features);
        CorrelationFeature(localIdX,gl,subGLCMs,features);
    }
    __syncthreads();
    switch (localIdX)
    {
    case 0:
        d_feat[ 5 * blockID + localIdX] = features[localIdX];
        break;
    case 1:
        d_feat[ 5 * blockID + localIdX] = features[localIdX];
    break;
    case 2:
        d_feat[ 5 * blockID + localIdX] = features[localIdX];
    break;
    case 3:
        d_feat[ 5 * blockID + localIdX] = features[localIdX];
    break;
    case 4:
        d_feat[ 5 * blockID + localIdX] = features[localIdX];
    break;
    default:
        break;
    }

}


float* FeatureComputation::getFeatures(int* subGLCM,int gl){
    float* h_feat;
    float* d_feat;
    int* d_subGLCM;
    size_t intsize = sizeof(int);
    size_t floatsize = sizeof(float);
    int N = 64 * 64; // Number of blocks per image
    int bytes = 4 * gl * gl * N * intsize;

    int THREADS = gl * gl * 4;
    int BLOCKS = ( gl * gl * 4 * N + THREADS -1 )/ THREADS;
    //std::cout<<BLOCKS;
    dim3 threads(THREADS);
    dim3 blocks(BLOCKS);
    

    h_feat = (float*)malloc(floatsize * BLOCKS * 5);
    
    cudaMalloc(&d_feat, floatsize * 5  * BLOCKS);
    cudaMalloc(&d_subGLCM, intsize * gl * gl * 4  * BLOCKS);
    cudaMemset(d_feat,0,BLOCKS *  floatsize*5);
    cudaMemcpy(d_subGLCM,subGLCM,bytes,cudaMemcpyHostToDevice);
    computeFeatures<<<blocks,threads>>>(d_subGLCM,d_feat,gl);
    gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );
    cudaDeviceSynchronize();
    cudaMemcpy(h_feat, d_feat, BLOCKS*5*floatsize, cudaMemcpyDeviceToHost );
    cudaFree(d_feat);
    cudaFree(d_subGLCM);

     /*for(int h =0 ; h < BLOCKS;h++){
        
        std:: cout << h_feat[h*5 + 0] << " ";
        std:: cout << h_feat[h*5 + 1] << " ";
        std:: cout << h_feat[h*5 + 2] << " ";
        std:: cout << h_feat[h*5 + 3] << " ";
        std:: cout << h_feat[h*5 + 4] << " \n";
        std:: cout << h << "\n";
        //std:: cout << h << " ";
        
    }  */

    return h_feat;



}
