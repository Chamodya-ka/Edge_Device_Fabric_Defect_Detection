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

/*
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

}*/


float* FeatureComputation::getFeatures(int* subGLCM,int gl){
    float* h_feat;
    float* d_feat;
    int* d_subGLCM;
    size_t intsize = sizeof(int);
    size_t floatsize = sizeof(float);
    int N = 64 * 64; // Number of blocks per image
    int bytes = 4 * gl * gl * N * intsize;

	// ADDED HERE
	
	

	float* h_feat_1;
	float* h_feat_2;
	float* h_feat_3;
	float* h_feat_4;
	float* h_feat_5;    
	float* d_feat_1;
	float* d_feat_2;
	float* d_feat_3;
	float* d_feat_4;
	float* d_feat_5;
	int* d_inp_1; 
  
	h_feat_1 = (float*) malloc(floatsize * N);
	h_feat_2 = (float*) malloc(floatsize * N);
	h_feat_3 = (float*) malloc(floatsize * N);
	h_feat_4 = (float*) malloc(floatsize * N);
	h_feat_5 = (float*) malloc(floatsize * N);
	
	cudaMalloc(&d_feat_1,floatsize * N);
 	cudaMalloc(&d_feat_2,floatsize * N);
	cudaMalloc(&d_feat_3,floatsize * N);
	cudaMalloc(&d_feat_4,floatsize * N);
	cudaMalloc(&d_feat_5,floatsize * N);
	
	cudaMalloc(&d_inp_1, intsize * 4 * gl * gl * N);
	//cudaMalloc(&d_inp_2, intsize * gl * gl * N);
	//cudaMalloc(&d_inp_3, intsize * gl * gl * N);
	//cudaMalloc(&d_inp_4, intsize * gl * gl * N);

			
		
	int byte = 4 * gl * gl * N * intsize;
	int *h_sGLCM;//,*h_sGLCM2,*h_sGLCM3,*h_sGLCM4;
	cudaMallocHost(&h_sGLCM,byte);
		
	for (int i = 0 ; i < N * gl *gl * 4 ; i++){
		h_sGLCM[i] = subGLCM[i];				
	}	

		
	int N_STREAMS = 5; // 5 features
	float *results[N_STREAMS];
	int *data[N_STREAMS];	
	
	cudaStream_t stream[N_STREAMS];
	for (int i = 0 ; i <N_STREAMS ; i++){
		cudaStreamCreate(&stream[i]);
				
	}


 	int THREADS = gl * gl * 4;
    int BLOCKS = ( gl * gl * 4 * N + THREADS -1 )/ THREADS;
	dim3 threads(THREADS);
    dim3 blocks(BLOCKS);
	cudaMemcpy(d_inp_1,h_sGLCM,N * gl * gl *4 * intsize,cudaMemcpyHostToDevice);
	EnergyFeature<<<blocks,threads,0,stream[0]>>>(gl , d_inp_1 ,d_feat_1);
	//cudaMemcpy(d_inp_2,h_sGLCM,N * gl * gl *4 * intsize,cudaMemcpyHostToDevice);
	ContrastFeature<<<blocks,threads,0,stream[1]>>>(gl , d_inp_1 ,d_feat_2);
	//cudaMemcpy(d_inp_3,h_sGLCM,N * gl * gl *4 * intsize,cudaMemcpyHostToDevice);
	EntropyFeature<<<blocks,threads,0,stream[2]>>>(gl , d_inp_1 ,d_feat_3);
	//cudaMemcpy(d_inp_4,h_sGLCM,N * gl * gl *4 * intsize,cudaMemcpyHostToDevice);
	HomogeneityFeature<<<blocks,threads,0,stream[3]>>>(gl , d_inp_1 ,d_feat_4);
	//cudaMemcpy(d_inp_5,h_sGLCM,N * gl * gl *4 * intsize,cudaMemcpyHostToDevice);
	CorrelationFeature<<<blocks,threads,0,stream[4]>>>(gl , d_inp_1 ,d_feat_5);

	gpuErrchk( cudaDeviceSynchronize() );
	cudaMemcpy(h_feat_1, d_feat_1, N * floatsize,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_feat_2, d_feat_2, N * floatsize,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_feat_3, d_feat_3, N * floatsize,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_feat_4, d_feat_4, N * floatsize,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_feat_5, d_feat_5, N * floatsize,cudaMemcpyDeviceToHost);
	cudaFree(d_feat_1);
	cudaFree(d_feat_2);
	cudaFree(d_feat_3);
	cudaFree(d_feat_4);
	cudaFree(d_feat_5);
	cudaFree(d_inp_1);
	
		
	for (int i = 0 ; i < N ;i++){
		std::cout<<h_feat_1[i]<<" ";
		std::cout<<h_feat_2[i]<<" ";
		std::cout<<h_feat_3[i]<<" ";
		std::cout<<h_feat_4[i]<<" ";
		std::cout<<h_feat_5[i]<<" \n";
	}
	
	// ADDED END HERE
/*

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
	*/
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
