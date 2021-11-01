#include "FeatureComputation.h"
#include "FeatureCalculation.cu"
#include "New_FeatureCalculation.cu"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



float* FeatureComputation::getFeatures(int* subGLCM,int gl,int rows, int cols,unsigned int subImgDim){
    float* h_feat;
    //float* d_feat;
    //int* d_subGLCM;
    size_t intsize = sizeof(int);
    size_t floatsize = sizeof(float);
	int blocksX = rows/subImgDim;
	int blocksY = cols/subImgDim;
	int N = blocksX * blocksY;
	std::cout << "N " << N;
	std::cout << " blocksX " << blocksX;
	std::cout << " blocksY " << blocksY << "\n";
    //int N = 64 * 64; // Number of blocks per image
    //int bytes = 4 * gl * gl * N * intsize;
	int num_intermediates = 4 * N;
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
	int* d_inp_2; 
	int* d_inp_3; 
	int* d_inp_4; 
	int* d_inp_5; 
	//float* stddX_h;
	//float* stddY_h;
	float* stddX_d;
	float* stddY_d;
	//int* meanX_h;
	//int* meanY_h;
	int* meanX_d;
	int* meanY_d;
  
	h_feat_1 = (float*) malloc(floatsize * N);
	h_feat_2 = (float*) malloc(floatsize * N);
	h_feat_3 = (float*) malloc(floatsize * N);
	h_feat_4 = (float*) malloc(floatsize * N);
	h_feat_5 = (float*) malloc(floatsize * N);
	std::cout << " Here1 "<<"\n";	
	//stddX_h = (float*) malloc(floatsize * num_intermediates);
	//stddY_h = (float*) malloc(floatsize * num_intermediates);
	//meanX_h = (int*) malloc(intsize * num_intermediates);
	//meanY_h = (int*) malloc(intsize * num_intermediates);
	
	cudaMalloc(&d_feat_1,floatsize * N);
 	cudaMalloc(&d_feat_2,floatsize * N);
	cudaMalloc(&d_feat_3,floatsize * N);
	cudaMalloc(&d_feat_4,floatsize * N);
	cudaMalloc(&d_feat_5,floatsize * N);
	std::cout << " Here2 "<<"\n";	
	cudaMalloc(&d_inp_1, intsize * 4 * gl * gl * N);
	cudaMalloc(&d_inp_2, intsize * 4  * gl * gl * N);
	cudaMalloc(&d_inp_3, intsize * 4  * gl * gl * N);
	cudaMalloc(&d_inp_4, intsize * 4  * gl * gl * N);
	cudaMalloc(&d_inp_5, intsize * 4  * gl * gl * N);

	cudaMalloc(&stddX_d, floatsize * num_intermediates);	
	cudaMalloc(&stddY_d, floatsize * num_intermediates);	
	cudaMalloc(&meanX_d, intsize * num_intermediates);		
	cudaMalloc(&meanY_d, intsize * num_intermediates);		 						
	std::cout << " Here3 "<<"\n";
	int byte = 4 * gl * gl * N * intsize;
	float *h_sGLCM;//,*h_sGLCM2,*h_sGLCM3,*h_sGLCM4;
	float *h_sGLCM_1;
	float *h_sGLCM_2;
	float *h_sGLCM_3;
 	float *h_sGLCM_4;
	cudaMallocHost(&h_sGLCM,byte);
	cudaMallocHost(&h_sGLCM_1,byte);
	cudaMallocHost(&h_sGLCM_2,byte);
	cudaMallocHost(&h_sGLCM_3,byte);
	cudaMallocHost(&h_sGLCM_4,byte);
	//cudaMallocHost(&)
	std::cout << " Here4 "<<"\n";
	float* host_data1= (float*) malloc(byte);
	float* host_data2= (float*) malloc(byte);
	float* host_data3= (float*) malloc(byte);	
	float* host_data4= (float*) malloc(byte);	
	float* host_data5= (float*) malloc(byte);	
	std::cout << " Here5 "<<"\n";
	memcpy(host_data1,subGLCM,byte);
	memcpy(host_data2,subGLCM,byte);
	memcpy(host_data2,subGLCM,byte);
	memcpy(host_data4,subGLCM,byte);
	memcpy(host_data5,subGLCM,byte);
	h_sGLCM = host_data1;
	h_sGLCM_1 = host_data2;
	h_sGLCM_2 = host_data3;
	h_sGLCM_3 = host_data4;
	h_sGLCM_4 = host_data5;  
	
	/*  for (int i = 0 ; i < N * gl *gl * 4 ; i++){
		h_sGLCM[i] = subGLCM[i];
		h_sGLCM_1[i] = subGLCM[i];
		h_sGLCM_2[i] = subGLCM[i];
		h_sGLCM_3[i] = subGLCM[i];
		h_sGLCM_4[i] = subGLCM[i];				
	}  */	

		
	int N_STREAMS = 5; // 5 features
	//float *results[N_STREAMS];
	//int *data[N_STREAMS];	
	
	cudaStream_t stream[N_STREAMS];
	for (int i = 0 ; i <N_STREAMS ; i++){
		cudaStreamCreate(&stream[i]);
				
	}



 	int THREADS = gl * gl * 2; // HALF THREADS
    int BLOCKS =  N  ;//( gl * gl * 4 * N + THREADS -1 )/ THREADS;

	dim3 intermediateTHREADS(THREADS/2);
	dim3 threads(THREADS);
    dim3 blocks(BLOCKS);

	cudaMemsetAsync(stddX_d,0,floatsize *num_intermediates,stream[4]);
	cudaMemsetAsync(stddY_d,0,floatsize *num_intermediates,stream[4]);
	cudaMemsetAsync(meanX_d,0,intsize *num_intermediates,stream[4]);
	cudaMemsetAsync(meanY_d,0,intsize *num_intermediates,stream[4]);

	cudaMemcpyAsync(d_inp_1,h_sGLCM,N * gl * gl *4 * intsize,cudaMemcpyHostToDevice,stream[0]);
	cudaMemcpyAsync(d_inp_2,h_sGLCM_1,N * gl * gl *4 * intsize,cudaMemcpyHostToDevice,stream[1]);
	cudaMemcpyAsync(d_inp_3,h_sGLCM_2,N * gl * gl *4 * intsize,cudaMemcpyHostToDevice,stream[2]);
	cudaMemcpyAsync(d_inp_4,h_sGLCM_3,N * gl * gl *4 * intsize,cudaMemcpyHostToDevice,stream[3]);
	cudaMemcpyAsync(d_inp_5,h_sGLCM_4,N * gl * gl * 4 * intsize,cudaMemcpyHostToDevice,stream[4]);
	
	
	CalculateMeanX<<<blocks,intermediateTHREADS,0,stream[4]>>>(gl, d_inp_5,meanX_d);
	CalculateMeanY<<<blocks,intermediateTHREADS,0,stream[4]>>>(gl, d_inp_5,meanY_d);
	CalculateStddX<<<blocks,intermediateTHREADS,0,stream[4]>>>(gl, d_inp_5,meanX_d,stddX_d);
	CalculateStddY<<<blocks,intermediateTHREADS,0,stream[4]>>>(gl, d_inp_5,meanY_d,stddY_d);
	
	EnergyFeature2<<<blocks,threads,0,stream[0]>>>(gl , d_inp_1 ,d_feat_1);
	ContrastFeature2<<<blocks,threads,0,stream[1]>>>(gl , d_inp_2 ,d_feat_2);
	EntropyFeature2<<<blocks,threads,0,stream[2]>>>(gl , d_inp_3 ,d_feat_3);
	HomogeneityFeature2<<<blocks,threads,0,stream[3]>>>(gl , d_inp_4 ,d_feat_4);
	CorrelationFeature2<<<blocks,threads,0,stream[4]>>>(gl , d_inp_5 ,d_feat_5,meanX_d,meanY_d,stddX_d,stddY_d);

	gpuErrchk( cudaDeviceSynchronize() );
	cudaMemcpyAsync(h_feat_1, d_feat_1, N * floatsize,cudaMemcpyDeviceToHost,stream[0]);
	cudaMemcpyAsync(h_feat_2, d_feat_2, N * floatsize,cudaMemcpyDeviceToHost,stream[1]);
	cudaMemcpyAsync(h_feat_3, d_feat_3, N * floatsize,cudaMemcpyDeviceToHost,stream[2]);
	cudaMemcpyAsync(h_feat_4, d_feat_4, N * floatsize,cudaMemcpyDeviceToHost,stream[3]);
	cudaMemcpyAsync(h_feat_5, d_feat_5, N * floatsize,cudaMemcpyDeviceToHost,stream[4]);
	cudaFree(d_feat_1);
	cudaFree(d_feat_2);
	cudaFree(d_feat_3);
	cudaFree(d_feat_4);
	cudaFree(d_feat_5);
	cudaFree(d_inp_1);
	cudaFree(d_inp_2);
	cudaFree(d_inp_3);
	cudaFree(d_inp_4);
	cudaFree(d_inp_5);
	cudaFree(meanY_d);
	cudaFree(meanX_d);
	cudaFree(stddX_d);
	cudaFree(stddY_d);
	
	//free(h_sGLCM);
	//free(h_sGLCM_1);
	///free(h_sGLCM_2);
	//free(h_sGLCM_3);
	//free(h_sGLCM_4);

	
	std::cout << h_feat_1[1493]<<"\n";	
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
