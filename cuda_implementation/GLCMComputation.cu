#include "GLCMComputation.h"
#include <assert.h>


GLOBAL void ComputeCoOccurenceMat(const int *pixels, int *d_out, const int N,const int rows, const int cols
            , int gl){
                //HERE HARDCORED SIZE OF GL * GL * 4 DUE TO CONSTANT INT REQUIREMENT
                __shared__ int subGLCM[8 * 8 * 4];
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
                    
                    if(idX + 1 < N && floorf((idX + 1)/cols)==floorf(idX/cols)){
                        //d = 0 - Compare and add Current index and Current Index  + 1  
                        //assert(localIdX < 0); 
                        atomicAdd( &subGLCM[pixels[idX] * gl + pixels[idX+1] ],1);
                        //assert(localIdX < 0); 
                         
                    }
                    if(idX - cols>=0){
                        //d = 90
                        //assert(localIdX < 0); 
                        atomicAdd( &subGLCM[(1 * gl * gl) + pixels[idX] * gl +  pixels[idX-cols]], 1);
                        //assert(localIdX < 0); 
                    }
                    //assert(localIdX < 0); 
                    if(idX - cols+1>=0){
                        //d = 45
                        //assert(localIdX < 0); 
                        atomicAdd( &subGLCM[(2 * gl* gl)  + pixels[idX] * gl] +  pixels[idX-cols+1], 1);
                    }
                    
                    if(idX - cols-1>=0){
                        //d = 135
                        atomicAdd( &subGLCM[(3 * gl* gl)  + pixels[idX] * gl] +  pixels[idX-cols-1], 1);
                        
                    }
                   // assert(localIdX < 0); 
                }
                
                __syncthreads();
                
                
                if (localIdX< gl * gl * 4){
                    
                    //assert(threadIdx.x < 2); 
                    //CHECK AGAIN
                    //assert(threadIdx.x + threadIdx.y * blockDim.x < 256); 
                    d_out[(blockIdx.x + blockIdx.y * gridDim.x) * gl * gl * 4 + localIdX] = subGLCM[localIdX];
                    //assert(localIdX < 0); 
                }
                
                

                 
            }


int* GLCMComputation :: GetSubGLCM(Image img,const int d, const int angle){
    int* h_out;
    int* d_pixels;
    int* d_out;
    std::vector<int> v = img.getPixels();
    int* host_pixels = &v[0];
    int rows = img.get_rows();
    int cols = img.get_cols();
    int gl = img.get_maxGL();
    int N = rows * cols;

    size_t bytes = rows * cols * sizeof(int);
    size_t intsize = sizeof(int);

    

    int THREADS = 32;
    //rows = cols because square shaped
    int BLOCKS = ( rows + THREADS -1 )/ THREADS;
    dim3 threadsPerBlock(THREADS,THREADS);
    dim3 blocksPerGrid(BLOCKS,BLOCKS);

    h_out = (int*)malloc(intsize * gl *gl *4 *BLOCKS *BLOCKS);
    cudaMalloc(&d_pixels,bytes);
    cudaMalloc(&d_out,BLOCKS*BLOCKS*gl*gl*intsize*4);
    cudaMemset(d_out, 0,BLOCKS*BLOCKS*gl*gl*intsize*4);

    cudaMemcpy(d_pixels,host_pixels,bytes,cudaMemcpyHostToDevice);
    
    //LAUCNCH KERNEL HERE
    ComputeCoOccurenceMat<<<blocksPerGrid,threadsPerBlock>>>(d_pixels,d_out,N,rows,cols,8);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, BLOCKS*BLOCKS*gl*gl*intsize*4, cudaMemcpyDeviceToHost );


    return h_out;
}

/* __global__ void getCorMat(int* pixels ,int gl, int rows, int cols,int d, int theta,int* cm){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx<rows*cols){
        if (theta == 0){
            
            int next_idx = idx+d;
            if (floorf(next_idx/cols)==floorf(idx/cols)){
                    atomicAdd(&cm[gl*pixels[idx] + pixels[next_idx]], 1);
            }         
        }
    }
} */


/* int getglcm(){
    
    int *d_pixels;
    int *d_cm;
    srand((unsigned)time(NULL));
    int rows = 1250;
    int cols = 1250;
    int total = rows * cols;  
    int gl = 8;
    int d = 1;
    int theta = 0;
    int *pix;
    int *result;

    size_t bytes = total* sizeof(int);
    size_t intsize = sizeof(int);

    pix = (int*)malloc(bytes);
    
    result = (int*)malloc(intsize * gl *gl);

    for (int i=0; i<gl*gl; i++) result[i] = 0;

    vector<int> pixels(total,0);

    for (int i =0; i < total; i++){
        //int b = rand() % gl + 1;
        int b = 7;
        //pixels[i]=b;
        //pixels.at(i) =b;
        pix[i] = b;
        //cout << pixels[i] << endl;
    } 


    vector<int> cm(gl*gl,0);

    cudaMalloc(&d_pixels,bytes);
    cudaMalloc(&d_cm,gl*gl*intsize);
    cudaMemset(d_cm, 0, gl*gl*intsize);

    cudaMemcpy(d_pixels,pix,bytes,cudaMemcpyHostToDevice);
   

    int threads,blocksize;
	threads = 256;
	
	blocksize = (int)ceil((float)total/threads);
    
    getCorMat<<<blocksize, threads>>>(d_pixels, gl, rows, cols, d ,theta , d_cm);

    cudaMemcpy(result, d_cm, intsize * gl * gl, cudaMemcpyDeviceToHost );

    for (int j =0 ; j < gl ; j++){
        for (int k =0 ; k < gl ; k++){
            cout<< result[j*gl + k] <<" "        ;
        }
        cout<<"\n";
    }
 
    cudaFree(d_pixels);
	cudaFree(d_cm);


    return 0;
} */