#include <iostream>
#include <vector>
#include <stdio.h>
using namespace std;

__global__ void getCorMat(int* pixels ,int gl, int rows, int cols,int d, int theta,int* cm){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx<rows*cols){
        if (theta == 0){
            
            int next_idx = idx+d;
            if (floorf(next_idx/cols)==floorf(idx/cols)){
                    atomicAdd(&cm[gl*pixels[idx] + pixels[next_idx]], 1);
            }         
        }
    }
}


int main(){
    
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

    //auto start  = chrono::high_resolution_clock::now();
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
}
