#ifndef NFEATURECALCULATION
#define NFEATURECALCULATION
#define GLOBAL __global__ 
#define HOSTDEV __host__ __device__
#define DEV __device__



// USES HALF THE NUMBER OF THREADS PER BLOCK ie. 8 x 8 x 4 threads per block.
__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32]; 
    sdata[tid] += sdata[tid + 16]; 
    sdata[tid] += sdata[tid +  8]; 
    sdata[tid] += sdata[tid +  4]; 
    sdata[tid] += sdata[tid +  2]; 
    sdata[tid] += sdata[tid +  1]; 
}
__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32]; 
    sdata[tid] += sdata[tid + 16]; 
    sdata[tid] += sdata[tid +  8];
    sdata[tid] += sdata[tid +  4]; 
    sdata[tid] += sdata[tid +  2]; 
    sdata[tid] += sdata[tid +  1]; 
}


GLOBAL void EnergyFeature2(int gl, float* subGLCM, float* feature){
    //unsigned int blockID = blockIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int id = threadIdx.x;

    __shared__ float engSubGLCM[8 * 8 * 2];

    //First level reduction done while loading data
    engSubGLCM[id] = __powf(subGLCM[i],2) + __powf(subGLCM[i+blockDim.x] , 2);
     __syncthreads();

    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (id < s) {
            engSubGLCM[id] += engSubGLCM[id + s];
        }
        __syncthreads();
    }
    if (id < 32) warpReduce(engSubGLCM, id);

    if(id == 1)
        feature[blockIdx.x] = engSubGLCM[0]/4;

}

GLOBAL void ContrastFeature2(int gl, float* subGLCM, float* feature){
    //unsigned int blockID = blockIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int id = threadIdx.x;

    __shared__ float conSubGLCM[8 * 8 * 2];

    //First level reduction done while loading data
    conSubGLCM[id] = subGLCM[i] * __powf( abs(id%gl - floorf(id/gl) + gl * floorf(id/(gl*gl))) ,2) + 
                                subGLCM[i+blockDim.x] * __powf( abs((id+blockDim.x)%gl - floorf((id+blockDim.x)/gl) + gl * floorf((id+blockDim.x)/(gl*gl))) ,2);
     __syncthreads();

    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (id < s) {
            conSubGLCM[id] += conSubGLCM[id + s];
        }
        __syncthreads();
    }
    if (id < 32) warpReduce(conSubGLCM, id);

    if(id == 1)
        feature[blockIdx.x] = conSubGLCM[0]/4;

}

GLOBAL void EntropyFeature2(int gl, float* subGLCM, float* feature){
    //unsigned int blockID = blockIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int id = threadIdx.x;

    __shared__ float conSubGLCM[8 * 8 * 2];

    //First level reduction done while loading data
    conSubGLCM[id] = subGLCM[i] *  __log10f(subGLCM[i]+1) + 
                                subGLCM[i+blockDim.x] *  __log10f(subGLCM[i+blockDim.x]+1);
     __syncthreads();

    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (id < s) {
            conSubGLCM[id] += conSubGLCM[id + s];
        }
        __syncthreads();
    }
    if (id < 32) warpReduce(conSubGLCM, id);

    if(id == 1)
        feature[blockIdx.x] = conSubGLCM[0]/4;

}

GLOBAL void HomogeneityFeature2(int gl, float* subGLCM, float* feature){
    //unsigned int blockID = blockIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int id = threadIdx.x;

    __shared__ float homSubGLCM[8 * 8 * 2];

    //First level reduction done while loading data
    homSubGLCM[id] = subGLCM[i] / ( 1 +  __powf( abs(id%gl - floorf(id/gl) + gl * floorf(id/(gl*gl))) ,2) ) + 
                                subGLCM[i+blockDim.x] / ( 1 +  __powf( abs((id+blockDim.x)%gl - floorf(id+blockDim.x/gl) + gl * floorf(id+blockDim.x/(gl*gl))) ,2) );
     __syncthreads();

    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (id < s) {
            homSubGLCM[id] += homSubGLCM[id + s];
        }
        __syncthreads();
    }
    if (id < 32) warpReduce(homSubGLCM, id);
    __syncthreads();
    if(id == 1)
        feature[blockIdx.x] = homSubGLCM[0]/4;

}


GLOBAL void CalculateMeanX(int gl,float* SubGLCM,float* meanX){
    //SubGLCM in shared memory,
    //Using 64 threads.
    unsigned int i = blockIdx.x*(blockDim.x*4) + threadIdx.x;
    unsigned int id = threadIdx.x;

    __shared__ int localGLCM1[8 * 8];
    __shared__ int localGLCM2[8 * 8];
    __shared__ int localGLCM3[8 * 8];
    __shared__ int localGLCM4[8 * 8];

    localGLCM1[id] = SubGLCM[i] * ((id%gl) + 1);
    localGLCM2[id] = SubGLCM[i + blockDim.x * 1] * ((id%gl) + 1);
    localGLCM3[id] = SubGLCM[i + blockDim.x * 2] * ((id%gl) + 1);
    localGLCM4[id] = SubGLCM[i + blockDim.x * 3] * ((id%gl) + 1);


    __syncthreads();


    if (id < 32) {
        warpReduce(localGLCM1, id); 
        warpReduce(localGLCM2, id); 
        warpReduce(localGLCM3, id); 
        warpReduce(localGLCM4, id);  
    }  
    __syncthreads();
    if (id==0){
        meanX[blockIdx.x * 4 + 0] = localGLCM1[0];
        meanX[blockIdx.x * 4 + 1] = localGLCM2[0];
        meanX[blockIdx.x * 4 + 2] = localGLCM3[0];
        meanX[blockIdx.x * 4 + 3] = localGLCM4[0];
        }

}
GLOBAL void CalculateMeanY(int gl,float* SubGLCM,float* meanY){
    //SubGLCM in shared memory,
    //Using 64 threads.
    unsigned int i = blockIdx.x*(blockDim.x*4) + threadIdx.x;
    unsigned int id = threadIdx.x;

    __shared__ int localGLCM1[8 * 8];
    __shared__ int localGLCM2[8 * 8];
    __shared__ int localGLCM3[8 * 8];
    __shared__ int localGLCM4[8 * 8];

    localGLCM1[id] = SubGLCM[i] * ( (floorf(id/gl) - floorf(id/(gl*gl)) * gl )  + 1);
    localGLCM2[id] = SubGLCM[i + blockDim.x * 1] * ( (floorf(id/gl) - floorf(id/(gl*gl)) * gl )  + 1);
    localGLCM3[id] = SubGLCM[i + blockDim.x * 2] * ( (floorf(id/gl) - floorf(id/(gl*gl)) * gl )  + 1);
    localGLCM4[id] = SubGLCM[i + blockDim.x * 3] * ( (floorf(id/gl) - floorf(id/(gl*gl)) * gl )  + 1);


    __syncthreads();

    if (id < 32) {
        warpReduce(localGLCM1, id); 
        warpReduce(localGLCM2, id); 
        warpReduce(localGLCM3, id); 
        warpReduce(localGLCM4, id);  
    }  
    __syncthreads();
    if (id==0){
        meanY[blockIdx.x * 4 + 0] = localGLCM1[0];
        meanY[blockIdx.x * 4 + 1] = localGLCM2[0];
        meanY[blockIdx.x * 4 + 2] = localGLCM3[0];
        meanY[blockIdx.x * 4 + 3] = localGLCM4[0];
        }

}

GLOBAL void CalculateStddX(int gl,float* SubGLCM, float* meanX,float* stddX){
    //64 thread  laucnhed
    __shared__ float localGLCM1[8 * 8];
    __shared__ float localGLCM2[8 * 8];
    __shared__ float localGLCM3[8 * 8];
    __shared__ float localGLCM4[8 * 8];

    unsigned int i = blockIdx.x*(blockDim.x*4) + threadIdx.x;
    unsigned int id = threadIdx.x;
    
    localGLCM1[id] = SubGLCM[i] * __powf( meanX[blockIdx.x * 4 + 0] - id%gl  - 1 ,2);
    localGLCM2[id] = SubGLCM[i + blockDim.x * 1] * __powf( meanX[blockIdx.x * 4 + 1] - id%gl  - 1 ,2);
    localGLCM3[id] = SubGLCM[i + blockDim.x * 2] * __powf( meanX[blockIdx.x * 4 + 2] - id%gl  - 1 ,2);
    localGLCM4[id] = SubGLCM[i + blockDim.x * 3] * __powf( meanX[blockIdx.x * 4 + 3] - id%gl  - 1 ,2);
    
    if (id < 32) {
        warpReduce(localGLCM1, id); 
        warpReduce(localGLCM2, id); 
        warpReduce(localGLCM3, id); 
        warpReduce(localGLCM4, id);  
    }  
    __syncthreads();
    if (id==0){
        stddX[blockIdx.x * 4 + 0] =localGLCM1[0];
        stddX[blockIdx.x * 4 + 1] =localGLCM2[0];
        stddX[blockIdx.x * 4 + 2] =localGLCM3[0];
        stddX[blockIdx.x * 4 + 3] =localGLCM4[0];
        }
}

GLOBAL void CalculateStddY(int gl,float* SubGLCM, float* meanY,float* stddY){
    //64 thread  laucnhed
    __shared__ float localGLCM1[8 * 8];
    __shared__ float localGLCM2[8 * 8];
    __shared__ float localGLCM3[8 * 8];
    __shared__ float localGLCM4[8 * 8];

    unsigned int i = blockIdx.x*(blockDim.x*4) + threadIdx.x;
    unsigned int id = threadIdx.x;
    
    localGLCM1[id] = SubGLCM[i] * __powf( abs(meanY[blockIdx.x * 4 + 0] - (floorf(id/gl) + floorf(id/(gl*gl)) * gl )  - 1) ,2);
    localGLCM2[id] = SubGLCM[i + blockDim.x * 1] * __powf( abs( meanY[blockIdx.x * 4 + 1] - (floorf(id/gl) + floorf(id/(gl*gl)) * gl )  - 1 ),2);
    localGLCM3[id] = SubGLCM[i + blockDim.x * 2] * __powf( abs(meanY[blockIdx.x * 4 + 2] - (floorf(id/gl) + floorf(id/(gl*gl)) * gl )  - 1 ),2);
    localGLCM4[id] = SubGLCM[i + blockDim.x * 3] * __powf( abs(meanY[blockIdx.x * 4 + 3] - (floorf(id/gl) + floorf(id/(gl*gl)) * gl )  - 1) ,2);
    
    if (id < 32) {
        warpReduce(localGLCM1, id); 
        warpReduce(localGLCM2, id); 
        warpReduce(localGLCM3, id); 
        warpReduce(localGLCM4, id);  
    }  
    __syncthreads();
    if (id==0){
        stddY[blockIdx.x * 4 + 0] = localGLCM1[0];
        stddY[blockIdx.x * 4 + 1] = localGLCM2[0];
        stddY[blockIdx.x * 4 + 2] = localGLCM3[0];
        stddY[blockIdx.x * 4 + 3] = localGLCM4[0];
        }
}

GLOBAL void CorrelationFeature2(int gl, float* subGLCM, float* feature, float* meanX, float* meanY,float* stddX, float* stddY){
    
    //128 threads launched
    unsigned int blockdim = blockDim.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    unsigned int id = threadIdx.x;

    unsigned int value_index = blockIdx.x * 4 + floorf(id/(gl*gl));
    unsigned int id_step = blockdim+id;
    

    __shared__ float corSubGLCM[8 * 8 * 2];

    //First level reduction done while loading data
     corSubGLCM[id] = subGLCM[i] * ( floorf(id/gl) - gl * floorf(id/gl*gl) + 1 - meanY[value_index] ) * (id%gl + 1 - meanX[value_index])
                        /__powf((stddX[value_index] * stddY[value_index]),0.5)
                        +  subGLCM[i+blockDim.x] * ( floorf(id_step/gl) - gl * floorf(id_step/gl*gl) + 1 - meanY[value_index+2] ) * (id_step%gl + 1 - meanX[value_index+2])
                        /__powf((stddX[value_index+2] * stddY[value_index+2]),0.5); 
/*     corSubGLCM[id] = subGLCM[i] * __powf( abs(floorf(id/gl) - gl * floorf(id/gl*gl) + 1 - meanY[value_index]) ,2)
                        /stddX[value_index]
                        +  subGLCM[i+blockDim.x] * __powf(abs(floorf(id_step/gl) - gl * floorf(id_step/gl*gl) + 1 - meanY[value_index+2]),2)
                        /stddX[value_index+2]; */
                                
     __syncthreads();

    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (id < s) {
            corSubGLCM[id] += corSubGLCM[id + s];
        }
        __syncthreads();
    }
    if (id < 32) {
        warpReduce(corSubGLCM, id);
    } 
    
    if(id == 1)
        feature[blockIdx.x] = corSubGLCM[0]/4;
}

#endif
