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


GLOBAL void EnergyFeature2(int gl, int* subGLCM, float* feature){
    //unsigned int blockID = blockIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int id = threadIdx.x;

    __shared__ float engSubGLCM[8 * 8 * 2];

    //First level reduction done while loading data
    engSubGLCM[id] = pow(subGLCM[i],2) + pow(subGLCM[i+blockDim.x] , 2);
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

GLOBAL void ContrastFeature2(int gl, int* subGLCM, float* feature){
    //unsigned int blockID = blockIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int id = threadIdx.x;

    __shared__ float conSubGLCM[8 * 8 * 2];

    //First level reduction done while loading data
    conSubGLCM[id] = subGLCM[i] * pow( (id%gl - floorf(id/gl) + gl * floorf(id/(gl*gl))) ,2) + 
                                subGLCM[i+blockDim.x] * pow( ((id+blockDim.x)%gl - floorf((id+blockDim.x)/gl) + gl * floorf((id+blockDim.x)/(gl*gl))) ,2);
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

GLOBAL void EntropyFeature2(int gl, int* subGLCM, float* feature){
    //unsigned int blockID = blockIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int id = threadIdx.x;

    __shared__ float conSubGLCM[8 * 8 * 2];

    //First level reduction done while loading data
    conSubGLCM[id] = subGLCM[i] *  logf(subGLCM[i]+1) + 
                                subGLCM[i+blockDim.x] *  logf(subGLCM[i+blockDim.x]+1);
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

GLOBAL void HomogeneityFeature2(int gl, int* subGLCM, float* feature){
    //unsigned int blockID = blockIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int id = threadIdx.x;

    __shared__ float homSubGLCM[8 * 8 * 2];

    //First level reduction done while loading data
    homSubGLCM[id] = subGLCM[i] / ( 1 +  pow( (id%gl - floorf(id/gl) + gl * floorf(id/(gl*gl))) ,2) ) + 
                                subGLCM[i+blockDim.x] / ( 1 +  pow( (id+blockDim.x)%gl - floorf(id+blockDim.x/gl) + gl * floorf(id+blockDim.x/(gl*gl)) ,2) );
     __syncthreads();

    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (id < s) {
            homSubGLCM[id] += homSubGLCM[id + s];
        }
        __syncthreads();
    }
    if (id < 32) warpReduce(homSubGLCM, id);

    if(id == 1)
        feature[blockIdx.x] = homSubGLCM[0]/4;

}



GLOBAL void CorrelationFeature2(int gl, int* subGLCM, float* feature){
    //unsigned int blockID = blockIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int id = threadIdx.x;

    __shared__ float meanSubGLCMx[8 * 8 * 2];
    __shared__ float meanSubGLCMy[8 * 8 * 2];
    __shared__ float stddSubGLCMx[8 * 8 * 2];
    __shared__ float stddSubGLCMy[8 * 8 * 2];
    __shared__ float corSubGLCM[8 * 8 * 2];

    //First level reduction done while loading data
    meanSubGLCMx[id] = subGLCM[i] * (floorf(id/gl) - gl * floorf(id/gl*gl)+1) + 
                                subGLCM[i+blockDim.x] * (floorf((id+blockDim.x)/gl) - gl * floorf((id+blockDim.x)/gl*gl)+1);
    meanSubGLCMy[id] = subGLCM[i] * (id%gl+1) + 
                                subGLCM[i+blockDim.x] * ((id+blockDim.x)%gl + 1);
     __syncthreads();

    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (id < s) {
            meanSubGLCMx[id] += meanSubGLCMx[id + s];
            meanSubGLCMy[id] += meanSubGLCMy[id + s];
        }
        __syncthreads();
    }
    if (id < 32) {
        warpReduce(meanSubGLCMx, id);
        warpReduce(meanSubGLCMy, id);
    }

    stddSubGLCMx[id] = subGLCM[i] * pow((floorf(id/gl) - gl * floorf(id/gl*gl)+1) - meanSubGLCMx[0],2) +
                                subGLCM[i+blockDim.x] * pow((floorf((id+blockDim.x)/gl) - gl * floorf((id+blockDim.x)/gl*gl)+1) - meanSubGLCMx[0],2);
    stddSubGLCMy[id] = subGLCM[i] * pow(id%gl - meanSubGLCMy[0],2) +
                                subGLCM[i+blockDim.x] * pow((id+blockDim.x)%gl - meanSubGLCMy[0],2);                           
    
    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (id < s) {
            stddSubGLCMx[id] += stddSubGLCMx[id + s];
            stddSubGLCMy[id] += stddSubGLCMy[id + s];
        }
        __syncthreads();
    }
    if (id < 32) {
        warpReduce(stddSubGLCMx, id);
        warpReduce(stddSubGLCMy, id);
    }

    corSubGLCM[id] = subGLCM[i] * (floorf(id/gl) - gl * floorf(id/gl*gl)+1 - meanSubGLCMx[0]) * (id%gl - meanSubGLCMy[0]) / ( stddSubGLCMx[0] * stddSubGLCMy[0] + + 0.0000000001 )+
                                subGLCM[i+blockDim.x] * (floorf((id+blockDim.x)/gl) - gl * floorf((id+blockDim.x)/gl*gl)+1 - meanSubGLCMx[0]) * ((id+blockDim.x)%gl - meanSubGLCMy[0]) / ( stddSubGLCMx[0] * stddSubGLCMy[0] + + 0.0000000001);
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