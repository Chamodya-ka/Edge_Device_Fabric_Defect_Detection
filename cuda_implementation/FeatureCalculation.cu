#ifndef FEATURECALCULATION
#define FEATURECALCULATION
#define GLOBAL __global__ 
#define HOSTDEV __host__ __device__
#define DEV __device__

/* 

NOTE (i,j) of the GLCM should start with 1. 
In this case I have used indices of an array as i and j. Therefore +1 to both i and j

 */

GLOBAL void EnergyFeature(int gl, int* subGLCM, float* feature){
    int blockID = blockIdx.x;
    //int threadid = blockID * blockDim.x + threadIdx.x;
    int id = threadIdx.x;

    __shared__ float engSubGLCM[8 * 8 * 4];
    engSubGLCM[id] = pow(subGLCM[id],2);
     __syncthreads();
    if(fmodf(id,2)==0){
        engSubGLCM[id]+=engSubGLCM[id+1];
    }
    __syncthreads();
    if(fmodf(id,4)==0){
        engSubGLCM[id]+=engSubGLCM[id+2];
    }
    __syncthreads();
    if(fmodf(id,8)==0){
        engSubGLCM[id]+=engSubGLCM[id+4];
    }
    __syncthreads();
    if(fmodf(id,16)==0){
        engSubGLCM[id]+=engSubGLCM[id+8];
    }
    __syncthreads();
    if(fmodf(id,32)==0){
        engSubGLCM[id]+=engSubGLCM[id+16];
    }
    __syncthreads();
    if(fmodf(id,64)==0){
        engSubGLCM[id]+=engSubGLCM[id+32];
        //atomicAdd(&energyFeature, engSubGLCM[id]); 
    }
    __syncthreads();
    if (id==0){
        //feature[id]=energyFeature/4;
        feature[blockID] =( engSubGLCM[0] + engSubGLCM[64] + engSubGLCM[128] + engSubGLCM[192])/4;
    }
}

GLOBAL void ContrastFeature(int gl, int* subGLCM, float* feature){
    __shared__ float conSubGLCM[8 * 8 * 4];
    int blockID = blockIdx.x;
    //int threadid = blockID * blockDim.x + threadIdx.x;
    int id = threadIdx.x;
    conSubGLCM[id] = pow(subGLCM[id],2) * pow( floorf(id/gl) - gl * floorf(id/(gl*gl)) - fmodf(id,gl) , 2 );
    __syncthreads();
    if(fmodf(id,2)==0){
        conSubGLCM[id]+=conSubGLCM[id+1];
    }
    __syncthreads();
    if(fmodf(id,4)==0){
        conSubGLCM[id]+=conSubGLCM[id+2];
    }
    __syncthreads();
    if(fmodf(id,8)==0){
        conSubGLCM[id]+=conSubGLCM[id+4];
    }
    __syncthreads();
    if(fmodf(id,16)==0){
        conSubGLCM[id]+=conSubGLCM[id+8];
    }
    __syncthreads();
    if(fmodf(id,32)==0){
        conSubGLCM[id]+=conSubGLCM[id+16];
    }
    __syncthreads();
    if(fmodf(id,64)==0){
        conSubGLCM[id]+=conSubGLCM[id+32];
        
    }
    __syncthreads();
	if(fmodf(id,128)==0){
        conSubGLCM[id]+=conSubGLCM[id+64];
    }
	__syncthreads();
	if(fmodf(id,256)==0){
        conSubGLCM[id]+=conSubGLCM[id+128];
		feature[blockID] = conSubGLCM[id]/4;
    }
    //if (id==1){
    //   feature[blockID]=( conSubGLCM[0] + conSubGLCM[64] + conSubGLCM[128] + conSubGLCM[192])/4;
    //}
} 

GLOBAL void EntropyFeature(int gl, int* subGLCM, float* feature){

    __shared__ float entSubGlCM[8 * 8 * 4];
    int blockID = blockIdx.x;
    //int threadid = blockID * blockDim.x + threadIdx.x;
    int id = threadIdx.x;
    /*  CHECK HERE LOG(0) CAUSE NAN THEREFORE ADDED SMALL NUMBER */
    entSubGlCM[id] = subGLCM[id] *  logf(subGLCM[id]+1);
    __syncthreads();
    if(fmodf(id,2)==0){
        entSubGlCM[id]+=entSubGlCM[id+1];
    }
    __syncthreads();
    if(fmodf(id,4)==0){
        entSubGlCM[id]+=entSubGlCM[id+2];
    }
    __syncthreads();
    if(fmodf(id,8)==0){
        entSubGlCM[id]+=entSubGlCM[id+4];
    }
    __syncthreads();
    if(fmodf(id,16)==0){
        entSubGlCM[id]+=entSubGlCM[id+8];
    }
    __syncthreads();
    if(fmodf(id,32)==0){
        entSubGlCM[id]+=entSubGlCM[id+16];
    }
    __syncthreads();
    if(fmodf(id,64)==0){
        entSubGlCM[id]+=entSubGlCM[id+32];
       // atomicAdd(&entropyFeature, entSubGlCM[id]/4); 
    }
    __syncthreads();
    if (id==2){
        feature[blockID]=( entSubGlCM[0] + entSubGlCM[64] + entSubGlCM[128] + entSubGlCM[192])/4;
    }
    
}


GLOBAL void HomogeneityFeature(int gl, int* subGLCM, float* feature){
    int blockID = blockIdx.x;
    //int threadid = blockID * blockDim.x + threadIdx.x;
    int id = threadIdx.x;
    __shared__ float homSubGlCM[8 * 8 * 4];

    homSubGlCM[id] = subGLCM[id] *  (1 / (1 + pow( floorf(id/gl) - floorf(id/(gl*gl)) - fmodf(id,gl) , 2)) ); 
    __syncthreads();
    if(fmodf(id,2)==0){
        homSubGlCM[id]+=homSubGlCM[id+1];
    }
    __syncthreads();
    if(fmodf(id,4)==0){
        homSubGlCM[id]+=homSubGlCM[id+2];
    }
    __syncthreads();
    if(fmodf(id,8)==0){
        homSubGlCM[id]+=homSubGlCM[id+4];
    }
    __syncthreads();
    if(fmodf(id,16)==0){
        homSubGlCM[id]+=homSubGlCM[id+8];
    }
    __syncthreads();
    if(fmodf(id,32)==0){
        homSubGlCM[id]+=homSubGlCM[id+16];
    }
    __syncthreads();
    if(fmodf(id,64)==0){
        homSubGlCM[id]+=homSubGlCM[id+32];
        //atomicAdd(&homogeneityFeature, homSubGlCM[id]/4); 
    }
    __syncthreads();
    if (id==3){
        feature[blockID]=( homSubGlCM[0] + homSubGlCM[64] + homSubGlCM[128] + homSubGlCM[192])/4;
    }
    
}

GLOBAL void CorrelationFeature(int gl, int* subGLCM, float* feature){
    __shared__ float mean[4],stdd[4];
    int blockID = blockIdx.x;
    //int threadid = blockID * blockDim.x + threadIdx.x;
    int id = threadIdx.x;
    if ( id < 4){
        mean[id] = 0;
        stdd[id] = 0;
    }
    //USING REDUCTIONS 
    __shared__ int meanSubGLCM[8 * 8 * 4];
    meanSubGLCM[id] = subGLCM[id] * (floorf(id/gl) - gl * floorf(id/gl*gl)+1); // +1 because NOTE above
    
    __shared__ float stddSubGLCM[8 * 8 * 4];
    __shared__ float corrSubGLCM[8 * 8 * 4];
    
    
    
    __syncthreads();
    if (fmodf(id,2)==0){
        meanSubGLCM[id] += meanSubGLCM[id+1];
    }
    __syncthreads();
    if (fmodf(id,4)==0){
        meanSubGLCM[id] += meanSubGLCM[id+2];
    }
    __syncthreads();
    if (fmodf(id,8)==0){
        meanSubGLCM[id] += meanSubGLCM[id+4];
    }
    __syncthreads();
    if (fmodf(id,16)==0){
        meanSubGLCM[id] += meanSubGLCM[id+8];
    }
    __syncthreads();
    if (fmodf(id,32)==0){
        meanSubGLCM[id] += meanSubGLCM[id+16];
    }
    __syncthreads();
    if (fmodf(id,64)==0){
        meanSubGLCM[id] += meanSubGLCM[id+32];
        //mean[ (int) floorf(id/gl*gl)] = meanSubGLCM[id];
    }
    __syncthreads();
    switch (id)
    {
    case 0:
        mean[id] = meanSubGLCM[id*64];
        break;
    case 1:
        mean[id] = meanSubGLCM[id*64];
    break;
    case 2:
        mean[id] = meanSubGLCM[id*64];
    break;
    case 3:
        mean[id] = meanSubGLCM[id*64];
    break;
    default:
        break;
    }
    __syncthreads();
    stddSubGLCM[id] = subGLCM[id] * pow( floorf(id/gl) - gl * floorf(id/(gl*gl)) + 1  - mean[(int) floorf(id/(gl*gl))] , 2 );

    __syncthreads();
    if (fmodf(id,2)==0){
        stddSubGLCM[id] += stddSubGLCM[id+1];
    }
    __syncthreads();
    if (fmodf(id,4)==0){
        stddSubGLCM[id] += stddSubGLCM[id+2];
    }
    __syncthreads();
    if (fmodf(id,8)==0){
        stddSubGLCM[id] += stddSubGLCM[id+4];
    }
    __syncthreads();
    if (fmodf(id,16)==0){
        stddSubGLCM[id] += stddSubGLCM[id+8];
    }
    __syncthreads();
    if (fmodf(id,32)==0){
        stddSubGLCM[id] += stddSubGLCM[id+16];
    }
    __syncthreads();
    if (fmodf(id,64)==0){
        stddSubGLCM[id] += stddSubGLCM[id+32];
        //stdd[(int) floorf(id/64)] = pow(stddSubGLCM[id],0.5);
    }

	__syncthreads();
	switch (id)
    {
    case 0:
        stdd[id] = pow(stddSubGLCM[id*64],0.5);
        break;
    case 1:
        stdd[id] = pow(stddSubGLCM[id*64],0.5);
    break;
    case 2:
        stdd[id] = pow(stddSubGLCM[id*64],0.5);
    break;
    case 3:
        stdd[id] = pow(stddSubGLCM[id*64],0.5);
    break;
    default:
        break;
    }
    __syncthreads();

    /* CHECK THIS DIV BY 0 */
    corrSubGLCM[id] = subGLCM[id] * ((floorf(id/gl) - gl * floorf(id/(gl*gl)) - mean[(int) floorf(id/(gl*gl))] + 1) * 
                                            (fmodf(id,gl) - mean[(int) floorf(id/(gl*gl))] +1)) / 
                                            pow(stdd[(int) floorf(id/(gl*gl))] + 0.0000000001,2); 

    __syncthreads();
    if (fmodf(id,2)==0){
        corrSubGLCM[id] += corrSubGLCM[id+1];
    }
    __syncthreads();
    if (fmodf(id,4)==0){
        corrSubGLCM[id] += corrSubGLCM[id+2];
    }
    __syncthreads();
    if (fmodf(id,8)==0){
        corrSubGLCM[id] += corrSubGLCM[id+4];
    }
    __syncthreads();
    if (fmodf(id,16)==0){
        corrSubGLCM[id] += corrSubGLCM[id+8];
    }
    __syncthreads();
    if (fmodf(id,32)==0){
        corrSubGLCM[id] += corrSubGLCM[id+16];
    }
    __syncthreads();
    if (fmodf(id,64)==0){
        corrSubGLCM[id] += corrSubGLCM[id+32];
        //atomicAdd( &corrFeature , corrSubGLCM[id]/4 );
        //corrFeature += corrSubGLCM[id];
    }
	__syncthreads();
    if (id==4){
        feature[blockID]=( corrSubGLCM[0] + corrSubGLCM[64] + corrSubGLCM[128] + corrSubGLCM[192])/4;
    }
    

}

/*
GLOBAL void EnergyFeature(int id, int gl, int* subGLCM, float* feature){
    
    __shared__ float engSubGLCM[8 * 8 * 4];
    engSubGLCM[id] = pow(subGLCM[id],2);
     __syncthreads();
    if(fmodf(id,2)==0){
        engSubGLCM[id]+=engSubGLCM[id+1];
    }
    __syncthreads();
    if(fmodf(id,4)==0){
        engSubGLCM[id]+=engSubGLCM[id+2];
    }
    __syncthreads();
    if(fmodf(id,8)==0){
        engSubGLCM[id]+=engSubGLCM[id+4];
    }
    __syncthreads();
    if(fmodf(id,16)==0){
        engSubGLCM[id]+=engSubGLCM[id+8];
    }
    __syncthreads();
    if(fmodf(id,32)==0){
        engSubGLCM[id]+=engSubGLCM[id+16];
    }
    __syncthreads();
    if(fmodf(id,64)==0){
        engSubGLCM[id]+=engSubGLCM[id+32];
        //atomicAdd(&energyFeature, engSubGLCM[id]); 
    }
    __syncthreads();
    if (id==0){
        //feature[id]=energyFeature/4;
        feature[id] =( engSubGLCM[0] + engSubGLCM[64] + engSubGLCM[128] + engSubGLCM[192])/4;
    }
}


DEV void ContrastFeature(int id, int gl, int* subGLCM, float* feature){
    __shared__ float conSubGLCM[8 * 8 * 4];

    conSubGLCM[id] = pow(subGLCM[id],2) * pow( floorf(id/gl) - gl * floorf(id/(gl*gl)) - fmodf(id,gl) , 2 );
    __syncthreads();
    if(fmodf(id,2)==0){
        conSubGLCM[id]+=conSubGLCM[id+1];
    }
    __syncthreads();
    if(fmodf(id,4)==0){
        conSubGLCM[id]+=conSubGLCM[id+2];
    }
    __syncthreads();
    if(fmodf(id,8)==0){
        conSubGLCM[id]+=conSubGLCM[id+4];
    }
    __syncthreads();
    if(fmodf(id,16)==0){
        conSubGLCM[id]+=conSubGLCM[id+8];
    }
    __syncthreads();
    if(fmodf(id,32)==0){
        conSubGLCM[id]+=conSubGLCM[id+16];
    }
    __syncthreads();
    if(fmodf(id,64)==0){
        conSubGLCM[id]+=conSubGLCM[id+32];
        //atomicAdd(&contrastFeature, conSubGLCM[id]/4); 
    }
    __syncthreads();
    if (id==1){
        feature[id]=( conSubGLCM[0] + conSubGLCM[64] + conSubGLCM[128] + conSubGLCM[192])/4;
    }
} 
*/
/*
GLOBAL void EntropyFeature(int id, int gl, int* subGLCM, float* feature){

    __shared__ float entSubGlCM[8 * 8 * 4];

    /*  CHECK HERE LOG(0) CAUSE NAN THEREFORE ADDED SMALL NUMBER */
    /*entSubGlCM[id] = subGLCM[id] *  logf(subGLCM[id]+1);
    __syncthreads();
    if(fmodf(id,2)==0){
        entSubGlCM[id]+=entSubGlCM[id+1];
    }
    __syncthreads();
    if(fmodf(id,4)==0){
        entSubGlCM[id]+=entSubGlCM[id+2];
    }
    __syncthreads();
    if(fmodf(id,8)==0){
        entSubGlCM[id]+=entSubGlCM[id+4];
    }
    __syncthreads();
    if(fmodf(id,16)==0){
        entSubGlCM[id]+=entSubGlCM[id+8];
    }
    __syncthreads();
    if(fmodf(id,32)==0){
        entSubGlCM[id]+=entSubGlCM[id+16];
    }
    __syncthreads();
    if(fmodf(id,64)==0){
        entSubGlCM[id]+=entSubGlCM[id+32];
       // atomicAdd(&entropyFeature, entSubGlCM[id]/4); 
    }
    __syncthreads();
    if (id==2){
        feature[id]=( entSubGlCM[0] + entSubGlCM[64] + entSubGlCM[128] + entSubGlCM[192])/4;
    }
    
}


GLOBAL void HomogeneityFeature(int id, int gl, int* subGLCM, float* feature){

    __shared__ float homSubGlCM[8 * 8 * 4];

    homSubGlCM[id] = subGLCM[id] *  (1 / (1 + pow( floorf(id/gl) - floorf(id/(gl*gl)) - fmodf(id,gl) , 2)) ); 
    __syncthreads();
    if(fmodf(id,2)==0){
        homSubGlCM[id]+=homSubGlCM[id+1];
    }
    __syncthreads();
    if(fmodf(id,4)==0){
        homSubGlCM[id]+=homSubGlCM[id+2];
    }
    __syncthreads();
    if(fmodf(id,8)==0){
        homSubGlCM[id]+=homSubGlCM[id+4];
    }
    __syncthreads();
    if(fmodf(id,16)==0){
        homSubGlCM[id]+=homSubGlCM[id+8];
    }
    __syncthreads();
    if(fmodf(id,32)==0){
        homSubGlCM[id]+=homSubGlCM[id+16];
    }
    __syncthreads();
    if(fmodf(id,64)==0){
        homSubGlCM[id]+=homSubGlCM[id+32];
        //atomicAdd(&homogeneityFeature, homSubGlCM[id]/4); 
    }
    __syncthreads();
    if (id==3){
        feature[id]=( homSubGlCM[0] + homSubGlCM[64] + homSubGlCM[128] + homSubGlCM[192])/4;
    }
    
}

GLOBAL void CorrelationFeature(int id, int gl, int* subGLCM, float* feature){
    __shared__ float mean[4],stdd[4];
    
    if ( id < 4){
        mean[id] = 0;
        stdd[id] = 0;
    }
    //USING REDUCTIONS 
    __shared__ int meanSubGLCM[8 * 8 * 4];
    meanSubGLCM[id] = subGLCM[id] * (floorf(id/gl) - gl * floorf(id/gl*gl)+1); // +1 because NOTE above
    
    __shared__ float stddSubGLCM[8 * 8 * 4];
    __shared__ float corrSubGLCM[8 * 8 * 4];
    
    
    
    __syncthreads();
    if (fmodf(id,2)==0){
        meanSubGLCM[id] += meanSubGLCM[id+1];
    }
    __syncthreads();
    if (fmodf(id,4)==0){
        meanSubGLCM[id] += meanSubGLCM[id+2];
    }
    __syncthreads();
    if (fmodf(id,8)==0){
        meanSubGLCM[id] += meanSubGLCM[id+4];
    }
    __syncthreads();
    if (fmodf(id,16)==0){
        meanSubGLCM[id] += meanSubGLCM[id+8];
    }
    __syncthreads();
    if (fmodf(id,32)==0){
        meanSubGLCM[id] += meanSubGLCM[id+16];
    }
    __syncthreads();
    if (fmodf(id,64)==0){
        meanSubGLCM[id] += meanSubGLCM[id+32];
        //mean[ (int) floorf(id/gl*gl)] = meanSubGLCM[id];
    }
    __syncthreads();
    switch (id)
    {
    case 0:
        mean[id] = meanSubGLCM[id*64];
        break;
    case 1:
        mean[id] = meanSubGLCM[id*64];
    break;
    case 2:
        mean[id] = meanSubGLCM[id*64];
    break;
    case 3:
        mean[id] = meanSubGLCM[id*64];
    break;
    default:
        break;
    }
    __syncthreads();
    stddSubGLCM[id] = subGLCM[id] * pow( floorf(id/gl) - gl * floorf(id/(gl*gl)) + 1  - mean[(int) floorf(id/(gl*gl))] , 2 );

    __syncthreads();
    if (fmodf(id,2)==0){
        stddSubGLCM[id] += stddSubGLCM[id+1];
    }
    __syncthreads();
    if (fmodf(id,4)==0){
        stddSubGLCM[id] += stddSubGLCM[id+2];
    }
    __syncthreads();
    if (fmodf(id,8)==0){
        stddSubGLCM[id] += stddSubGLCM[id+4];
    }
    __syncthreads();
    if (fmodf(id,16)==0){
        stddSubGLCM[id] += stddSubGLCM[id+8];
    }
    __syncthreads();
    if (fmodf(id,32)==0){
        stddSubGLCM[id] += stddSubGLCM[id+16];
    }
    __syncthreads();
    if (fmodf(id,64)==0){
        stddSubGLCM[id] += stddSubGLCM[id+32];
        //stdd[(int) floorf(id/64)] = pow(stddSubGLCM[id],0.5);
    }

	__syncthreads();
	switch (id)
    {
    case 0:
        stdd[id] = pow(stddSubGLCM[id*64],0.5);
        break;
    case 1:
        stdd[id] = pow(stddSubGLCM[id*64],0.5);
    break;
    case 2:
        stdd[id] = pow(stddSubGLCM[id*64],0.5);
    break;
    case 3:
        stdd[id] = pow(stddSubGLCM[id*64],0.5);
    break;
    default:
        break;
    }
    __syncthreads();

    /* CHECK THIS DIV BY 0 */
    /*corrSubGLCM[id] = subGLCM[id] * ((floorf(id/gl) - gl * floorf(id/(gl*gl)) - mean[(int) floorf(id/(gl*gl))] + 1) * 
                                            (fmodf(id,gl) - mean[(int) floorf(id/(gl*gl))] +1)) / 
                                            pow(stdd[(int) floorf(id/(gl*gl))] + 0.0000000001,2); 

    __syncthreads();
    if (fmodf(id,2)==0){
        corrSubGLCM[id] += corrSubGLCM[id+1];
    }
    __syncthreads();
    if (fmodf(id,4)==0){
        corrSubGLCM[id] += corrSubGLCM[id+2];
    }
    __syncthreads();
    if (fmodf(id,8)==0){
        corrSubGLCM[id] += corrSubGLCM[id+4];
    }
    __syncthreads();
    if (fmodf(id,16)==0){
        corrSubGLCM[id] += corrSubGLCM[id+8];
    }
    __syncthreads();
    if (fmodf(id,32)==0){
        corrSubGLCM[id] += corrSubGLCM[id+16];
    }
    __syncthreads();
    if (fmodf(id,64)==0){
        corrSubGLCM[id] += corrSubGLCM[id+32];
        //atomicAdd( &corrFeature , corrSubGLCM[id]/4 );
        //corrFeature += corrSubGLCM[id];
    }
	__syncthreads();
    if (id==4){
        feature[id]=( corrSubGLCM[0] + corrSubGLCM[64] + corrSubGLCM[128] + corrSubGLCM[192])/4;
    }
    

}
*/
#endif
