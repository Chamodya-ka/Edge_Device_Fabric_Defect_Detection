#ifndef FEATURECALCULATION
#define FEATURECALCULATION
#define GLOBAL __global__ 
#define HOSTDEV __host__ __device__
#define DEV __device__

DEV void EnergyFeature(int id, int gl, int* subGLCM, float* feature){
    __shared__ float a,b,c,d;
    switch (id)
    {
    case 0:
        a=0;
        break;
    case 1:
        b=0;
    break;
    case 2:
        c=0;
    break;
    case 3:
        d=0;
    break;
    
    default:
        break;
    }

    __syncthreads();
    
    if (id<gl*gl){
        atomicAdd(&a,pow(subGLCM[id],2));
    }
    else if(id<gl*gl*2){
        atomicAdd(&b,pow(subGLCM[id],2));
    }
    else if(id<gl*gl*3){
        atomicAdd(&c,pow(subGLCM[id],2));
    }
    else if(id<gl*gl*4){
        atomicAdd(&d,pow(subGLCM[id],2));
    }
    //printf("%d",a);
    __syncthreads();
    if (id<=0)
        feature[id] = (float)(a+b+c+d)/4;

}

DEV void ContrastFeature(int id, int gl, int* subGLCM, float* feature){
    __shared__ float a,b,c,d;
    switch (id)
    {
    case 0:
        a=0;
        break;
    case 1:
        b=0;
    break;
    case 2:
        c=0;
    break;
    case 3:
        d=0;
    break;
    
    default:
        break;
    }

    __syncthreads();
    
    if (id<gl*gl){
        atomicAdd(&a,subGLCM[id] * abs(floorf(id/gl) - gl * 0 - fmodf(id,gl))); 
    }
    else if(id<gl*gl*2){
        atomicAdd(&a,subGLCM[id] * abs(floorf(id/gl) - gl * 1 - fmodf(id,gl))); 
    }
    else if(id<gl*gl*3){
        atomicAdd(&a,subGLCM[id] * abs(floorf(id/gl) - gl * 2 - fmodf(id,gl))); 
    }
    else if(id<gl*gl*4){
        atomicAdd(&a,subGLCM[id] * abs(floorf(id/gl) - gl * 3 - fmodf(id,gl))); 
    }
    //printf("%d",a);
    __syncthreads();
    if (id==1)
        feature[id] = (float)(a+b+c+d)/4;
}

#endif