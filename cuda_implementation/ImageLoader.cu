#include "ImageLoader.h"
#define GLOBAL __global__ 

/* 
    Currently using opencv without cuda
 */

GLOBAL void quantizeArray(uchar* arr,int* out,unsigned int max_gl,unsigned  int min_gl,unsigned int desired_gl,int N){
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx<N){
        out[idx] = rintf(__fdividef( (arr[idx] - min_gl) * (desired_gl) , (max_gl-min_gl)));
    }

}
Mat ImageLoader::readImageFromFile(string filename,unsigned int d_sizex, unsigned int d_sizey){

    Mat image;

    try{
        image = imread(filename, IMREAD_GRAYSCALE);
    }catch (cv::Exception& e) {
        const char *err_msg = e.what();
        cerr << "Exception occurred: " << err_msg << endl;
    }

    if(! image.data )  
    {
        cout <<  "Could not open or find the image" << std::endl ;
        exit(-1);
    }
    cout <<  "Channels - " << image.channels() << std::endl ;
    cout <<  "Type - " << image.type() << std::endl ;

    if((image.channels() != 1) || (image.type() != 0))
    {
        cout <<  "Changing the depth of image" << std::endl ;
        //cvtColor(image, image, CV_RGB2GRAY);
        image.convertTo(image, CV_8UC1);
    }
    cout <<  "Channels - " << image.channels() << std::endl ;
    cout <<  "Type - " << image.type() << std::endl ;
    cout <<  "Depth - " << image.depth() << std::endl ;

    Mat resized;
    cv::resize(image,resized,Size(d_sizex,d_sizey),INTER_LINEAR);    
    image.release();
    return resized;
}

Image ImageLoader::readImage(string filename, unsigned int inmaxlevel, unsigned int inminlevel, unsigned int maxgraylevel,
                                unsigned int d_sizex, unsigned int d_sizey){
    double minVal; 
    double maxVal; 
    Point minLoc; 
    Point maxLoc;

    
    Mat im = readImageFromFile(filename, d_sizex,d_sizey);
    minMaxLoc( im, &minVal, &maxVal, &minLoc, &maxLoc );
    cout <<  "Max pixel - " << maxVal << std::endl ;
    cout <<  "Min pixel - " << minVal << std::endl ;
    

    //im = quantize(im, maxgraylevel, inmaxlevel, inminlevel);
    vector<int> pixels = quantize2(im, maxgraylevel, inmaxlevel, inminlevel);
    //vector<int> pixels = quantize(im, maxgraylevel, inmaxlevel, inminlevel);

    cout<<"MATRIX TOTAL = "+to_string(im.total()) +"| pixels length = "+ to_string(pixels.size())<<endl;
    
    
    /* Used to generate test iamge */

    /* im = getQuantizedMat(im, maxgraylevel, maxVal, minVal);
    normalize(im, im, 0, 255, NORM_MINMAX,CV_8UC1);
    imshow("New converted mat",im);
    waitKey(0);
    imwrite("../testimg/Galaxytest.png", im); */
    
    Image image = Image(pixels, im.rows, im.cols, 0, maxgraylevel);
    return image;
}

Mat ImageLoader:: getQuantizedMat(Mat& img, unsigned int maxgraylevel, unsigned int inmaxlevel ,unsigned int inminlevel){
    
    Mat convertedImage = img.clone();
    MatIterator_<uchar> it ;
    int c = 0;
    //int array[10] = {9,9,9,9,9,9,9,9,9,9} ;
    for (it =  convertedImage.begin<uchar>();it != convertedImage.end<uchar>(); it++){
        uint8_t intensity = *it;
        uint8_t newintensity = (uint8_t)round( (intensity - inminlevel) * (maxgraylevel) / (double)(inmaxlevel-inminlevel) );
        *it = newintensity;
        c++;
    }

    return convertedImage;


}

/* MAKE THIS QUANTIZE EFFICIENT */
 vector<int> ImageLoader::quantize(Mat& img, unsigned int maxgraylevel, unsigned int inmaxlevel ,unsigned int inminlevel)
{

    

    vector<int> pixels;
    pixels.reserve(img.total());
    cout<<"AT INITIALIZATION SIZE =  " + to_string(pixels.size())<<endl;
    MatIterator_<uchar> it ;
    int c = 0;

    for (it =  img.begin<uchar>();it != img.end<uchar>(); it++){
        uint intensity = *it;
        int newintensity = (int)round( (intensity - inminlevel) * (maxgraylevel) / (double)(inmaxlevel-inminlevel) );
        pixels.push_back(newintensity);
        c++;
    }
    cout << "number of times items inserted " +to_string(c) << endl;
    return pixels; 

}
 vector<int> ImageLoader::quantize2(Mat& img, unsigned int maxgraylevel, unsigned int inmaxlevel ,unsigned int inminlevel)
{
    //int* inPixels;    
    //inPixels = img.ptr<int>(0);
    int N = img.rows * img.cols * img.channels();
    
    uchar* inPixels = img.data;
    int inBytes = N* sizeof(uchar);
    int outBytes = N * sizeof(int);

    uchar* dInPixels;
    int* dOutPixels;
    int* hOutPixels = (int*) malloc(outBytes);
    cudaMalloc(&dInPixels,inBytes);
    cudaMalloc(&dOutPixels,outBytes);
    cudaMemcpy(dInPixels,inPixels,inBytes,cudaMemcpyHostToDevice);
    
    int threads = 256;
    dim3 THREADS(threads);
    dim3 BLOCKS(( N - 1 + threads )/ threads);
    std::cout<<"here"<<endl;
    quantizeArray<<<BLOCKS,THREADS>>>(dInPixels,dOutPixels,inmaxlevel,inminlevel,maxgraylevel,N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(hOutPixels,dOutPixels,outBytes,cudaMemcpyDeviceToHost);
    /* CONTINUE ARRAY TO VECTOR AND RETURN */
    std::vector<int> pixels(hOutPixels,hOutPixels+N); 
    return pixels;

}
