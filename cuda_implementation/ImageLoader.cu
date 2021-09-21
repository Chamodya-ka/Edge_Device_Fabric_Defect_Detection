#include "ImageLoader.h"

/* 
    Currently using opencv without cuda
 */


Mat ImageLoader::readImageFromFile(string filename,int d_size){

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
    cv::resize(image,resized,Size(d_size,d_size),INTER_LINEAR);    
    image.release();
    return resized;
}

Image ImageLoader::readImage(string filename, unsigned int inmaxlevel, unsigned int inminlevel, unsigned int maxgraylevel){
    double minVal; 
    double maxVal; 
    Point minLoc; 
    Point maxLoc;

    
    Mat im = readImageFromFile(filename);
    minMaxLoc( im, &minVal, &maxVal, &minLoc, &maxLoc );
    cout <<  "Max pixel - " << maxVal << std::endl ;
    cout <<  "Min pixel - " << minVal << std::endl ;
    

    //im = quantize(im, maxgraylevel, inmaxlevel, inminlevel);
    vector<int> pixels = quantize(im, maxgraylevel, inmaxlevel, inminlevel);

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

