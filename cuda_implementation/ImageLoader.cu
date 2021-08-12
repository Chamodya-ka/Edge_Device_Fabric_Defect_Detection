#include "ImageLoader.h"

/* 
Currently using opencv without cuda
 */
Mat ImageLoader::readImageFromFile(string filename){

    Mat image;

    try{
        image = imread(filename, IMREAD_UNCHANGED);
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

    if((image.channels() != 1) || (image.type() != 8))
    {
        cout <<  "Changing the depth of image" << std::endl ;
        cvtColor(image, image, CV_RGB2GRAY);
        image.convertTo(image, CV_8UC1);
    }
    cout <<  "Channels - " << image.channels() << std::endl ;
    cout <<  "Type - " << image.type() << std::endl ;
    return image;

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
    
    //Debugging
    imshow("test1",im);
    Mat dst = im.clone();
    imshow("dest one",im);
    waitKey(0);
    //Debuggin end


    im = quantize(im, maxgraylevel, inmaxlevel, inminlevel);

    normalize(im, dst, 0, 255, NORM_MINMAX,CV_8UC1);
    imshow("dest2",dst);
    waitKey(0);

    vector<uint> pixels(im.total());

    if (im.isContinuous()) {
        pixels.assign((uint*)im.datastart, (uint*)im.dataend);
    } else {
        for (int i = 0; i < im.rows; ++i) {
            pixels.insert(pixels.end(), im.ptr<uint>(i), im.ptr<uint>(i)+im.cols);
        }
    }

    Image image = Image(pixels, im.rows, im.cols, 0, maxgraylevel);
    return image;
}

/* MAKE THIS QUANTIZE EFFICIENT */
 Mat ImageLoader::quantize(Mat& img, unsigned int maxgraylevel, unsigned int inmaxlevel ,unsigned int inminlevel)
{
    Mat convertedImage = img.clone();

    
    MatIterator_<uchar> it ;
    
    for (it =  convertedImage.begin<uchar>();it != convertedImage.end<uchar>(); it++){
        uint intensity = *it;
        std::cout << "old "+ std::to_string(intensity)  << "\n";
        uint newintensity = (uint)round( (intensity - inminlevel) * (maxgraylevel) / (double)(inmaxlevel-inminlevel) );
        *it = newintensity;
        std::cout << "new "+ std::to_string(newintensity)  << "\n";
    }
    
    return convertedImage; 

}



/* Mat ImageLoader::toGrayScale(const Mat& inputImage) {
    Mat convertedImage = inputImage.clone();
    normalize(convertedImage, convertedImage, 0, 255, NORM_MINMAX, CV_8UC1);
    return convertedImage;
} */