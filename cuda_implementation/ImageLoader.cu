#include "ImageLoader.h"

Mat ImageLoader::readImageFromFile(string filename){

    Mat image;

    try{
        image = imread(filename);
    }catch (cv::Exception& e) {
        const char *err_msg = e.what();
        cerr << "Exception occurred: " << err_msg << endl;
    }

    if(! image.data )  // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        exit(-1);
    }
    // If the input is not a gray-scale image, it is converted in a color image
    if((image.depth() != CV_8UC1))
    {
        // Reducing the number of color channels from 3 to 1
        cvtColor(image, image, CV_RGB2GRAY);
        image.convertTo(image, CV_8UC1);
    }

    return image;

}

Image ImageLoader::readImage(string filename){
    Mat im = readImageFromFile(filename);

    
}
/* MAKE THIS QUANTIZE EFFICIENT */
Mat ImageLoader::quantize(Mat& img, unsigned int graylevel ,unsigned int minlevel)
{
    Mat convertedImage = img.clone();

    typedef MatIterator_<ushort> MI;

    for(MI element = convertedImage.begin<ushort>() ; element != convertedImage.end<ushort>() ; element++)
    {

        uint intensity = *element;
        uint newIntensity = (uint)round((intensity - minlevel) * (graylevel))/(double)(graylevel - minlevel));

        *element = newIntensity;
    }

    return convertedImage;
}
Mat ImageLoader::toGrayScale(const Mat& inputImage) {
    // Converting the image to a 255 gray-scale
    Mat convertedImage = inputImage.clone();
    normalize(convertedImage, convertedImage, 0, 255, NORM_MINMAX, CV_8UC1);
    return convertedImage;
}