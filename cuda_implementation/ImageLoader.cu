#include "ImageLoader.h"

/* 
Currently using opencv without cuda
 */
Mat ImageLoader::readImageFromFile(string filename){

    Mat image;

    try{
        image = imread(filename);
    }catch (cv::Exception& e) {
        const char *err_msg = e.what();
        cerr << "Exception occurred: " << err_msg << endl;
    }

    if(! image.data )  
    {
        cout <<  "Could not open or find the image" << std::endl ;
        exit(-1);
    }

    if((image.depth() != CV_8UC1))
    {

        cvtColor(image, image, CV_RGB2GRAY);
        image.convertTo(image, CV_8UC1);
    }

    return image;

}

Image ImageLoader::readImage(string filename, unsigned int maxgraylevel = 8, unsigned int inmaxlevel = 255 , unsigned int inminlevel = 0 ){
    Mat im = readImageFromFile(filename);
    im = quantize(im, maxgraylevel, inmaxlevel, inminlevel);
  //  Vector<uchar> pixels(im.total);
    vector<uint> pixels(im.total());
    //readUchars(pixels,im);
    if (im.isContinuous()) {
        pixels.assign((uchar*)im.datastart, (uchar*)im.dataend);
    } else {
        for (int i = 0; i < im.rows; ++i) {
            pixels.insert(pixels.end(), im.ptr<uchar>(i), im.ptr<uchar>(i)+im.cols);
        }
    }
    Image image = Image(pixels, im.rows, im.cols, 0, maxgraylevel);
    return image;
}

/* MAKE THIS QUANTIZE EFFICIENT */
 Mat ImageLoader::quantize(Mat& img, unsigned int maxgraylevel, unsigned int inmaxlevel ,unsigned int inminlevel)
{
    Mat convertedImage = img.clone();

    convertedImage = img / inminlevel;
    MatIterator_<uchar> it ;
    
    for (it =  convertedImage.begin<uchar>();it != convertedImage.end<uchar>(); it++){
        uchar intensity = *it;
        uchar newintensity = (uchar)round( (intensity - inminlevel) * (maxgraylevel) / (inmaxlevel-inminlevel) );
        *it = newintensity;

    }
    
    return convertedImage; 

}



/* Mat ImageLoader::toGrayScale(const Mat& inputImage) {
    Mat convertedImage = inputImage.clone();
    normalize(convertedImage, convertedImage, 0, 255, NORM_MINMAX, CV_8UC1);
    return convertedImage;
} */