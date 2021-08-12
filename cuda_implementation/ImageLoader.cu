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

Image ImageLoader::readImage(string filename, unsigned int inmaxlevel, unsigned int inminlevel, unsigned int maxgraylevel){
    Mat im = readImageFromFile(filename);
    
    //Debugging
    
    imshow("test1",im);
    waitKey(0);
    //Debuggin end
    im = quantize(im, maxgraylevel, inmaxlevel, inminlevel);

    Mat dst;
    normalize(im, dst, 0, 255, cv::NORM_MINMAX,CV_8UC1);
    imshow("test2",dst);
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

    
    MatIterator_<ushort> it ;
    
    for (it =  convertedImage.begin<ushort>();it != convertedImage.end<ushort>(); it++){
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