#include "Image.h"


Image :: Image(vector<unsigned int> pixels, unsigned int rows, unsigned int cols,
unsigned int mingraylevel, unsigned int maxgraylevel){
    this->pixels = pixels;
    this->rows = rows;
    this->cols = cols;
    this->mingraylevel = mingraylevel;
    this->maxgraylevel = maxgraylevel;
}

vector<uint> Image :: getPixels(){
    return pixels;
}