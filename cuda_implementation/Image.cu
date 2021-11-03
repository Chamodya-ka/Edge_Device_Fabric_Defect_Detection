#include "Image.h"


Image :: Image(const vector<int> &pixels, unsigned int rows, unsigned int cols,
unsigned int mingraylevel, unsigned int maxgraylevel){
    this->pixels = pixels;
    this->rows = rows;
    this->cols = cols;
    this->mingraylevel = mingraylevel;
    this->maxgraylevel = maxgraylevel;
}

vector<int> Image :: getPixels(){
    return pixels;
}
/* 
Used for visualizations only
 */
vector<vector<uint8_t>> Image :: get2dVector(){
    //vector<int> map_floor1D;
    cout<<to_string(rows * cols) +"="+ to_string(pixels.size())<<endl;
    vector<vector<uint8_t> > vector2d;
    vector2d.resize(rows);
    for (int i = 0; i < rows; i++)
    {
        vector2d[i].resize(cols);
    }
    
    for (int i = 0; i < pixels.size(); i++)
    {
        
        int row = i / rows;
        int col = i % cols;
        vector2d[row][col] = (uint8_t)pixels[i];
        
        
    }
    cout<<"getting 2d vector"<<endl;
    return vector2d;
}

uint Image :: get_rows(){
    return rows;
}
uint Image :: get_cols(){
    return cols;
}
uint Image :: get_maxGL(){
    return maxgraylevel;
}