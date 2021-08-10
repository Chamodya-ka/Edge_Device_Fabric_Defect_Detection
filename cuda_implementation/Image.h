#include <iostream>
#include <vector>

using namespace std;


class Image {

public:
    Image(vector<unsigned int> pixels, unsigned int rows, unsigned int columns,
    unsigned int graylevel);
    vector<unsigned int> getPixels();
    

private:
    vector<unsigned int> pixels;
    unsigned int rows;
    unsigned int columns;
    unsigned int graylevel;

};