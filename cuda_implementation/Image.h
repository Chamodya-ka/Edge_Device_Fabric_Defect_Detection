#include <iostream>
#include <vector>

using namespace std;


class Image {

public:
    Image(vector<unsigned int> pixels, unsigned int rows, unsigned int cols,unsigned int mingraylevel,
    unsigned int maxgraylevel);
    vector<unsigned int> getPixels();
    

private:
    vector<unsigned int> pixels;
    unsigned int rows;
    unsigned int cols;
    unsigned int graylevel;
    unsigned int mingraylevel;
    unsigned int maxgraylevel;

};