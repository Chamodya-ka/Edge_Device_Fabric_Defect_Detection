#include <iostream>
#include <vector>

using namespace std;


class Image {

public:
    Image(vector<uint> pixels, unsigned int rows, unsigned int cols,unsigned int mingraylevel,
    unsigned int maxgraylevel);
    vector<uint> getPixels();
    unsigned int get_rows();
    unsigned int get_cols();
    unsigned int get_maxGL();
    vector<vector<uint8_t>> get2dVector();


private:
    vector<uint> pixels;
    unsigned int rows;
    unsigned int cols;
    unsigned int graylevel;
    unsigned int mingraylevel;
    unsigned int maxgraylevel;

};