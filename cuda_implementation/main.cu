#include "ImageLoader.h"
#include "GLCMComputation.h"
#include "FeatureComputation.h"
#include "knncuda.h"
#include <sys/time.h>
#include "Image.h"
#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
using namespace std::chrono;

//nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
//#include <opencv2/opencv.hpp>
//using namespace cv;
/* 
    Main method  --
 */
using namespace std;

/**
 * Test an input k-NN function implementation by verifying that its output
 * results (distances and corresponding indexes) are similar to the expected
 * results (ground truth).
 *
 * Since the k-NN computation might end-up in slightly different results
 * compared to the expected one depending on the considered implementation,
 * the verification consists in making sure that the accuracy is high enough.
 *
 * The tested function is ran several times in order to have a better estimate
 * of the processing time.
 *
 * @param ref            reference points
 * @param ref_nb         number of reference points
 * @param query          query points
 * @param query_nb       number of query points
 * @param dim            dimension of reference and query points
 * @param k              number of neighbors to consider
 * @param gt_knn_dist    ground truth distances
 * @param gt_knn_index   ground truth indexes
 * @param knn            function to test
 * @param name           name of the function to test (for display purpose)
 * @param nb_iterations  number of iterations
 * return false in case of problem, true otherwise
 */
bool test(const float * ref,
          int           ref_nb,
          const float * query,
          int           query_nb,
          int           dim,
          int           k,
          float *       gt_knn_dist,
          int *         gt_knn_index,
          bool (*knn)(const float *, int, const float *, int, int, int, float *, int *),
          const char *  name,
          int           nb_iterations) {

    // Parameters
    const float precision    = 0.2f; // distance error max
    const float min_accuracy = 0.5f; // percentage of correct values required

    // Display k-NN function name
    printf("- %-17s : ", name);

    // Allocate memory for computed k-NN neighbors
    float * test_knn_dist  = (float*) malloc(query_nb * k * sizeof(float));
    int   * test_knn_index = (int*)   malloc(query_nb * k * sizeof(int));

    // Allocation check
    if (!test_knn_dist || !test_knn_index) {
        printf("ALLOCATION ERROR\n");
        free(test_knn_dist);
        free(test_knn_index);
        return false;
    }

    // Start timer
    struct timeval tic;
    gettimeofday(&tic, NULL);

    // Compute k-NN several times
    for (int i=0; i<nb_iterations; ++i) {
        if (!knn(ref, ref_nb, query, query_nb, dim, k, test_knn_dist, test_knn_index)) {
            free(test_knn_dist);
            free(test_knn_index);
            return false;
        }
    }

    // Stop timer
    struct timeval toc;
    gettimeofday(&toc, NULL);

    // Elapsed time in ms
    double elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;

    // Verify both precisions and indexes of the k-NN values
    int nb_correct_precisions = 0;
    int nb_correct_indexes    = 0;
    for (int i=0; i<query_nb*k; ++i) {
        if (fabs(test_knn_dist[i] - gt_knn_dist[i]) <= precision) {
            nb_correct_precisions++;
        }
        if (test_knn_index[i] == gt_knn_index[i]) {
            nb_correct_indexes++;
        }
    }

    // Compute accuracy
    float precision_accuracy = nb_correct_precisions / ((float) query_nb * k);
    float index_accuracy     = nb_correct_indexes    / ((float) query_nb * k);

    // Display report
    if (precision_accuracy >= min_accuracy && index_accuracy >= min_accuracy ) {
        printf("PASSED in %8.5f seconds (averaged over %3d iterations)\n", elapsed_time / nb_iterations, nb_iterations);
    }
    else {
        printf("FAILED\n");
    }

    // Free memory
    free(test_knn_dist);
    free(test_knn_index);

    return true;
}

int main(int argc, char *argv[]){
    unsigned int  maxgl = 255;
    unsigned int  mingl = 0;
    unsigned int  desiredgl = 7;
    unsigned int subImgDim = 32;
    uint r = 2048;
    uint c = 2048 ;
    int N = r*c/(subImgDim*subImgDim);
    GLCMComputation glcm = GLCMComputation();
     if (argc>1){
        if (!strcmp(argv[1],"-n")){
            cout<<"Generating csv records"<<"\n";
            fstream fout;
            fout.open("../ref_data/ref_csv/ref_data2.csv",ios::out);
            string fname = argv[2];
            vector<String> filenames;
            cv::glob(fname, filenames);
            Image img = ImageLoader::readImage(filenames[0],maxgl,mingl,desiredgl,r,c);
            float* features = (float*) malloc(N *5* sizeof(float));
            float* d_out;
            std::string segment;
            for (size_t i=0; i<filenames.size(); i++)
            {
                
                img = ImageLoader::readImage(filenames[i],maxgl,mingl,desiredgl,r,c);
                d_out = glcm.GetSubGLCM(img,1,1,subImgDim);
                FeatureComputation::getFeatures(d_out,8,r,c,subImgDim,features);
                for (int j  = 0 ; j< N * 5 ; j++){
                    if(j==0)
                        fout<< filenames[i]<<",";
                    if (j!=0 and j%5==0)
                        {fout << "\n";
                        fout<< filenames[i]<<",";}
                         
                    fout << *(features+j);
                    if (j%5!=4)
                        fout<<",";
                    //cout<<  *(features + j)<<" ";
                }
                fout << "\n";
                
            }
            free(d_out);
            free(features);
            cout<<argv[1]<<"\n";
            return 0 ;

        }
        if (!strcmp(argv[1],"-i")){
            int ref_nb;
            int query_nb = 64*64;
            int dim = 5;
            int k = 16;
            float * knn_dist   = (float*) malloc(query_nb * k   * sizeof(float));
            int   * knn_index  = (int*)   malloc(query_nb * k   * sizeof(int));
            cudaFree(0);
            cout<<"Inferencing"<<"\n";
            fstream fin;
            fin.open("../ref_data/data.csv",ios::in);
            vector<float> ref_data;
            vector<float> ref_label;
            string line;
            int pos;
            while (getline(fin,line)){
                c=0;
                //ref_nb++;
                while((pos = line.find(',')) >= 0)
                {
                    string field = line.substr(0,pos);
                    line = line.substr(pos+1);
                    ref_data.push_back(stof(field));
                    
                    c++;
                    if (c>=5){
                        ref_label.push_back(stof(line));
                        break;
                    }
                }
            }
            ref_nb = ref_data.size()/dim;
            //cout<<"number of ref points "<<ref_nb<<endl;
            cout<<"size of data vector "<<ref_data.size()<<endl;
            printf("PARAMETERS\n");
            printf("- Number reference points : %d\n",   ref_nb);
            printf("- Number query points     : %d\n",   query_nb);
            printf("- Dimension of points     : %d\n",   dim);
            printf("- Number of neighbors     : %d\n\n", k);
            float * ref = ref_data.data();
            



            string fname = "../ref_data/ref_images/sample/01.bmp";
            Image img = ImageLoader::readImage(fname,maxgl,mingl,desiredgl,r,c);
            float* d_out = glcm.GetSubGLCM(img,1,1,subImgDim);
            float* features = (float*) malloc(N *5* sizeof(float));
            FeatureComputation::getFeatures(d_out,8,r,c,subImgDim,features);
             c=0;
            for(float i:ref_data)
            {
                c++;
                cout<< i << " ";
                if (c>=5){
                    cout<< "\n";
                    break;
                }
            }
          
            for (int i = 0 ; i < N*5  ; i ++){
                if (i%5==0 and i !=0)
                    {cout  <<"\n" ;} 
                
                
                cout<<  *(features + i)<<" ";
            }
            //test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_c,            "knn_c",              2);
            test(ref, ref_nb, features, query_nb, dim, k, knn_dist, knn_index, &knn_cuda_texture,  "knn_cuda_texture",  1); 
            free(d_out);
            free(features);
            return 0;
        }
    }  
    cout<<"Inferencing"<<"\n";
    int ref_nb;
    int query_nb = 64*64;
    int dim = 5;
    int k = 16;
    float * knn_dist   = (float*) malloc(query_nb * k   * sizeof(float));
    int   * knn_index  = (int*)   malloc(query_nb * k   * sizeof(int));
    cudaFree(0);
    auto start = high_resolution_clock::now();
    string fname = "../ref_data/ref_images/sample/01.bmp";

    Image img = ImageLoader::readImage(fname,maxgl,mingl,desiredgl,2048,2048);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
 
    cout << duration.count() <<" ms"<< endl;    

    //cout<<"From main r : "<< r << "\n";
    //cout<<"From main c : "<< c << "\n";
    //auto stop = high_resolution_clock::now();
    //auto duration = duration_cast<milliseconds>(stop - start);
 
    //cout << duration.count() << endl;
    //vector<int> pixels = img.getPixels();
    //vector<vector<uint8_t>> vector2d = img.get2dVector();

    //cout << "matrix size "+to_string(img.get_rows() * img.get_cols())+" vector size = "+to_string(pixels.size()) << endl;
 
    //Mat dst = Mat(r,c, CV_8UC1 ,&pixels,c * sizeof(uint8_t));
    
    //cout << "NEW MAT "+to_string(dst.rows * dst.cols) << endl;
    //cout << dst.total() << endl;
    //cout << dst.rows << endl;
    //cout << dst.cols << endl;
    //cout << dst.type() << endl;
    //cout << dst.depth() << endl;
    //cout << dst.channels() << endl;


    //used for testing
    
    /* uint zero= 0;uint one= 0;uint two= 0;uint three= 0;uint four= 0;uint five= 0;uint six= 0;uint seven= 0;uint eight = 0;
     
     for(int i = 0; i < pixels.size(); i++) {
        
        if (pixels[i]==0)
            zero+=1;
        else if (pixels[i]==1)
            one+=1;
        else if (pixels[i]==2)
            two+=1;
        else if (pixels[i]==3)
            three+=1;
        else if (pixels[i]==4)
            four+=1;
        else if (pixels[i]==5)
            five+=1;
        else if (pixels[i]==6)
            six+=1;
        else if (pixels[i]==7)
            seven+=1;
        else if (pixels[i]==8)
            eight+=1;
        else
            cout<<"OTHERSS!!"<<endl;
    } 
    std::cout << "0 : " + to_string(zero) << endl;
    std::cout << "1 : " + to_string(one) << endl;
    std::cout << "2 : " + to_string(two) << endl;
    std::cout << "3 : " + to_string(three) << endl;
    std::cout << "4 : " + to_string(four) << endl;
    std::cout << "5 : " + to_string(five) << endl;
    std::cout << "6 : " + to_string(six) << endl;
    std::cout << "7 : " + to_string(seven) << endl;  
    std::cout << "8 : " + to_string(eight) << endl; 
    std::cout << "Size of pixels vector : " + to_string(pixels.size()) << endl;  */
    
 
    
    float* d_out = glcm.GetSubGLCM(img,1,1,subImgDim); // host pointer.

  
  
   /*  for (int i = 0 ; i < 64*64*64*4  ; i ++){
        if (i%8==0)
            cout << "\n";
        if (i%256==0)
            cout << "\n";
        
        cout<<  *(d_out + i);
    }      */                       
 


 
    //cout <<"GLCMs computed"<< "\n";
    float* features = (float*) malloc(N *5* sizeof(float));
    FeatureComputation::getFeatures(d_out,8,r,c,subImgDim,features);

    for (int i = 0 ; i < 5  ; i ++){
        if (i%5==0 and i !=0)
            {cout  <<"\n" ;} 
        
        
        cout<<  *(features + i)<<" ";
    }        
    
    fstream fin;
    fin.open("../ref_data/data.csv",ios::in);
    vector<float> ref_data;
    vector<float> ref_label;
    string line;
    int pos;
    while (getline(fin,line)){
        c=0;
        //ref_nb++;
        while((pos = line.find(',')) >= 0)
        {
            string field = line.substr(0,pos);
            line = line.substr(pos+1);
            ref_data.push_back(stof(field));
            
            c++;
            if (c>=5){
                ref_label.push_back(stof(line));
                break;
            }
        }
    }
    ref_nb = ref_data.size()/dim;
    //cout<<"number of ref points "<<ref_nb<<endl;
    cout<<"size of data vector "<<ref_data.size()<<endl;
    printf("PARAMETERS\n");
    printf("- Number reference points : %d\n",   ref_nb);
    printf("- Number query points     : %d\n",   query_nb);
    printf("- Dimension of points     : %d\n",   dim);
    printf("- Number of neighbors     : %d\n\n", k);
    float * ref = ref_data.data();
    c=0;
    for(float i:ref_data)
    {
        c++;
        cout<< i << " ";
        if (c>=5){
            cout<< "\n";
            break;
        }
    }
    //test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_c,            "knn_c",              2);
    test(ref, ref_nb, features, query_nb, dim, k, knn_dist, knn_index, &knn_cuda_texture,  "knn_cuda_texture",  1); 
    free(d_out);
    free(features);                 
//end testing

/*
    Implement

    For training iterate over a images and proceed

    For usage get images using camera
 */
    





}
