## Read image from file and Preprocessing

Using ImageLoader images are loaded as grayscaled images of depth : ```CV_8UC1``` using opencv. The image is quantized and converted to a uint vector of pixels. An Image object is returned containing the quantized pixels, number of rows, number of columns and gray level

#####  ```Image img = ImageLoader::readImage(fname,maxgl,mingl,desiredgl);```

##### params 

```fname``` - String filename

```maxgl``` - Maximum intensity in the given image ```DEFAULT = 255```

```mingl``` - Minimum intensity in the given image ```DEFAULT = 0```

```desiredgl``` - How many gray levels required  ```DEFAULT = 7``` ie: 8 graylevels including 0 (Check if required is excluding 0)
##### returns
```Image``` - Object  
###### Input image
<img src="https://github.com/Chamodya-ka/Edge_Device_Fabric_Defect_Detection/blob/main/cuda_implementation/testimg/gradient.png" width="200">

###### Quantized image
<img src="https://github.com/Chamodya-ka/Edge_Device_Fabric_Defect_Detection/blob/main/cuda_implementation/testimg/gradientresult.png" width="200">

These pixel values are stored in an vector of the Image object.

#### (After Mid)
Quantization is done parallelly.

## Computing GLCM

<img src="https://github.com/Chamodya-ka/Edge_Device_Fabric_Defect_Detection/blob/main/cuda_implementation/images/2D_Representation.jpg" width="600">

The 1D image vector obtained from the ImageLoader is fed into the GLCMComputation object. GLCMComputation::GetSubGLCM(params) returns the GLCMS calculated for 32x32 pixels^2 sub windows and it is used for demonstration. 4 Co-occurence matrices are calculated for ```theta=0``` ```theta=45``` ```theta=90``` ```theta=135```.Hence the resulting array would contain 64x64x4 sub GLCMs. These sub GLCMs will be used to calculate the Haralick features to obtain a feature vector.

#####  ```int* out = GLCMComputation::GetSubGLCM(img,d,angle);```
Kernel used to calculate sub GLCMs will be extended to calculate the features. Hence this function will be only used for testing and demonstration. (96-130 ms)

##### params
```img``` - Image object

```d``` , ```angle``` -  Currently not used.

##### returns
```out``` - pointer to the array containing the 64x64x4 sub GLCMS  

## Computing Haralick Texture Features from sub-GLCMs

<img src="https://github.com/Chamodya-ka/Edge_Device_Fabric_Defect_Detection/blob/main/cuda_implementation/images/FeatureCalculation.jpg" width="600">

This is an extension to the above mentioned function. The kernel used to calculate the sub GLCMs was modified to calculate the features. The sub GLCMs array is not required hence it will be discarded later (Not yet discarded for testing purposed). The produced 4 sub GLCMs from 64x64 blocks of the image is passed onto 5 device kernels to calculate the 5 texture features. Each block uses a ```__shared__ float[5]``` to store the feature vectors. This shared memory array is copied on to a ```float [64 x 64 x 5]``` array in global memory.

#####  ```int* out = GLCMComputation::GetSubGLCM(img,d,angle);``` (Needs name change)

##### params
```img``` - Image object

```d``` , ```angle``` -  Currently not used.

##### returns
```out``` - pointer to the array containing the 64x64x5 feature vectors // 1 line change not yet implemented

#### (Modifications)
A sperate set of kernels computes the 5 Haralick features and 4 kernels to calculate the means and standard deviations. (in NewFeatureCalculation.cu). This was split up due to exceeding resources available on the TX2. Moreover, further optimizations such as using parallel reductions and single precision intrinsic functions are utilized. (on average <10ms for calculation of features).


## Feedback received from Mid-Evaluation
It was suggested that I start optimizing the sequentially running part of the program. The recorded time (30ms) was the total time GPU computes the GLCM and its features however profiling suggests minimal GPU utilization. 
Therefore, I have minimized loops outside parallelized part of the algorithm. OpenCV image reads and grayscales consume considerable amount of time however.

## Results Achieved 

- Image size = 2048 x 2048
- Tile size  = 32 x 32
- Note : times before computed for 16 gray levels and times after are for 8 gray levels
         
| Component/Calculation | Times before | Times after parallelization |
|---|---|---|
|Quantization | 7 ms | 1.4ms|
|Extracting GLCMs| 180ms| -|
|Normalize GLCM| 51ms| -|
|Flatten GLCM|1.8us|13ms|
|ENERGY feature| 46ms| 412us|
|CONTRAST feature| 55.8ms| 764.7us|
|HOMOGEINITY feature| 53.5ms| 1.46ms|
|CORRELATION feature| 149ms| 1.31ms|
|Means and Stdd | 246ms| 2.53ms|
|ENTROPY feature| 504ms| 512us|


## Challenges

#### GLCM calculations
Time spent by blocked threads during synchronized operations are significant. The trial implementation done last week which computed a **single** GLCM on the global memory for a 1250x1250 image requred ```~0.96ms```. This is due to high number of frequently occuring data patters written to a small space (8x8 space). 
To overcome this issue in computing the sub GLCMs, each sub grid of the image that was processed by a block of threads was given an 8x8x4 grid in the shared memory for the 4 resultant GLCMs. By doing this the global memory access were reduced as well.

12.09
#### Limitations Using Atomic Functions
When threads update a single memory location Atomic functions are used to avoid race conditions because of this threads spend majority of their time blocked.
In the feature calculation kernels, the concept of parellel reductions were used maximize performance. (Need to check if this can be adopted in the kernel calculating SUB GLCMs as well)
#### Division by 0
Correlation feature calculation includes a step to divide by the σ^2 (standard deviation of the sub GLCM squared). For cases such as a monotone images, the GLCM will consist of mostly 0s. Actual reason not yet found, hence for now small value is added ```pow(stdd[(int) floorf(id/(gl*gl))] + 0.0000001,2); ```

30.09
#### Requested too many resources on Jetson-TX2
The existing implmentation that is an extension to the GLCM calculating kernel will reduce the memory copy overheads (device to host to device not needed). Due to this reason the kernel becomes very long and resource extensive. Upon testing on the Jetson it threw a "Too many resources requested expection". To work around this issue I tried spliting the single kernel to two kernels at the expense of 2 extra data transfers between host and device. It did work as intended but it seems to take more time (significanlt time consumed by the feature extraction kernel).
#### Too much time consumption by feature extraction kernel
The feature extraction kernel was implemented to compute each feature serially. I thought that this could be the issue, hence implemented them to run concurrently  using streams. Results should some improvement but not sufficient. I will be looking at ways to optimize the feature extraction part before the mid-evaluations, and work on the ML part afterwards.
#### RISK : a graylevel of 8 might not be sufficient.
This decision was based on observation. The resulting GLCMs for (32x32 pixels of a real example) only consisted mainly of 2 quantized gray levels (like 6 and 7) this might cause for inaccuracy. I will make alterations to increase the number of gray levels to 16 or 32. Upon checking final results this can be decided on.
#### Misleading exception
Without proper degubbing tools, the CUDA runtime throws out the last expection that occurs. Due to this reason I spent almost 2 days to fix a bug caused by an indexing error. Using ```cud-memcheck``` any out of bounds exceptions can be located.

14.20 (Before mid)
#### Scope reduced
Scope of the project was reduced to calculating GLCMs and extracting their features. Further a kNN based classifier can be used to attempt to classify defective images.
