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

## Computing GLCM

<img src="https://github.com/Chamodya-ka/Edge_Device_Fabric_Defect_Detection/blob/main/cuda_implementation/images/2D_Representation.jpg" width="600">

The 1D image vector obtained from the ImageLoader is fed into the GLCMComputation object. GLCMComputation::GetSubGLCM(params) returns the GLCMS calculated for 32x32 pixels^2 sub windows and it is used for demonstration. 4 Co-occurence matrices are calculated for ```theta=0``` ```theta=45``` ```theta=90``` ```theta=135```.Hence the resulting array would contain 64x64x4 sub GLCMs. These sub GLCMs will be used to calculate the Haralick features to obtain a feature vector.

#####  ```int* out = GLCMComputation::GetSubGLCM(img,d,angle);```
Kernel used to calculate sub GLCMs will be extended to calculate the features. Hence this function will be only used for testing and demonstration.

##### params
```img``` - Image object

```d``` , ```angle``` -  Currently not used.

##### returns
```out``` - pointer to the array containing the 64x64x4 sub GLCMS   

#### Challenges
##### GLCM calculations
Time spent by blocked threads during synchronized operations are significant. The trial implementation done last week which computed a **single** GLCM on the global memory for a 1250x1250 image requred ```~0.96ms```. This is due to high number of frequently occuring data patters written to a small space (8x8 space). 
To overcome this issue in computing the sub GLCMs, each sub grid of the image that was processed by a block of threads was given an 8x8x4 grid in the shared memory for the 4 resultant GLCMs. By doing this the global memory access were reduced as well.


