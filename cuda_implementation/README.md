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
