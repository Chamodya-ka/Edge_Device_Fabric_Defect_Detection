# Edge_Device_Fabric_Defect_Detection

## Description

High detailed images captured at 5 FPS for fabric defect detection in the FabVis project is around 14MB. Hence it requires a transmission rate of ~ 72MBps per camera used. These images are sent to a backend computer where the defect is localized, defect type is classified, etc.

6 cameras are required to capture the entire width of the fabric (1.2m) this amounts for ~ 480MBps transmission rate. 
To lower the transmission rate required the proposed solution is to add an edge device to filter out only the defective images. Hence, significantly lowering the transmission rate. A jetson TX2 board will be used as the edge device.


![prob and sol](/images/ProbNSol.jpg)

The edge device performs an efficient defect detection and classifies the input images as defective or non defective.

![edge](/images/Edge.jpg)
