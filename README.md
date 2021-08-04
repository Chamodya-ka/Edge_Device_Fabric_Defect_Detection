# Edge_Device_Fabric_Defect_Detection

## Description

High detailed images captured at 5 FPS for fabric defect detection in the FabVis project is around 14MB. Hence it requires a transmission rate of ~ 72MBps per camera used. These images are sent to a backend computer where the defect is localized, defect type is classified, etc.

To lower the transmission rate required the proposed solution is to add an edge device to filter out only the defective images. Hence, significantly lowering the transmission rate.
A jetson TX2 board will be used as the edge device.
