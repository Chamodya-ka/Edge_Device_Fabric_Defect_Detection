import cv2
import glob
import os

def split(fname,split_size=32):
    img = cv2.imread(fname)
    img = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(2048,2048))
    print(img.shape)
    for i in range(0,img.shape[0],split_size):
        for j in range(0,img.shape[1],split_size):
            crop = img[j:j+split_size,i:i+split_size]
            print(crop.shape)
            print(crop)
            cv2.imshow("s",crop)
            cv2.waitKey(0)
            break
        break

folder = "../testimg"
try:
	os.mkdir(folder+"/resized")
except:
	pass
for filepath in glob.iglob(folder+"/*.png"):
	print(filepath)
	x = filepath.split("/")
	split(filepath)