import cv2
import glob
import os

def resize(fname,dest):
	img = cv2.imread(fname)
	img = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(2048,2048))
	print(dest)
	cv2.imwrite(dest,img)

folder = "../testimg"
try:
	os.mkdir(folder+"/resized")
except:
	pass
for filepath in glob.iglob(folder+"/*.png"):
	print(filepath)
	x = filepath.split("/")
	resize(filepath,folder+"/resized/"+x[-1])


