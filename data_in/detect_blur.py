# import the necessary packages
from imutils import paths
import argparse
import cv2
import os
def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=16.9,
                help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

# loop over the input images 
n = 0
for imagePath in paths.list_images(args["images"]):
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
#    text = "Sharp"
    text = "0"
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < args["threshold"]:
#        text = "Blurry"
        text = "1"
    filename = imagePath.split("/")[-1]
    command = "cp "+imagePath+" /Users/wangmingjian/All_code/python/YAYA/data_in/pic/picTrain/"+text+"."+filename
    os.system(command)
#    os.system("cp "+imagePath+" /Users/wangmingjian/All_code/python/YAYA/data_in/pic/pic_with_label_train/"+text+"_"+filename)
    print("Processed "+str(n)+"/777 pic "+filename+" "+"\n\t"+text)
    if n<32:
        n += 1
    else:
        break



'''
    # show the image
    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    key = cv2.waitKey(0)
#'''
