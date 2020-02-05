# import the necessary packages
from skimage.measure import compare_ssim
import imutils
import cv2
import argparse
import os
import matplotlib.pyplot as plt


# settings
input_dir = os.path.join(os.getcwd(),"input")
output_dir = os.path.join(os.getcwd(),"output")

img_width = 600
img_height = 400

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-first', type=str,help='first image to compare')
    parser.add_argument('-second', type=str,help="second image to compare ")
    parser.add_argument('-output', type=str,help="output directory",default=output_dir)

    args = parser.parse_args()

    return args

def filter_contours(cnts,limit_of_pixels):
    # compute the bounding box of the contour and then draw the
    # bounding box on both input input to represent where the two input differ
    countours = []
    for c in cnts:
        box = cv2.boundingRect(c)
        if box[2] * box[3] > limit_of_pixels:  # if bounding box is small, dont't append
            countours.append(box)
    return countours

def append_boxes_to_image(image,countours):
    #append contours to input
    for frame in countours:
        x = frame[0]
        y = frame[1]
        w = frame[2]
        h = frame[3]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)


if __name__ =="__main__":

    # parse arguments
    args = parser()
    imageA_path = args.first
    imageB_path = args.second
    output_dir = args.output

    # load the two input input from argparse
    imageA = cv2.imread(imageA_path)
    imageB = cv2.imread(imageB_path)

    #resize image
    imageA = cv2.resize(imageA,(img_width,img_height) ,cv2.INTER_AREA)
    imageB = cv2.resize(imageB,(img_width,img_height),cv2.INTER_AREA)

    # convert the input to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # input, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input input that differ
    thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_TOZERO_INV | cv2.THRESH_OTSU)[1]

    #find contours from treshold
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    filtered_countours = filter_contours(cnts,200)
    append_boxes_to_image(imageB,filtered_countours)

    # save input
    cv2.imwrite(os.path.join(output_dir,"input.png"),imageA)
    cv2.imwrite(os.path.join(output_dir,"output.png"), imageB)

    # save debug
    plt.figure()

    plt.subplot(2,2,1)
    plt.axis("off")
    plt.title("First image")
    plt.imshow(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)) #"Original"

    plt.subplot(2,2,2)
    plt.axis("off")
    plt.title("Second image")
    plt.imshow(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)) #"Modiied",

    plt.subplot(2,2,3)
    plt.axis("off")
    plt.title("Diff")
    plt.imshow(diff,cmap="gray") # "Diff",

    plt.subplot(2,2,4)
    plt.axis("off")
    plt.imshow(thresh,cmap="gray") # Tresh
    plt.title("Threshhold")

    plt.savefig(os.path.join(output_dir,"debug.png"))

    print("Process was done ")