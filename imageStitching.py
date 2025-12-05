import cv2 as cv
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113


def parse_command_line_arguments():# Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--kp", default="GFTT", help="key point (or corner) detector: GFTT ORB SIFT ")
    parser.add_argument("-n", "--nbKp", default=1000, type=int, help="Number of key point required (if configurable) ")
    parser.add_argument("-d", "--descriptor", default=True, type=bool, help="compute descriptor associated with detector (if available)")
    parser.add_argument("-m", "--matching", default="NORM_L1", help="Brute Force norm: NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2")
    parser.add_argument("-i1", "--image1", default="IMG_1_reduced.jpg", help="path to image1")
    parser.add_argument("-i2", "--image2", default="IMG_2_reduced.jpg", help="path to image2")
    parser.add_argument("-t", "--threshold", default=3, type=float, help="Ratio threshold for Lowe's ratio test")
    # other argument may need to be added
    return parser

def test_load_image(img):
    if img is None or img.size == 0 or (img.shape[0] == 0) or (img.shape[1] == 0):
        print("Could not load image !")
        print("Exiting now...")
        exit(1)

def load_gray_image(path):
    if(path != None):
        img = cv.imread(path)
        test_load_image(img)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    else:
        img = None
        gray = None
    return img, gray

def display_image(img, image_window_name):
    cv.namedWindow(image_window_name)
    cv.imshow(image_window_name, img)

def feature_detector(type, gray, nb):
    if gray is not None :
        match type :
            case "GFTT":
                gftt = cv.GFTTDetector_create(nb)
                kp = gftt.detect(gray, None)  
            case "ORB":
                orb = cv.ORB_create(nb)
                kp = orb.detect(gray, None)
            case "SIFT":
                sift = cv.SIFT_create(nb)
                kp=sift.detect(gray, None)
    else:
        kp =  None
    return kp

def feature_extractor(type, img, kp):
    desc = None
    if img is not None and kp is not None:
        match type:
            case "GFTT":
                detector = cv.ORB_create()
                kp, desc = detector.compute(img, kp)
            case "ORB":
                detector = cv.ORB_create()
                kp, desc = detector.compute(img, kp)
            case "SIFT":
                detector = cv.SIFT_create()
                kp, desc = detector.compute(img, kp)
    return desc

def feature_matching(matching, desc1, desc2, threshold):
    match matching:
        case "NORM_L1":
            bf = cv.BFMatcher().create(normType=cv.NORM_L1)

        case "NORM_L2":
            bf = cv.BFMatcher().create(normType=cv.NORM_L2)

        case "NORM_HAMMING":
            bf = cv.BFMatcher().create(normType=cv.NORM_HAMMING)

        case "NORM_HAMMING2":
            bf = cv.BFMatcher().create(normType=cv.NORM_HAMMING2)

    matches = bf.knnMatch(desc1,desc2,k=2)

    good = []
    for m,n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])

    print(good)

    return good

def main():

    parser = parse_command_line_arguments()
    args = vars(parser.parse_args())

    # Load, transform to gray the 2 input images
    print("load image 1")
    img1, gray1 = load_gray_image(args["image1"])
    print("load image 2")
    img2, gray2 = load_gray_image(args["image2"])

    # displays the 2 input images
    if img1 is not None : display_image(img1, "Image 1")
    if img2 is not None : display_image(img2, "Image 2")

    # Apply the choosen feature detector
    print(args["kp"]+" detector")
    
    kp1 = feature_detector(args["kp"], gray1, args["nbKp"])
    if img2 is not None: kp2 = feature_detector(args["kp"], gray2, args["nbKp"])

    # Display the keyPoint on the input images
    img_kp1=cv.drawKeypoints(gray1,kp1,img1)
    if img2 is not None: img_kp2=cv.drawKeypoints(gray2,kp2,img2)
    
    display_image(img_kp1, "Image 1 "+args["kp"])
    if img2 is not None : display_image(img_kp2, "Image 2 "+args["kp"])

    desc1 = feature_extractor(args["kp"], gray1, kp1)
    if img2 is not None: desc2 = feature_extractor(args["kp"], gray2, kp2)

    good = feature_matching(args["matching"], desc1, desc2, args["threshold"])

    # code to complete (using functions):
    # - to extract feature and compute descriptor with ORB and SIFT 
    # - to calculate brute force matching between descriptor using different norms
    # - to calculate and apply homography 
    # - to stich and display resulting image

    # waiting for user action
    key = 0
    while key != ESC_KEY and key!= Q_KEY :
        key = cv.waitKey(1)

    # Destroying all OpenCV windows
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()