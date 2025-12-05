import cv2 as cv
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113


def parse_command_line_arguments():# Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--kp", default="SIFT", help="key point (or corner) detector: GFTT ORB SIFT BRIEF")
    parser.add_argument("-n", "--nbKp", default=100, type=int, help="Number of key point required (if configurable) ")
    parser.add_argument("-e", "--extractor", default=None, help="descriptor extractor (if None, uses detector's descriptor)")
    parser.add_argument("-d", "--descriptor", default=True, type=bool, help="compute descriptor associated with detector (if available)")
    parser.add_argument("-m", "--matching", default="NORM_HAMMING2", help="Brute Force norm: NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2")
    parser.add_argument("-i1", "--image1", default="IMG_1_reduced.jpg", help="path to image1")
    parser.add_argument("-i2", "--image2", default="IMG_2_reduced.jpg", help="path to image2")
    parser.add_argument("-a", "--alpha", default=10, type=float, help="Multiplicative coefficient for distance threshold (alpha * distMin)")
    parser.add_argument("-min", "--minMatch", default=10, type=int, help="Min match count for homography")

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

def display_matching(img1, kp1, img2, kp2, good):
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    display_image(img3, "Matching")

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
                kp = sift.detect(gray, None)
            case "BRIEF":
                star = cv.xfeatures2d.StarDetector_create()
                kp = star.detect(gray, None)
                if len(kp) > nb:
                    kp = sorted(kp, key=lambda x: -x.response)[:nb]
    else:
        kp =  None
    return kp

def feature_extractor(detector_type, img, kp, extractor_type=None):
    desc = None
    if img is not None and kp is not None:
        match detector_type:
            case "GFTT":
                detector = cv.ORB_create()
                kp, desc = detector.compute(img, kp)
            case "ORB":
                detector = cv.ORB_create()
                kp, desc = detector.compute(img, kp)
            case "SIFT":
                detector = cv.SIFT_create()
                kp, desc = detector.compute(img, kp)
            case "BRIEF":
                brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
                kp, desc = brief.compute(img, kp)
    return kp, desc

def feature_matching(matching, desc1, desc2, alpha):
    match matching:
        case "NORM_L1":
            bf = cv.BFMatcher().create(normType=cv.NORM_L1)

        case "NORM_L2":
            bf = cv.BFMatcher().create(normType=cv.NORM_L2)

        case "NORM_HAMMING":
            bf = cv.BFMatcher().create(normType=cv.NORM_HAMMING)

        case "NORM_HAMMING2":
            bf = cv.BFMatcher().create(normType=cv.NORM_HAMMING2)

    matches = bf.match(desc1, desc2)
    
    if len(matches) > 0:
        dist_min = min([m.distance for m in matches])
        threshold = alpha * dist_min
        good = [[m] for m in matches if m.distance <= threshold]
    else:
        good = []
    
    return good

def homography(min_match, good, kp1, kp2, img1, img2):
    if len(good) > min_match:
        matches_flat = [m[0] for m in good]
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches_flat]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches_flat]).reshape(-1,1,2)
        H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape[:2]
        pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts, H)
        img2_with_box = cv.polylines(img2.copy(), [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        
        draw_params = dict(matchColor = (0,255,0), 
                       singlePointColor = None,
                       matchesMask = matchesMask, 
                       flags = 2)
        
        img3 = cv.drawMatches(img1, kp1, img2_with_box, kp2, matches_flat, None, **draw_params)
        display_image(img3, "Homography Matches")
        return H
    else:
        print("Not enough matches are found - {}/{}".format(len(good), min_match))
        return None
    
def finalImageStitching(img1, img2, H):
    if H is None:
        print("Pas de matrice d'homographie")
        return None
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    pts2_transformed = cv.perspectiveTransform(pts2, H)
    
    all_pts = np.concatenate((pts1, pts2_transformed), axis=0)
    [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    
    translation = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation[0]], 
                               [0, 1, translation[1]], 
                               [0, 0, 1]], dtype=np.float64)
    
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    
    output_img = cv.warpPerspective(img2, H_translation.dot(H), 
                                     (canvas_width, canvas_height))
    
    y1, y2 = translation[1], translation[1] + h1
    x1, x2 = translation[0], translation[0] + w1
    
    mask = output_img[y1:y2, x1:x2] != 0
    output_img[y1:y2, x1:x2][~mask] = img1[~mask]
    
    display_image(output_img, 'Stitched Result')
    return output_img


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

    kp1, desc1 = feature_extractor(args["kp"], gray1, kp1, args["extractor"])
    if img2 is not None: kp2, desc2 = feature_extractor(args["kp"], gray2, kp2, args["extractor"])

    good = feature_matching(args["matching"], desc1, desc2, args["alpha"])
    display_matching(img1, kp1, img2, kp2, good)

    H = homography(args["minMatch"], good, kp1, kp2, gray1, gray2)
    finalImageStitching(img1, img2, H)

    # waiting for user action
    key = 0
    while key != ESC_KEY and key!= Q_KEY :
        key = cv.waitKey(1)

    # Destroying all OpenCV windows
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()