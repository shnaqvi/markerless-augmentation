import numpy as np
from scipy.misc import *
import matplotlib.pyplot as plt
import cv2
import time
from itertools import compress
from utils.utils import *

def main():
    kp1, des1, kp2, des2 = compute_correspondence(None, None, 0)
    src_pts, dst_pts = match_correspondence(None, None, kp1, des1, kp2, des2, 1)

def compute_correspondence(im1, im2, debug):
    if im1 is None or im2 is None:
        im1 = cv2.imread('images/palace.jpg')
        im2 = cv2.imread('images/palace1.jpg')

    # Get correspondence pairs from ORB detector & SIFT descriptor
    orb = cv2.ORB_create()
    # orb = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)

    if debug:
        im1 = cv2.drawKeypoints(im1, kp1, None, color=(0,0,255), flags=0)
        im2 = cv2.drawKeypoints(im2, kp2, None, color=(0,0,255), flags=0)

        cvDrawWindow('im1', im1, .3, 1)
        cvDrawWindow('im2', im2, .3, 1)

    return kp1, des1, kp2, des2

def match_correspondence(im1, im2, kp1, des1, kp2, des2, debug):
    if im1 is None or im2 is None:
        im1 = cv2.imread('images/palace.jpg')
        im2 = cv2.imread('images/palace1.jpg')

    # Time for performance
    start = time.time()

    # Match Correspondence using Brute-Force Matcher (closest points in distance metric)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)   #cv2.NORM_HAMMING
    matches = bf.knnMatch(des1, des2, k=2)

    # using FLANN Matches (optimized for large datasets and high-dim feature vector)
    # FLANN_INDEX_LSH = 0
    # index_params= dict(algorithm = FLANN_INDEX_LSH,
    #                    table_number = 12,
    #                    key_size = 20,
    #                    multi_probe_level = 2)
    # search_params = dict(checks=50)
    #
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1.astype(np.float32),des2.astype(np.float32),k=2)

    # Store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # Get the Matches and the homography via RANSAC
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    # Draw polygon lines when query image will be transformed to
    h,w,c = im1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(im2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    im3 = cv2.drawMatches(im1,kp1,im2,kp2,good,None,**draw_params)

    # List of Inlier Matches (Points in Source and Destination Images)
    good_inliers =  list(compress(good, matchesMask)) #np.array(good)[np.where(matchesMask)[0]]
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_inliers ]).reshape(-1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_inliers ]).reshape(-1,2)

    end = time.time()

    if debug:
        print "time elapsed: %f" %(end-start)

        cvDrawWindow('im3', im3, .2, 1)

    return src_pts, dst_pts

if __name__ == '__main__':
    main()
