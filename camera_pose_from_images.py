import numpy as np
import cv2
from compute_camera_pose import *

def main():
    im1 = cv2.imread('images/IMG_0081.jpg')
    im2 = cv2.imread('images/IMG_0082.jpg')

    # Compute Features, Find Correspondence & Refine Matches
    kp1, des1, kp2, des2 = compute_correspondence(im1, im2, 1)
    points1, points2 = match_correspondence(im1, im2, kp1, des1, kp2, des2, 1)

    points = np.hstack((points1, points2)).reshape(-1,2,2)

    # Homogenize points
    points1h = np.hstack((points1, np.ones((points1.shape[0],1))))
    points2h = np.hstack((points2, np.ones((points2.shape[0],1))))

    # Compute Fundamental Matrix
    F = normalized_eight_point_alg(points1h, points2h)

    # Form Intrinsic Matrix from Offline Calibration
    # Compute Essential Matrix from Fundamental and Intrinsic Matrices
    focal_length = np.array([1776, 1780])
    principal_point = np.array([762, 1025])
    K = np.array([[focal_length[0], 0, principal_point[0]],[0, focal_length[1], principal_point[1]],[0,0,1]])

    # Compute Essential Matrix
    E = K.T.dot(F).dot(K)

    # Determine the Camera Pose (correct R, T) from Essential Matrix
    estimated_RT = estimate_RT_from_E(E, points, K)

if __name__ == '__main__':
    main()
