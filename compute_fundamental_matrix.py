import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from compute_correspondences import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
'''
def lls_eight_point_alg(points1, points2):
    # Algorithm: 1) Setup homogeneous system, 2) Compute SVD and F as last right singular vector, 3) Enforce rank 2 on F
    W = np.zeros((points1.shape[0],9), points1.dtype)
    for i in range(points1.shape[0]):
        W[i,:] = np.array([points1[i,0]*points2[i,0], points1[i,0]*points2[i,1], points2[i,0], points1[i,1]*points2[i,0],
                           points1[i,1]*points2[i,1], points2[i,1], points1[i,0], points1[i,1], 1])
    # Compute Matrix F from SVD
    U,S,Vt = np.linalg.svd(W)
    f = Vt[-1,:]
    rank3F = np.reshape(f, (3,3))  # np.array([f[0:3],f[3:6],f[6:9]])

    # Enforce rank 2 on F
    U2,S2,V2t = np.linalg.svd(rank3F)
    F = U2[:,:2].dot(np.diag(S2[:2])).dot(V2t[:2,:])
    return F

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    #Algorithm: 1) Find centroid, t_c, of points (mean in x & y), 2) Find scale factor, s, to normalize based on RMS distance
    # from centroid, 3) Setup matrix T from s & t_c, 4) Transform points1 and point2, 5) Compute F, 6) Un-normalize F
    # Compute Translation and Scale Matrices
    # Find centroid, t_c, of points (mean in x & y)
    centroid1 = np.mean(points1,0)
    centroid2 = np.mean(points2,0)

    # Find scale factor, s, to normalize based on RMS distance from centroid
    distance_points1 = np.sqrt(np.sum(np.power((points1-centroid1),2),1))
    mean_sqrd_distance1 = np.mean(np.power(distance_points1,2))
    scale1 = np.sqrt(2/mean_sqrd_distance1)

    distance_points2 = np.sqrt(np.sum(np.power((points2 - centroid2),2),1))
    mean_sqrd_distance2 = np.mean(np.power(distance_points2,2))
    scale2 = np.sqrt(2/mean_sqrd_distance2)

    # Setup matrix T from s & t_c
    T = np.array([[1, 0, -centroid1[0]],[0, 1, -centroid1[1]], [0, 0, 1]])
    S = np.array([[scale1, 0, 0], [0, scale1, 0], [0, 0, 1]])
    T1 = S.dot(T)

    T = np.array([[1, 0, -centroid2[0]],[0, 1, -centroid2[1]], [0, 0, 1]])
    S = np.array([[scale2, 0, 0], [0, scale2, 0], [0, 0, 1]])
    T2 = S.dot(T)

    # Normalize Points (Transform by Ts)
    points1_normalized = T1.dot(points1.transpose()).transpose()
    points2_normalized = T2.dot(points2.transpose()).transpose()

    # Compute F from normalized points
    F_normalized = lls_eight_point_alg(points1_normalized, points2_normalized)

    # Un-normalize F
    F = T2.transpose().dot(F_normalized).dot(T1)

    return F

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image
    im2 - a HxW(xC) matrix that contains pixel values from the second image
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    # Algorithm: 1) show images, 2) loop through the set of points, 3) compute epipolar lines for 1 & 2,
    # 4) Plot range of points on the line around the imaged point
    fig1 = plt.figure(); sp1 = fig1.add_subplot(111); sp1.imshow(im1,cmap='gray')
    fig2 = plt.figure(); sp2 = fig2.add_subplot(111); sp2.imshow(im2,cmap='gray')

    for i in range(points1.shape[0]):
        # compute epipolar line with l=F_transpose*p equation being for line in im2 corresponding to point in im1
        l1 = F.transpose().dot(points2[i,:])
        l2 = F.dot(points1[i,:])

        # overlay epipolar lines in a range of points around the image point
        xrange1 = np.arange(points1[i,0]-30, points1[i,0]+30)
        xrange2 = np.arange(points2[i,0]-30, points2[i,0]+30)
        sp1.plot(xrange1, (-l1[0]*xrange1-l1[2])/l1[1], 'r')
        sp2.plot(xrange2, (-l2[0]*xrange2-l2[2])/l2[1], 'r')

        # overlay points
        sp1.plot(points1[i,0], points1[i,1],'b.')
        sp2.plot(points2[i,0], points2[i,1],'b.')

    plt.show()

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set of
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # Algorithm: 1) loop over all points in the two images, 2) compute epipolar line in im1 associated with point in im2,
    # 3) compute distance between point in im1 with its corresponding epipolar line computed, 4) sum & average
    sum_distance = 0
    for i in range(points1.shape[0]):
        l1 = F.dot(points2[i,:])
        sum_distance += abs(l1.dot(points1[i,:]))/np.sqrt(np.power(l1[0],2) + np.power(l1[1],2))

    return sum_distance/points1.shape[0]

if __name__ == '__main__':
    # Read in the data
    im1 = cv2.imread('images/IMG_0081.jpg')
    im2 = cv2.imread('images/IMG_0082.jpg')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGBA)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGBA)

    kp1, des1, kp2, des2 = compute_correspondence(im1, im2, 0)
    points1, points2 = match_correspondence(im1, im2, kp1, des1, kp2, des2, 0)
    assert (points1.shape == points2.shape)

    # Homogenize points
    points1 = np.hstack((points1, np.ones((points1.shape[0],1))))
    points2 = np.hstack((points2, np.ones((points2.shape[0],1))))

    # Running the linear least squares eight point algorithm
    F_lls = lls_eight_point_alg(points1, points2)
    np.set_printoptions(suppress=True)
    print "Fundamental Matrix from LLS  8-point algorithm:\n", F_lls

    print "Distance to lines in image 1 for LLS:", \
        compute_distance_to_epipolar_lines(points1, points2, F_lls)
    print "Distance to lines in image 2 for LLS:", \
        compute_distance_to_epipolar_lines(points2, points1, F_lls.T)

    # Running the normalized eight point algorithm
    F_normalized = normalized_eight_point_alg(points1, points2)

    pFp = [points2[i].dot(F_normalized.dot(points1[i]))
           for i in xrange(points1.shape[0])]

    print "p'^T F p =", np.abs(pFp).max()
    print "Fundamental Matrix from normalized 8-point algorithm:\n", \
        F_normalized
    print "Distance to lines in image 1 for normalized:", \
        compute_distance_to_epipolar_lines(points1, points2, F_normalized)
    print "Distance to lines in image 2 for normalized:", \
        compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)

    # Plotting the epipolar lines
    plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
    plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

    plt.show()
