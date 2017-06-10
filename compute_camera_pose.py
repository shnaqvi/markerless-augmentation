import sys
import numpy as np
import os
import scipy.misc
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from compute_fundamental_matrix import *

def main():
    # Load the data
    im1 = cv2.imread('images/palace.jpg')
    im2 = cv2.imread('images/palace1.jpg')

    kp1, des1, kp2, des2 = compute_correspondence(im1, im2, 0)
    points1, points2 = match_correspondence(im1, im2, kp1, des1, kp2, des2, 0)

    points = np.hstack((points1, points2)).reshape(-1,2,2)

    # Homogenize points
    points1h = np.hstack((points1, np.ones((points1.shape[0],1))))
    points2h = np.hstack((points2, np.ones((points2.shape[0],1))))

    F = normalized_eight_point_alg(points1h, points2h)

    # Store points and matrices for all frames
    points_frames = None
    F_frames = None
    if (points_frames is None):
        points_frames = np.vstack((points1.T, points2.T))
        F_frames = F
    else:
        points_frames = np.stack((points_frames,np.vstack((points1.T, points2.T))), axis=0)
        F_frames = np.stack(F_frames, axis=0)

    # Compute Essential Matrix from Fundamental and Intrinsic Matrices
    focal_length = np.array([1776, 1780])
    principal_point = np.array([762, 1025])
    K = np.array([[focal_length[0], 0, principal_point[0]],[0, focal_length[1], principal_point[1]],[0,0,1]])
    E = K.T.dot(F).dot(K)

    np.set_printoptions(suppress=True, precision=4)

    # Compute the 4 initial R,T transformations from Essential Matrix
    estimated_RT = estimate_initial_RT(E)
    print "Estimated RT:\n", estimated_RT

    # Determine the best linear estimate of a 3D point
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(estimated_RT[0,:,:])
    estimated_3d_point = linear_estimate_3d_point(points[0,:,:], camera_matrices.copy())
    print "Sample Estimated 3D Point: ", estimated_3d_point

    # Calculate the reprojection error and its Jacobian
    estimated_error = reprojection_error(
            estimated_3d_point, points[0,:,:], camera_matrices)
    estimated_jacobian = jacobian(estimated_3d_point, camera_matrices)
    print "Sample Error Difference: ", estimated_error
    print "Sample Jacobian Difference: ", estimated_jacobian

    # Determine the best nonlinear estimate of a 3D point
    estimated_3d_point_linear = linear_estimate_3d_point(
        points[0,:,:], camera_matrices.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        points[0,:,:], camera_matrices.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, points[0,:,:], camera_matrices.copy())
    print "Linear method error:", np.linalg.norm(error_linear)
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, points[0,:,:], camera_matrices.copy())
    print "Nonlinear method error:", np.linalg.norm(error_nonlinear)

    # Determining the correct R, T from Essential Matrix
    estimated_RT, points3d = estimate_RT_from_E(E, points, K)
    print "Estimated RT:\n", estimated_RT

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # Algorithm: 1) Compute SVD of E, and Construct R from Q = UWVt or UWtVt,
    # 2)  Construct T from +ve and -ve 3rd col of U

    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    U,S,Vt = np.linalg.svd(E)
    Q1 = U.dot(W.dot(Vt))
    Q2 = U.dot(W.T.dot(Vt))

    R1 = np.linalg.det(Q1)*Q1
    R2 = np.linalg.det(Q2)*Q2

    T1 = np.array([U[:,2]]) #SAME as: np.array(U[:,2])[np.newaxis] to create 2D array from a 1D
    T2 = np.array([-U[:,2]])

    RT = np.array([np.hstack((R2,T1.T)),np.hstack((R2,T2.T)),np.hstack((R1,T1.T)),np.hstack((R1,T2.T))])
    return RT

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # Algorithm: 1) Form matrix A from observations and projection matrix, 2) Compute SVD and set point to right singular
    # vector corresponding to smallest singular value
    m, n = camera_matrices.shape[0], camera_matrices.shape[2]
    A = np.zeros((2*m, n))
    for i in range(m):
        A[i*2,:] = np.array([image_points[i,0]*camera_matrices[i,2,:] - camera_matrices[i,0,:]])
        A[i*2+1,:] = np.array([image_points[i,1]*camera_matrices[i,2,:] - camera_matrices[i,1,:]])

    U,S,Vt = np.linalg.svd(A)
    point = Vt[-1,:]
    point /= point[-1]
    return point[:3]

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # Algorithm:
    m = camera_matrices.shape[0]
    error = np.zeros(2*m)
    for i in range(m):
        p_reproj = camera_matrices[i,:,:].dot(np.append(point_3d,1))
        p_reproj /= p_reproj[-1]
        error[i*2:i*2+2] = p_reproj[:2] - image_points[i,:]

    return error

'''
JACOBIAN given a 3D point, compute the associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # Algorithm: 1) Loop over all matrices and Compute Image point, 2) Compute x & y components of each row of Jacobian
    # using the reprojected image point
    m = camera_matrices.shape[0]
    J = np.zeros((2*m,3))
    for i in range(m):
        p_reproj = camera_matrices[i,:,:].dot(np.append(point_3d,1))
        J[i*2,:] = np.array([(p_reproj[2]*camera_matrices[i,0,:3] - p_reproj[0]*camera_matrices[i,2,:3])/p_reproj[2]**2])
        J[i*2+1,:] = np.array([(p_reproj[2]*camera_matrices[i,1,:3] - p_reproj[1]*camera_matrices[i,2,:3])/p_reproj[2]**2])

    return J

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # Algorithm: 1) Compute Initial linear estimate of 3D point, 2) Loop over Number of iterations, 3) Compute jacobian,
    # and reprojection error of computed point over all image / camera pairs, 4) Compute new estimate of 3D point
    point = linear_estimate_3d_point(image_points, camera_matrices)
    for i in range(10):
        J = jacobian(point, camera_matrices)
        e = reprojection_error(point, image_points, camera_matrices)
        point = point - np.linalg.inv(J.T.dot(J)).dot(J.T).dot(e)
    return point

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # PART e) TODO: Implement this method!
    # Algorithm: 1) Get RT pairs from E, 2) For each RT pair & K (get camera matrix), and For each N sets of image points,
    # Compute 3D point and store z value, 3) Transform 3D point by R,T into 2nd camera frame, and store z-value,
    # 4) Find number of positive Nx2 z values for each R,T pair, 5) Return the R,T matrix with the most pos z count
    RT_pairs = estimate_initial_RT(E)

    num_z_pos_points = np.zeros(4)
    points = np.zeros((4,image_points.shape[0],3))
    for i in range(4):
        RT = RT_pairs[i,:,:]
        camera_matrices = np.array([K.dot(np.eye(3,4)), K.dot(RT)])

        z_points = np.zeros(image_points.shape[0]*2)

        for j in range(image_points.shape[0]):
            point = nonlinear_estimate_3d_point(image_points[j,:,:], camera_matrices)
            point2 = RT.dot(np.append(point,1))

            z_points[j*2] = point[2]
            z_points[j*2+1] = point2[2]
            points[i,j,:] = point

        num_z_pos_points[i] = np.where(z_points>0)[0].size

    best_RT_index = np.argmax(num_z_pos_points)
    points = np.squeeze(points[best_RT_index,:,:])
    RT = RT_pairs[best_RT_index]
    return RT, points

if __name__ == '__main__':
    main()
