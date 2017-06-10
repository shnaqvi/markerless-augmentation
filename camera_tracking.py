import numpy as np
import cv2
from compute_camera_pose import *

def main():
    cap = cv2.VideoCapture('images/palace.m4v')

    if(not cap.isOpened()):
        raise Exception("Can't Open File")
    ret, frame = cap.read()
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = np.zeros((num_frames,)+frame.shape, dtype=frame.dtype)

    for i in range(num_frames):
        ret, frame = cap.read()

        if frame is not None:
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv2.imshow('frame',gray)
            if cv2.waitKey(1) == 27 or cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Add to frames array
            frames[i,:,:,:] = frame

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    camera_matrices = np.empty((0,3,4))
    for i in range(1,num_frames):
        im1 = frames[i-1,:,:,:]
        im2 = frames[i,:,:,:]

        # Compute Features, Find Correspondence & Refine Matches
        kp1, des1, kp2, des2 = compute_correspondence(im1, im2, 0)
        points1, points2 = match_correspondence(im1, im2, kp1, des1, kp2, des2, 0)

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
        camera_matrices = np.vstack((camera_matrices, estimated_RT[np.newaxis,:,:]))

    pass

if __name__ == '__main__':
    main()
