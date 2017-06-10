import cv2.aruco as aruco
import numpy as np
from utils.utils import *


def main():
    # Read Built-in Camera Feed
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, im = cap.read()
        if (im is not None):

            # im = cv2.imread('images/marker.jpg', cv2.IMREAD_COLOR)
            marker_sz = 11.1    #cm (same units used to report tvecs)
            camera_matrix = np.array([[1776,0,762],[0,1780,1025],[0,0,1]],dtype=float)  #cx,cy ~= im.shape[1],im.shape[0]
            dist_coeffs = np.array([[0,0,0,0]],dtype=float) #need to be float type

            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            aruco_params = aruco.DetectorParameters_create()
            corners, ids, rejectedPts = aruco.detectMarkers(im, aruco_dict, parameters=aruco_params) #print np.squeeze(corners[0]).shape

            im = aruco.drawDetectedMarkers(im, corners, ids=ids, borderColor=1)

            rvecs, tvecs = aruco.estimatePoseSingleMarkers(corners, marker_sz, camera_matrix, dist_coeffs); #print rvecs
            if rvecs is not None:
                im = aruco.drawAxis(im, camera_matrix, dist_coeffs, rvecs, tvecs, marker_sz)

            cvDrawWindow('im',im, 1)
            if cv2.waitKey(1)==27:
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
