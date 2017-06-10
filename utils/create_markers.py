import numpy as np
import cv2
import cv2.aruco as aruco

def getMarker():
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)     # print (aruco_dict)
    img = aruco.drawMarker(aruco_dict, 2, 700)
    return img

def main():
    # help(cv2.aruco)
    img = getMarker()

    cv2.imshow('frame',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("test_marker.jpg", img)

if __name__ == '__main__':
    main()


