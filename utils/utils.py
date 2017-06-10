import cv2
import cv2.aruco as aruco

def cvDrawWindow(name, im, resize_factor, wait=0):
    im_resz = tuple([int(elem * resize_factor) for elem in im.shape[:2]])
    cv2.resizeWindow(name, im_resz[1], im_resz[0])
    im = cv2.resize(im, (im_resz[1], im_resz[0]))
    cv2.imshow(name, im)
    if wait:
        cv2.waitKey(0)


def getMarker():
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)     # print (aruco_dict)
    img = aruco.drawMarker(aruco_dict, 2, 700)
    return img

def detectMarker(im):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    aruco_params = aruco.DetectorParameters_create()
    corners, ids, rejectedPts = aruco.detectMarkers(im, aruco_dict, parameters=aruco_params)
    return corners, ids, rejectedPts
