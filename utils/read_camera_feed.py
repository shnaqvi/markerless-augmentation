import cv2

from utils.utils import *


def main():
    cap = cv2.VideoCapture(0)
    if (not cap.isOpened):
        raise Exception("camera not initalized")
    while(cap.isOpened()):
        ret, im = cap.read()
        if (im is not None):
            cvDrawWindow('im',im, 1)
            if cv2.waitKey(1)==27:
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except Exception as inst:
        print inst

