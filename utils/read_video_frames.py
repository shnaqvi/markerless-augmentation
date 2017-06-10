import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture('palace.m4v')
    if(not cap.isOpened()):
        raise Exception("Can't Open File")
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_secs = [0,5]
    frames = np.zeros(frame.shape+(len(frame_secs),), dtype=frame.dtype)
    for i in range(len(frame_secs)):
        frame_id = frame_secs[i]*fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if (frame is not None):
            frames[:,:,:,i] = frame


    gray = cv2.cvtColor(frames[:,:,:,0], cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    cv2.waitKey(0)

    cv2.imwrite('palace.jpg', frames[:,:,:,0])
    cv2.imwrite('palace1.jpg', frames[:,:,:,1])
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception as inst:
        print inst
