import numpy as np
import cv2

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

if __name__ == '__main__':
    try:
        main()
    except Exception as inst:
        print inst
