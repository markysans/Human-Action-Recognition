import cv2
import os


path = '/MyWork/Dataset'
for filename in os.listdir(path):
    file_path = path + '/' + filename
    cap = cv2.VideoCapture(file_path)

    success, frame = cap.read()

    no_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(no_of_frames, fps)

    first_frame = None
    i = 1
    while success:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # thresh_delta = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1]
        # fgmask = cv2.dilate(thresh_delta, None, iterations=0)
        #
        # cv2.imshow('bgframe', cv2.resize(fgmask, (120 * 5, 60 * 5)))

        cv2.imshow('Original', frame)
        success, frame = cap.read()

        k = cv2.waitKey(30) & 0xff
        if k == ord('q') or k == 27:
            break
        i = i + 1
    cap.release()
    cv2.destroyAllWindows()
