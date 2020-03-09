import cv2
import os


path = '/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/dataset/jogging'
for filename in sorted(os.listdir(path)):
    file_path = path + '/' + filename

    cap = cv2.VideoCapture(file_path)

    success, frame = cap.read()

    while success:
        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgray = cv2.GaussianBlur(imgray, (5, 5), 1)

        ret, thresh = cv2.threshold(imgray, 120, 255, cv2.THRESH_TOZERO)
        ret1, thresh1 = cv2.threshold(imgray, 120, 255, cv2.THRESH_TOZERO_INV)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgray, contours, -1, (0, 255, 0), 1)

        cv2.imshow("Original", cv2.resize(thresh, (imgray.shape[1] * 5, imgray.shape[0] * 5)))
        cv2.imshow("Original1", cv2.resize(thresh1, (imgray.shape[1] * 5, imgray.shape[0] * 5)))
        success, frame = cap.read()

        k = cv2.waitKey(30) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
