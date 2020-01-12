import cv2
import os
import numpy as np

path = '/dataset/running'
for filename in sorted(os.listdir(path)):
    file_path = path + '/' + filename
    first_frame = None
    success = True

    cap = cv2.VideoCapture(file_path)
    success, frame = cap.read()
    while success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if first_frame is None:
            first_frame = gray
            continue

        thresh_delta = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1]
        # thresh_delta = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
        thresh_delta = cv2.dilate(thresh_delta, None, iterations=0)
        cont_image = cv2.bitwise_not(thresh_delta.copy())
        (cnts, _) = cv2.findContours(cont_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # for contour in cnts:
        #     if cv2.contourArea(contour) < 400:
        #         continue
        #     (x, y, w, h) = cv2.boundingRect(contour)
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        crop_img = np.zeros((60, 120))

        if len(cnts) != 0:
            # cv2.drawContours(frame, cnts, -1, 255, 3)q
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # crop_img = frame[y-10:y+h+10, x-10:x+w+10]
            # if x!=0 and y!=0 and w!=0 and h!=0:
            #     crop_img = cv2.resize(frame[y-10:y+h+10, x-10:x+w+10], (60, 120), interpolation=cv2.INTER_AREA)

        cv2.imshow("thresh", thresh_delta)
        cv2.imshow("Original", frame)
        cv2.imshow("crop", crop_img)

        key = cv2.waitKey(30) & 0xff
        if key == ord('q') or key == 27:
            break
        success, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()