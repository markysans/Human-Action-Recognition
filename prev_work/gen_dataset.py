import cv2
import os
import math


path = '/dataset/handclapping'
for filename in sorted(os.listdir(path)):
    file_path = path + '/' + filename

    cap = cv2.VideoCapture(file_path)
    success, frame = cap.read()

    no_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    multiplier = int(math.floor(no_of_frames/40))
    img_array = []

    while success:
        frameId = int(round(cap.get(1)))

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # thresh_delta = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1]
        # fgmask = cv2.dilate(thresh_delta, None, iterations=0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        threshold = 50
        max_val = 255
        ret, fgmask = cv2.threshold(blur, threshold, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if frameId % multiplier == 0 and frameId != 0:
            cv2.imshow('bgframe', cv2.resize(fgmask, (120 * 5, 60 * 5)))
            bgr_frame = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
            img_array.append(bgr_frame)

        cv2.imshow('Original', frame)
        success, frame = cap.read()

        k = cv2.waitKey(30) & 0xff
        if k == ord('q') or k == 27:
            break

    # print(len(img_array))
    # out = cv2.VideoWriter('Dataset/handclapping/'+filename, 0, 25, (160, 120))
    # for i in range(len(img_array)):
    #     out.write(img_array[i])

    cap.release()
    cv2.destroyAllWindows()