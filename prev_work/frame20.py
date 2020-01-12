import cv2
import os


path = '/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/dataset/jogging'
for filename in sorted(os.listdir(path)):
    file_path = path + '/' + filename

    cap = cv2.VideoCapture(file_path)
    for i in range(1, 10):
        success, frame = cap.read()

    success, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    gray = gray[15:height - 15, 15:width - 15]
    gray_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1]
    mean_frame1 = cv2.mean(gray_th)[0]

    img_array = []

    while success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        gray = gray[15:height - 15, 15:width - 15]
        gray_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1]
        mean = cv2.mean(gray_th)[0]

        if abs(mean-mean_frame1) < 10:
            # Background Subtraction
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            ret, bgframe = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

            # Bounded Box
            height, width = bgframe.shape
            # Leftmost point
            d1 = []
            for i in range(height):
                for j in range(1, width - 1):
                    if (bgframe[i, j] == 0) and (bgframe[i, j + 1] == 0):
                        d1.append(j)
            if len(d1):
                p1 = min(d1)
            else:
                success, frame = cap.read()
                continue

            # Rightmost point
            d2 = []
            for i in range(1, height):
                for j in range(width - 1, 1, -1):
                    if (bgframe[i, j] == 0) and (bgframe[i, j - 1] == 0):
                        d2.append(j)
            if len(d2):
                p2 = max(d2)
            else:
                success, frame = cap.read()
                continue

            # Topmost point
            d3 = []
            for i in range(1, width - 1):
                for j in range(1, height):
                    if (bgframe[j, i] == 0) and (bgframe[j, i + 1] == 0):
                        d3.append(j)
            if len(d3):
                p3 = min(d3)
            else:
                success, frame = cap.read()
                continue

            # Bottommost point
            d4 = []
            for i in range(2, width):
                for j in range(height - 1, 1, -1):
                    if (bgframe[j, i] == 0) and (bgframe[j, i - 1] == 0):
                        d4.append(j)
            if len(d4):
                p4 = max(d4)
            else:
                success, frame = cap.read()
                continue

            # Crop
            crop_img = bgframe[p3:p4, p1:p2]

            # Resize
            if crop_img.size != 0:
                resized_img = cv2.resize(crop_img, (64, 128))
                output_frame = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB).copy()
                img_array.append(output_frame)
                # cv2.imshow('bgframe', output_frame)

        mean_frame1 = min(mean_frame1, mean)
        # cv2.imshow('Original', frame)

        success, frame = cap.read()

        k = cv2.waitKey(0) & 0xff
        if k == ord('q') or k == 27:
            break
    if len(img_array) >= 20:
        print(len(img_array))
        gap = len(img_array)/20
        out = cv2.VideoWriter('/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/NewData/jogging/'+filename, 0, 25, (64, 128))
        for i in range(0, len(img_array), int(gap)):
            out.write(img_array[i])
    else:
        print(filename)

    cap.release()
    cv2.destroyAllWindows()
