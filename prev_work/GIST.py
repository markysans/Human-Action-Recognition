import cv2
import numpy as np
import pandas as pd
from sparse_filtering import SparseFiltering


cap = cv2.VideoCapture('/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/dataset/boxing/person01_boxing_d3_uncomp.avi')

first_frame = None
d = np.zeros((0, 432))

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Original', frame)

        # pre-processing

        # background subtraction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh_delta = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1]
        fgmask = cv2.dilate(thresh_delta, None, iterations=0)

        cv2.imshow('bgframe', cv2.resize(fgmask, (120 * 5, 60 * 5)))

        # Complement
        comp_img = cv2.bitwise_not(frame)

        height, width = fgmask.shape

        # Leftmost point
        d1 = []
        for i in range(height):
            for j in range(1, width - 1):
                if (fgmask[i, j] == 0) and (fgmask[i, j + 1]):
                    d1.append(j)
        if len(d1):
            p1 = min(d1)
        else:
            continue

        # Rightmost point
        d2 = []
        for i in range(1, height):
            for j in range(width - 1, 1, -1):
                if (fgmask[i, j] == 0) and (fgmask[i, j - 1]):
                    d2.append(j)
        if len(d2):
            p2 = max(d2)
        else:
            continue

        # Topmost point
        d3 = []
        for i in range(1, width - 1):
            for j in range(1, height):
                if (fgmask[j, i] == 0) and (fgmask[j, i + 1]):
                    d3.append(j)
        if len(d3):
            p3 = min(d3)
        else:
            continue

        # Bottommost point
        d4 = []
        for i in range(2, width):
            for j in range(height - 1, 1, -1):
                if (fgmask[j, i] == 0) and (fgmask[j, i - 1]):
                    d4.append(j)
        if len(d4):
            p4 = max(d4)
        else:
            continue

        # Crop
        crop_img = fgmask[p3 - 10:p4 + 10, p1 - 10:p2 + 10]

        # Resize
        resized_img = cv2.resize(crop_img, (120, 60), interpolation=cv2.INTER_AREA)

        # GIST
        descriptor = gist.extract(resized_img)
        print(descriptor.shape)

        cv2.imshow('frame', cv2.resize(resized_img, (60 * 5, 120 * 5)))
    else:
        break

    k = cv2.waitKey(30) & 0xff
    if k == ord('q') or k == 27:
        break

# print(d.shape)
#
# # Sparse Filtering
# n_features = 2000   # How many features are learned
# estimator = SparseFiltering(n_features=n_features,
#                             maxfun=200,  # The maximal number of evaluations of the objective function
#                             iprint=100)  # after how many function evaluations is information printed
#
# features = estimator.fit_transform(h1)  # by L-BFGS. -1 for no information

# print(estimator.w_.shape)
# print(features.shape)

# data = pd.DataFrame(features)
# data.to_csv('/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/KTH/hog.csv', mode='a',
#               index=False, header=False)

cap.release()
cv2.destroyAllWindows()
