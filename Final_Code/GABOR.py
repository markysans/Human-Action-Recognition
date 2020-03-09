import cv2
import os
import numpy as np
import pandas as pd
from sparse_filtering import SparseFiltering
from skimage import feature
from skimage.filters import gabor


path = '/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/NewData/walking'
for filename in sorted(os.listdir(path)):
    file_path = path + '/' + filename
    cap = cv2.VideoCapture(file_path)

    first_frame = None
    d = np.zeros((0, 882))

    while True:
        ret, frame = cap.read()
        if ret:
            # cv2.imshow('Original', frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Local Binary Pattern


            # print(lbp.shape)  #(128, 64)
            f = lbp.reshape((1, lbp.shape[0]*lbp.shape[1]))
            # print(f.shape)   # (1, 8192)
            d = np.concatenate((d, f), axis=0)

            # cv2.imshow('frame', cv2.resize(frame, (60 * 5, 120 * 5)))
        else:
            break

        k = cv2.waitKey(0) & 0xff
        if k == ord('q') or k == 27:
            break

    # first 20 frames feature
    d = d[0:20, :]

    # # Sparse Filtering
    # # n_features = 50   # How many features are learned
    # # estimator = SparseFiltering(n_features=n_features,
    # #                             maxfun=100,  # The maximal number of evaluations of the objective function
    # #                             iprint=100)  # after how many function evaluations is information printed
    # #
    # # features = estimator.fit_transform(d)  # by L-BFGS. -1 for no information
    features = d.reshape((1, d.shape[0]*d.shape[1]))
    features = np.append(features, [['F']], axis=1)

    data = pd.DataFrame(features)
    data.to_csv('/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/KTH/lbp.csv', mode='a',
                index=False, header=False)
    #
    # print(d.shape)
    # print(estimator.w_.shape)
    # print(features.shape)

    cap.release()
    cv2.destroyAllWindows()
