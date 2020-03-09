import cv2
import os
import numpy as np
import pandas as pd
from sparse_filtering import SparseFiltering

path = '/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/NewData/walking'
for filename in sorted(os.listdir(path)):
    file_path = path + '/' + filename
    cap = cv2.VideoCapture(file_path)

    first_frame = None
    d = np.zeros((0, 108))

    while True:
        ret, frame = cap.read()
        if ret:
            # cv2.imshow('Original', frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # hog
            cell_size = (32, 32)  # h x w in pixels
            block_size = (2, 2)  # h x w in cells
            nbins = 9  # number of orientation bins

            # winSize is the size of the image cropped to an multiple of the cell size
            # cell_size is the size of the cells of the img patch over which to calculate the histograms
            # block_size is the number of cells which fit in the patch
            hog = cv2.HOGDescriptor(_winSize=(frame.shape[1] // cell_size[1] * cell_size[1],
                                              frame.shape[0] // cell_size[0] * cell_size[0]),
                                    _blockSize=(block_size[1] * cell_size[1],
                                                block_size[0] * cell_size[0]),
                                    _blockStride=(cell_size[1], cell_size[0]),
                                    _cellSize=(cell_size[1], cell_size[0]),
                                    _nbins=nbins)
            f = hog.compute(frame).T
            # print(f.shape)
            d = np.concatenate((d, f), axis=0)
            # print(f.shape, d.shape)  # size is (1, 756), (n, 756)

            # cv2.imshow('frame', cv2.resize(frame, (60 * 5, 120 * 5)))
        else:
            break

        k = cv2.waitKey(10) & 0xff
        if k == ord('q') or k == 27:
            break

    # first 20 frames feature
    d = d[0:20, :]
    features = d.reshape(1, 2160)
    print(features.shape)

    # Sparse Filtering
    # n_features = 100   # How many features are learned
    # estimator = SparseFiltering(n_features=n_features,
    #                             maxfun=100,  # The maximal number of evaluations of the objective function
    #                             iprint=10)  # after how many function evaluations is information printed
    # features = estimator.fit_transform(d)  # by L-BFGS. -1 for no information
    # features = features.reshape((1, features.shape[0]*features.shape[1]))
    features = np.append(features, [['F']], axis=1)
    #
    data = pd.DataFrame(features)
    data.to_csv('/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/KTH/hog1.csv', mode='a',
                index=False, header=False)
    #
    # print(d.shape)
    # print(estimator.w_.shape)
    # print(features.shape)

    cap.release()
    cv2.destroyAllWindows()
