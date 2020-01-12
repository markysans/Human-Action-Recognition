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
    d = np.zeros((0, 640))

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Original', frame)

            # SURF
            surf = cv2.xfeatures2d.SURF_create(10)
            (kps, descs) = surf.detectAndCompute(frame, None)
            # print(descs.shape)
            if len(kps):
                ff = descs[0:10]
                f = np.zeros((10, 64))
                f[:ff.shape[0], :ff.shape[1]] = ff[:, :]
                f = f.reshape((1, f.shape[0]*f.shape[1]))
                d = np.concatenate((d, f), axis=0)
            else:
                f = np.zeros((1, 640))
                d = np.concatenate((d, f), axis=0)
            # Visualise
            resized_img1 = cv2.drawKeypoints(frame, kps, frame)
            cv2.imshow('frame', resized_img1)

        else:
            break

        k = cv2.waitKey(30) & 0xff
        if k == ord('q') or k == 27:
            break

    # first 20 frames feature
    d = d[0:20, :]

    # Sparse Filtering
    n_features = 20   # How many features are learned
    estimator = SparseFiltering(n_features=n_features,
                                maxfun=200,  # The maximal number of evaluations of the objective function
                                iprint=100)  # after how many function evaluations is information printed
    features = estimator.fit_transform(d)  # by L-BFGS. -1 for no information
    features = features.reshape((1, features.shape[0]*features.shape[1]))
    # features = np.append(features, [['A']], axis=1)

    data = pd.DataFrame(features)
    data.to_csv('/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/KTH/surf.csv', mode='a',
                index=False, header=False)

    print(d.shape)
    print(estimator.w_.shape)
    print(features.shape)

    cap.release()
    cv2.destroyAllWindows()
