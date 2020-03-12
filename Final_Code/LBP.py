import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
from sparse_filtering import SparseFiltering

path = '/home/dolan/PycharmProjects/HAR/after_bremoval/Boxing'
for filename in sorted(os.listdir(path)):
    file_path = path + '/' + filename
    cap = cv2.VideoCapture(file_path)

    first_frame = None
    d = np.zeros((0, 26))

    X_test = []

    while True:
        ret, frame = cap.read()
        if ret:
            # cv2.imshow('Original', frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #local_binary_pattern

            radius = 3 #no of points to be considered as neighbours
            no_points = 8 * radius
            lbp = local_binary_pattern(frame, no_points, radius, method='uniform')
            cv2.imshow('Local Binary Image', lbp)


            #calculate the histogram
            (x, _) = np.histogram(lbp.ravel(), bins=np.arange(0, no_points + 3),
                                     range=(0, no_points + 2))
            # print(x)
            # print(x.shape) (26, )

            #normalize the histogram

            x = [x]
            x = np.array(x)
            print(x.shape)
            x = x.astype("float")
            x /= (x.sum() + 1e-7)
            # print(x)
            # np.reshape(x,(1,26))
            # # print(f.shape)
            d = np.concatenate([d, x], axis=0)
            print(d.shape)
            # # print(f.shape, d.shape)  # size is (1, 756), (n, 756)
            #
            # # cv2.imshow('frame', cv2.resize(frame, (60 * 5, 120 * 5)))
        else:
            break

        k = cv2.waitKey(100) & 0xff
        if k == ord('q') or k == 27:
            break

    # first 20 frames feature
    d = d[0:20, :]
    features = d.reshape(1, 520)
    print(features.shape)

    features = np.append(features, [['F']], axis=1)
    #
    data = pd.DataFrame(features)
    data.to_csv('/home/dolan/PycharmProjects/HAR/features/test.csv', mode='a',
                index=False, header=False)
    #
    # print(d.shape)
    # print(estimator.w_.shape)
    # print(features.shape)

    cap.release()
    cv2.destroyAllWindows()
