import cv2
import numpy as np
import pandas as pd
from sparse_filtering import SparseFiltering
from skimage import feature


cap = cv2.VideoCapture('/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/NewData/jogging/person01_jogging_d1_uncomp.avi')

first_frame = None
d = np.zeros((0, 72))

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Original', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # GLCM
        Grauwertmatrix = feature.greycomatrix(frame, [1, 2, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                              symmetric=False, normed=True)
        contrast = feature.greycoprops(Grauwertmatrix, 'contrast')
        dissimilarity = feature.greycoprops(Grauwertmatrix, 'dissimilarity')
        homogeneity = feature.greycoprops(Grauwertmatrix, 'homogeneity')
        energy = feature.greycoprops(Grauwertmatrix, 'energy')
        correlation = feature.greycoprops(Grauwertmatrix, 'correlation')
        ASM = feature.greycoprops(Grauwertmatrix, 'ASM')

        f = np.array([contrast, dissimilarity, homogeneity, energy, correlation, ASM])
        f = f.reshape((1, f.shape[0]*f.shape[1]*f.shape[2]))
        print(f.shape)
        d = np.concatenate((d, f), axis=0)

        cv2.imshow('frame', cv2.resize(frame, (60 * 5, 120 * 5)))
    else:
        break

    k = cv2.waitKey(30) & 0xff
    if k == ord('q') or k == 27:
        break

# first 20 frames feature
d = d[0:20, :]

# Sparse Filtering
n_features = 50   # How many features are learned
estimator = SparseFiltering(n_features=n_features,
                            maxfun=100,  # The maximal number of evaluations of the objective function
                            iprint=100)  # after how many function evaluations is information printed

features = estimator.fit_transform(d)  # by L-BFGS. -1 for no information
features = features.reshape((1, features.shape[0]*features.shape[1]))
features = np.append(features, [['A']], axis=1)

data = pd.DataFrame(features)
data.to_csv('/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/KTH/glcm.csv', mode='a',
            index=False, header=False)

print(d.shape)
print(estimator.w_.shape)
print(features.shape)

cap.release()
cv2.destroyAllWindows()
