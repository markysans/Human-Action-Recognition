import pandas as pd
import numpy as np
import glob

li = []
df1 = pd.read_csv("/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/KTH/lbp.csv", header=None)

# df2 = pd.read_csv("/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/KTH/surf.csv", header=None)

# df3 = pd.read_csv("/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/KTH/hog1.csv", header=None)

# print(df1.shape)
# print(df2.shape)
df = df1.reindex(np.random.permutation(df1.index))
# df = df3.sample(frac=1)
# print(df)


# df4 = pd.concat([df1, df2, df3], axis=1)
header = [[]]
for i in range(1, df.shape[1]):
    header[0].append('attr' + str(i))
header[0].append('class')
header = pd.DataFrame(header)

# print(df4.shape)
# print(header.shape)

header.to_csv('/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/KTH/outputLBP.csv', mode='a', index=False, header=False)
df.to_csv('/home/arijitiiest/Desktop/Workspace/Project/Human Action Recognition/MyWork/KTH/outputLBP.csv', mode='a', index=False, header=False)





# df4.to_csv('/home/dolan/PycharmProjects/HAR/features/output.csv', index=False)


# df1.columns = df2.columns
# df3 = pd.concat([df2, df1], axis=1)
# print(li)
# df3.to_csv('/home/dolan/PycharmProjects/HAR/features/output.csv', index=False)

# header.to_csv('/home/dolan/PycharmProjects/HAR/features/o.csv', mode='a', index=False, header=False)
# print(df3)

