import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

dict = unpickle('data_batch_1')
tr_N = len(dict['data'])
X = np.zeros((tr_N, 32, 32, 3), dtype=np.uint8)
y = np.zeros(tr_N, dtype=np.int64)

for i in range(tr_N):
    X[i] = dict['data'][i].reshape((32, 32, 3), order = 'F')
    y[i] = dict['labels'][i]

print(X)
print(y)
plt.imshow(X[0])
plt.show()