import numpy as np
import lmdb
import caffe
import matplotlib.pyplot as plt

env = lmdb.open('train_data_lmdb', readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000000')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flat_x = np.fromstring(datum.data, dtype=np.uint8)
x = flat_x.reshape(datum.height, datum.width, datum.channels)
y = datum.label

plt.imshow(x)
plt.savefig('test.png')
plt.show()
print x, np.shape(x), y